"""biPangolin runner: per-base donor/acceptor predictions from frozen Pangolin
representations, ensembled across replicates and tissues.

Quick start:
    from bipangolin import BiPangolinRunner
    runner = BiPangolinRunner()                              # auto-downloads weights
    result = runner.score_sequence("ACGT" * 500)             # short sequence
    result = runner.score_region("hg38.fa", "chr19", 13.2e6, 13.3e6)  # long region

Outputs are P(none), P(acceptor), P(donor) at each base, plus Pangolin's
P(spliced) and PSI per tissue.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored Pangolin architecture (no external pangolin package needed)
from .model import Pangolin, L, W, AR
from ._weights import resolve_pangolin_weights, resolve_probe_weights

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_LEN = 20000
PANGOLIN_CROP = 5000
USABLE_LEN = WINDOW_LEN - 2 * PANGOLIN_CROP   # 10000

TISSUE_NAMES = ("heart", "liver", "brain", "testis")
PROB_CHANNEL_MAP = [1, 4, 7, 10]    # P(spliced) per tissue
PSI_CHANNEL_MAP = [2, 5, 8, 11]     # PSI per tissue
NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2

# Pangolin uses the 12 even-indexed model files (3 replicates × 4 tissues)
PANGOLIN_FILE_RE = re.compile(r"^final\.([1-3])\.([0246])\.3\.v2$")

_BASE_TO_IDX = {b: i for i, b in enumerate("NACGT")}
_IN_MAP = torch.tensor([
    [0, 0, 0, 0],   # N
    [1, 0, 0, 0],   # A
    [0, 1, 0, 0],   # C
    [0, 0, 1, 0],   # G
    [0, 0, 0, 1],   # T
], dtype=torch.float32)


def one_hot_encode(seq: str) -> torch.Tensor:
    """ACGT/N string -> (4, len) float tensor."""
    idx = torch.tensor([_BASE_TO_IDX.get(b.upper(), 0) for b in seq], dtype=torch.long)
    return _IN_MAP[idx].T.contiguous()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BiPangolinResult:
    """Per-base predictions for a sequence or region.

    All tensors are length L (the input sequence length).
    """
    probe_none: torch.Tensor               # (L,) P(not a splice site)
    probe_acceptor: torch.Tensor           # (L,) P(acceptor)
    probe_donor: torch.Tensor              # (L,) P(donor)
    pangolin_prob: torch.Tensor            # (n_tissues, L) P(spliced) per tissue
    pangolin_psi: torch.Tensor             # (n_tissues, L) PSI per tissue
    tissues: tuple                         # tissue names matching pangolin_* rows
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.probe_none.shape[0]


# ---------------------------------------------------------------------------
# Probe builder (mirrors training-time make_probe)
# ---------------------------------------------------------------------------

def _build_probe(in_channels: int, kernel_size: int,
                 hidden_dim: Optional[int]) -> nn.Module:
    pad = kernel_size // 2
    if hidden_dim:
        return nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 3, kernel_size, padding=pad),
        )
    return nn.Conv1d(in_channels, 3, kernel_size, padding=pad)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BiPangolinRunner:
    """Score sequences with the biPangolin probe ensemble.

    Args:
        pangolin_model_dir: Path to directory of Pangolin .v2 weight files.
            If None, auto-downloads to ~/.cache/bipangolin/.
        probe_dir: Path to directory of trained probe .pt files.
            If None, uses bundled probes shipped with the package.
        device: "auto", "cuda", "cpu", or a torch.device.
        ensemble: If True, average across all matching model+probe pairs.
            If False, use only the first pair (faster, less accurate).
        tissue: One of "all_tissues" (default), or a specific tissue name
            from TISSUE_NAMES. The probe predictions are unaffected (the
            probe is tissue-agnostic), but Pangolin outputs are filtered.
    """

    def __init__(self,
                 pangolin_model_dir: Optional[Union[str, Path]] = None,
                 probe_dir: Optional[Union[str, Path]] = None,
                 device: Union[str, torch.device] = "auto",
                 ensemble: bool = True,
                 tissue: str = "all_tissues",
                 correction_k: Optional[float] = None,
                 correction_file: Optional[Union[str, Path]] = None):
        self.pangolin_model_dir = Path(pangolin_model_dir) if pangolin_model_dir \
                                   else resolve_pangolin_weights()
        self.probe_dir = Path(probe_dir) if probe_dir \
                         else resolve_probe_weights()

        if not self.pangolin_model_dir.exists():
            raise FileNotFoundError(
                f"Pangolin model directory not found: {self.pangolin_model_dir}")
        if not self.probe_dir.exists():
            raise FileNotFoundError(
                f"Probe directory not found: {self.probe_dir}")

        valid_tissues = ("all_tissues",) + TISSUE_NAMES
        if tissue not in valid_tissues:
            raise ValueError(f"tissue must be one of {valid_tissues}, got {tissue!r}")

        self.device = self._resolve_device(device)
        self.tissue = tissue
        self.ensemble = ensemble
        self._pair_specs = self._discover_pairs(tissue, ensemble)

        if not self._pair_specs:
            raise RuntimeError(
                f"No matching Pangolin model + probe pairs found "
                f"(tissue={tissue!r}, models={self.pangolin_model_dir}, "
                f"probes={self.probe_dir})")

        self.tissues_present = sorted({t for _, _, t in self._pair_specs})
        self.tissue_names = tuple(TISSUE_NAMES[t] for t in self.tissues_present)
        print(f"biPangolin: {len(self._pair_specs)} model+probe pairs "
              f"(tissue={tissue}) on {self.device}")

        self.correction_k = None
        self._load_correction(correction_file, correction_k)

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def _resolve_device(device):
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_correction(self, correction_file, correction_k):
        import json
        if correction_file is None:
            auto = self.probe_dir / "optimal_correction.json"
            if auto.exists():
                correction_file = auto
        if correction_file is not None:
            with open(correction_file) as f:
                self.correction_k = float(json.load(f)["recommended_k"])
            print(f"  correction k={self.correction_k:.1f} (from {Path(correction_file).name})")
        elif correction_k is not None:
            self.correction_k = float(correction_k)

    def _apply_correction(self, probe_sum: torch.Tensor) -> torch.Tensor:
        k = self.correction_k
        if k is None or k == 1.0:
            return probe_sum
        scaled = probe_sum.clone()
        scaled[NONE_CLASS] *= k
        return scaled / scaled.sum(dim=0, keepdim=True).clamp_min(1e-12)

    def _discover_pairs(self, tissue: str, ensemble: bool):
        """Build list of (pangolin_path, probe_path, tissue_idx) tuples."""
        candidates = []
        for p in sorted(self.pangolin_model_dir.glob("final.*.v2")):
            m = PANGOLIN_FILE_RE.match(p.name)
            if not m:
                continue
            t_idx = int(m.group(2)) // 2
            if tissue != "all_tissues" and TISSUE_NAMES[t_idx] != tissue:
                continue
            candidates.append((p, t_idx))

        if not ensemble:
            candidates = candidates[:1]

        pairs = []
        for p_path, t_idx in candidates:
            matches = sorted(self.probe_dir.glob(f"probe.{p_path.name}.*.pt"))
            if matches:
                pairs.append((p_path, matches[-1], t_idx))   # latest by name sort
            else:
                print(f"  warning: no probe found for {p_path.name}, skipping")
        return pairs

    # -- hooks ---------------------------------------------------------------

    @staticmethod
    def _attach_hooks(model, layers):
        handles = {}
        for layer in layers:
            cache = {}
            if layer == "skip":
                model.conv_last1.register_forward_pre_hook(
                    lambda _m, i, c=cache: c.update({"acts": i[0]}))
                is_cropped = True
            else:
                idx = int(layer.split("_")[1])
                model.resblocks[idx].register_forward_hook(
                    lambda _m, _i, o, c=cache: c.update({"acts": o}))
                is_cropped = False
            handles[layer] = {"cache": cache, "is_cropped": is_cropped}
        return handles

    # -- core forward -------------------------------------------------------

    @torch.no_grad()
    def _forward_window(self, m, probe, cfg, handles, seq_t):
        """Run one Pangolin + probe forward on a (1, 4, WINDOW_LEN) tensor.

        Returns (pangolin_out_12ch, probe_probs_3ch) both as (C, USABLE_LEN).
        """
        p_out = m(seq_t)[0]                           # (12, USABLE_LEN)

        layers = cfg["probe_layer"]
        if isinstance(layers, str):
            layers = layers.split("+")
        elif not isinstance(layers, list):
            layers = list(layers)

        acts = []
        for l_name in layers:
            val = handles[l_name]["cache"]["acts"]
            if not handles[l_name]["is_cropped"]:
                val = val[..., PANGOLIN_CROP:PANGOLIN_CROP + USABLE_LEN]
            acts.append(val)

        if cfg.get("include_sequence"):
            acts.append(seq_t[..., PANGOLIN_CROP:PANGOLIN_CROP + USABLE_LEN])

        probe_logits = probe(torch.cat(acts, dim=1))
        probe_probs = torch.softmax(probe_logits, dim=1)[0]   # (3, USABLE_LEN)

        return p_out, probe_probs

    def _iter_pairs(self):
        """Lazy generator yielding (m, probe, cfg, handles, t_idx) one at a time.

        Pangolin + probe are loaded into VRAM, yielded, then freed before the
        next pair. Caller is responsible for accumulating outputs.
        """
        for p_path, pr_path, t_idx in self._pair_specs:
            try:
                m = Pangolin(L, W, AR)
                m.load_state_dict(torch.load(p_path, map_location=self.device))
                m.to(self.device).eval()
                for p in m.parameters():
                    p.requires_grad_(False)

                blob = torch.load(pr_path, map_location=self.device)
                cfg = blob["config"]
                layers = cfg["probe_layer"]
                if isinstance(layers, str):
                    layers = layers.split("+")
                elif not isinstance(layers, list):
                    layers = list(layers)

                in_ch = 32 * len(layers) + (4 if cfg.get("include_sequence") else 0)
                probe = _build_probe(in_ch, cfg["kernel_size"], cfg.get("hidden_dim"))
                probe.load_state_dict(blob["state_dict"])
                probe.to(self.device).eval()
                for p in probe.parameters():
                    p.requires_grad_(False)

                handles = self._attach_hooks(m, layers)
                yield m, probe, cfg, handles, t_idx
            finally:
                # Always free, even if caller throws
                m = None; probe = None; handles = None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def score_sequence(self, seq: str) -> BiPangolinResult:
        """Score a sequence up to USABLE_LEN (10000) bp.

        For longer sequences, use score_long_sequence() or score_region().
        The sequence is internally padded with N x PANGOLIN_CROP on each side
        so the output is the same length as the input.
        """
        L_in = len(seq)
        if L_in == 0:
            raise ValueError("Sequence is empty")
        if L_in > USABLE_LEN:
            raise ValueError(
                f"Sequence length {L_in} exceeds single-window max {USABLE_LEN}. "
                f"Use score_long_sequence() for longer inputs.")

        # Pad to WINDOW_LEN with Ns on each side
        padded = "N" * PANGOLIN_CROP + seq + "N" * (WINDOW_LEN - PANGOLIN_CROP - L_in)
        seq_t = one_hot_encode(padded).unsqueeze(0).to(self.device)

        p_sums = {t: torch.zeros(L_in) for t in {t for _, _, t in self._pair_specs}}
        psi_sums = {t: torch.zeros(L_in) for t in p_sums}
        probe_sum = torch.zeros(3, L_in)

        for m, probe, cfg, handles, t_idx in self._iter_pairs():
            p_out, probe_probs = self._forward_window(m, probe, cfg, handles, seq_t)
            # First L_in positions of the USABLE_LEN output correspond to seq
            p_sums[t_idx]   += p_out[PROB_CHANNEL_MAP[t_idx], :L_in].cpu()
            psi_sums[t_idx] += p_out[PSI_CHANNEL_MAP[t_idx], :L_in].cpu()
            probe_sum       += probe_probs[:, :L_in].cpu()

        return self._assemble_result(p_sums, psi_sums, probe_sum, L_in,
                                     metadata={"length": L_in, "tiled": False})

    @torch.no_grad()
    def score_long_sequence(self, seq: str, overlap: int = 2000) -> BiPangolinResult:
        """Score an arbitrarily long sequence by tiling.

        Windows of WINDOW_LEN are run with USABLE_LEN-overlap stride.
        Predictions in overlap regions are linearly blended.
        """
        L_in = len(seq)
        if L_in == 0:
            raise ValueError("Sequence is empty")
        if L_in <= USABLE_LEN:
            return self.score_sequence(seq)

        if not (0 <= overlap < USABLE_LEN):
            raise ValueError(f"overlap must be in [0, {USABLE_LEN}), got {overlap}")

        stride = USABLE_LEN - overlap

        # Build window starts in usable-coordinate space, then convert to padded coords
        u_starts = list(range(0, max(1, L_in - USABLE_LEN + 1), stride))
        if u_starts[-1] + USABLE_LEN < L_in:
            u_starts.append(L_in - USABLE_LEN)

        # Triangular blending weights to soften overlap seams
        blend = self._triangular_blend(USABLE_LEN, overlap).to(self.device)

        # Pad the full input by PANGOLIN_CROP each side so any window is valid
        padded_seq = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
        padded_t = one_hot_encode(padded_seq).to(self.device)   # (4, L_in + 2*CROP)

        n_tissues_max = len(TISSUE_NAMES)
        p_sums = {t: torch.zeros(L_in) for t in {t for _, _, t in self._pair_specs}}
        psi_sums = {t: torch.zeros(L_in) for t in p_sums}
        probe_sum = torch.zeros(3, L_in)
        weight_sum = torch.zeros(L_in)
        # tissue-specific weight (some tissues only have a subset of pairs)
        tissue_weight_sum = {t: torch.zeros(L_in) for t in p_sums}

        # Outer = pairs (load each Pangolin + probe once); inner = tile windows
        for m, probe, cfg, handles, t_idx in self._iter_pairs():
            for u_start in u_starts:
                # window covers padded_t[u_start : u_start + WINDOW_LEN]
                window = padded_t[:, u_start:u_start + WINDOW_LEN].unsqueeze(0)  # (1, 4, WL)
                p_out, probe_probs = self._forward_window(m, probe, cfg, handles, window)

                # window predictions are at usable coords [u_start, u_start+USABLE_LEN)
                lo = u_start
                hi = min(u_start + USABLE_LEN, L_in)
                slice_len = hi - lo
                w = blend[:slice_len].cpu()

                p_sums[t_idx][lo:hi]   += (p_out[PROB_CHANNEL_MAP[t_idx], :slice_len].cpu() * w)
                psi_sums[t_idx][lo:hi] += (p_out[PSI_CHANNEL_MAP[t_idx], :slice_len].cpu() * w)
                probe_sum[:, lo:hi]    += (probe_probs[:, :slice_len].cpu() * w.unsqueeze(0))
                tissue_weight_sum[t_idx][lo:hi] += w
                weight_sum[lo:hi]      += w

        # Normalise: divide by accumulated weights
        for t in p_sums:
            tw = tissue_weight_sum[t].clamp_min(1e-9)
            p_sums[t]   = p_sums[t] / tw
            psi_sums[t] = psi_sums[t] / tw
        ws = weight_sum.clamp_min(1e-9).unsqueeze(0)
        probe_sum = probe_sum / ws

        # probe ensemble averaging done implicitly via the weighted sum above:
        # each tile contributes with `w` to numerator AND denominator equally
        # across pairs, so the ratio is still the per-position mean across pairs.

        return self._assemble_result(p_sums, psi_sums, probe_sum, L_in,
                                     metadata={"length": L_in, "tiled": True,
                                               "n_windows": len(u_starts),
                                               "overlap": overlap},
                                     skip_pair_normalisation=True)

    @torch.no_grad()
    def score_region(self, fasta_path: Union[str, Path], chrom: str,
                     start: int, end: int, **kwargs) -> BiPangolinResult:
        """Score a genomic region from a FASTA file.

        Coordinates are 0-based, half-open (BED style): [start, end).
        Auto-routes to score_sequence or score_long_sequence based on length.
        """
        try:
            import pyfastx
        except ImportError as e:
            raise ImportError(
                "score_region requires pyfastx: pip install pyfastx") from e
        fa = pyfastx.Fasta(str(fasta_path))
        if chrom not in fa:
            raise KeyError(f"Chromosome {chrom!r} not in FASTA "
                           f"(available: {list(fa.keys())[:5]}...)")
        start, end = int(start), int(end)
        if not (0 <= start < end <= len(fa[chrom])):
            raise ValueError(f"Bad coords [{start}, {end}) for {chrom} "
                             f"(length {len(fa[chrom])})")
        seq = fa[chrom][start:end].seq
        if len(seq) <= USABLE_LEN:
            r = self.score_sequence(seq)
        else:
            r = self.score_long_sequence(seq, **kwargs)
        r.metadata.update({"chrom": chrom, "start": start, "end": end,
                           "fasta": str(fasta_path)})
        return r

    # -- variant scoring ----------------------------------------------------

    def score_variant(self, fasta_path: Union[str, Path],
                      chrom: str, pos: int, ref: str, alt: str,
                      distance: int = 50):
        """Score a single variant's effect on splicing.

        Args:
            fasta_path: reference genome FASTA (must match the VCF's build)
            chrom, pos, ref, alt: VCF-style coordinates (pos is 1-based)
            distance: report max delta within ±distance of variant (default 50nt)

        Returns a VariantScore with DS_AG/DS_AL/DS_DG/DS_DL (probe deltas,
        tissue-agnostic) and per-tissue Pangolin P(spliced) deltas.

        Tissue selection follows whatever was set at runner init —
        pass tissue="brain" to BiPangolinRunner() for brain-only.
        """
        try:
            import pyfastx
        except ImportError as e:
            raise ImportError("score_variant requires pyfastx: pip install pyfastx") from e
        from ._variants import score_variant as _sv
        fa = pyfastx.Fasta(str(fasta_path))
        return _sv(self, fa, chrom, pos, ref, alt, distance=distance)

    def score_vcf(self, vcf_in: Union[str, Path], vcf_out: Union[str, Path],
                  fasta_path: Union[str, Path], distance: int = 50,
                  tissue_for_info: Optional[str] = None,
                  progress: bool = True) -> int:
        """Annotate a VCF with biPangolin variant effect predictions.

        Args:
            vcf_in: path to input VCF (gzipped supported via .vcf.gz)
            vcf_out: path to output annotated VCF
            fasta_path: reference genome FASTA matching the VCF build
            distance: report max delta within ±distance of each variant
            tissue_for_info: if set, only this tissue's Pangolin deltas appear
                in the INFO field. Defaults to all tissues currently loaded.
            progress: show tqdm progress bar

        Adds INFO field `biPangolin=` with format:
            ALT|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL|...

        Returns the number of variants scored successfully.
        """
        from ._variants import score_vcf as _sv
        return _sv(self, vcf_in, vcf_out, fasta_path,
                   distance=distance, tissue_for_info=tissue_for_info,
                   progress=progress)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _triangular_blend(usable_len: int, overlap: int) -> torch.Tensor:
        """Linear ramp at edges, flat 1.0 in the middle."""
        if overlap == 0:
            return torch.ones(usable_len)
        ramp = torch.linspace(0, 1, overlap + 2)[1:-1]   # excludes 0 and 1
        w = torch.ones(usable_len)
        w[:overlap] = ramp
        w[-overlap:] = ramp.flip(0)
        return w

    def _assemble_result(self, p_sums, psi_sums, probe_sum, L_in,
                         metadata=None, skip_pair_normalisation=False):
        """Average accumulated sums across the ensemble and pack into result."""
        if not skip_pair_normalisation:
            pair_t_indices = [spec[2] for spec in self._pair_specs]
            for t in p_sums:
                n_t = pair_t_indices.count(t)
                p_sums[t]   = p_sums[t] / max(n_t, 1)
                psi_sums[t] = psi_sums[t] / max(n_t, 1)
            probe_sum = probe_sum / len(self._pair_specs)

        probe_sum = self._apply_correction(probe_sum)

        return BiPangolinResult(
            probe_none=probe_sum[NONE_CLASS],
            probe_acceptor=probe_sum[ACC_CLASS],
            probe_donor=probe_sum[DON_CLASS],
            pangolin_prob=torch.stack([p_sums[t] for t in self.tissues_present]),
            pangolin_psi=torch.stack([psi_sums[t] for t in self.tissues_present]),
            tissues=self.tissue_names,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Calibration sequence (donor at 69, acceptor at 163)
# ---------------------------------------------------------------------------

CALIBRATION_SEQ = (
    "cacagcaccggcggcatggacgagctgtacaaggactacaaggacgatgatgacaagtgataaacaaatggt"
    "aaggaagggcacatcaatctttgcttaattgtcctttactctaaagatgtattttatcatactgaatgctaa"
    "acttgatatctccttttaggtcattgatgtccttcaccccgggaaggcgacagtgcctaagacagaaattcgg"
).upper()


def selftest(pangolin_model_dir=None, probe_dir=None, tissue="all_tissues"):
    """Run the calibration sequence through the runner. Should peak at don=69 acc=163."""
    runner = BiPangolinRunner(pangolin_model_dir, probe_dir, tissue=tissue)
    result = runner.score_sequence(CALIBRATION_SEQ)
    don_pos = int(result.probe_donor.argmax())
    acc_pos = int(result.probe_acceptor.argmax())
    print(f"  donor   peak: pos={don_pos:>4} (expected 69)  P={result.probe_donor.max():.3f}")
    print(f"  acceptor peak: pos={acc_pos:>4} (expected 163) P={result.probe_acceptor.max():.3f}")
    return result


def score_sequence_or_long_sequence(runner: BiPangolinRunner, seq: str) -> BiPangolinResult:
    """Module-level helper used by _variants — routes to the right method."""
    if len(seq) <= USABLE_LEN:
        return runner.score_sequence(seq)
    return runner.score_long_sequence(seq)
