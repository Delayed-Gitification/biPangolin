"""
biPangolin runner — load Pangolin + multi-layer sequence-aware probe(s) and score sequences.

Usage (in script):
    runner = BiPangolinRunner(
        pangolin_model_dir="/path/to/pangolin/models",
        probe_dir="/path/to/probes",
        device="auto",
        ensemble=True
    )
    result = runner.score_sequence(dna_string)

Usage (quick test in terminal):
    python runner.py /path/to/models /path/to/probes
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import torch
import torch.nn as nn

from .model import Pangolin, L, W, AR
from ._weights import resolve_pangolin_weights, resolve_probe_weights


# Geometry constants
PANGOLIN_CROP = 5000
WINDOW_LEN_DEFAULT = 20000   
WINDOW_LEN = WINDOW_LEN_DEFAULT
USABLE_LEN = WINDOW_LEN_DEFAULT - 2 * PANGOLIN_CROP
TILE_OVERLAP = 2000          

# Pangolin output channel mapping
PROB_CHANNEL_PER_TISSUE = [1, 4, 7, 10]   # P(spliced)
PSI_CHANNEL_PER_TISSUE  = [2, 5, 8, 11]   # PSI / usage
TISSUE_NAMES = ("heart", "liver", "brain", "testis")

# The Pangolin v2 weight files come in two flavours, fine-tuned for different
# output heads of the same multi-output architecture:
#   final.{1-3}.{0,2,4,6}.3.v2  →  P(spliced) heads — channels [1,4,7,10]
#   final.{1-3}.{1,3,5,7}.3.v2  →  usage/PSI heads — channels [2,5,8,11]
# Reading the "wrong" channel from a model gives a side-output the trunk wasn't
# tuned for, which is what Pangolin's own CLI silently does to nobody's benefit
# (it then ignores PSI entirely and reports only variant-induced ΔP).
# See discussion in Zeng & Li 2022 §"Training Pangolin".
PANGOLIN_FILE_RE     = re.compile(r"^final\.([1-3])\.([0246])\.3\.v2$")
PANGOLIN_PSI_FILE_RE = re.compile(r"^final\.([1-3])\.([1357])\.3\.v2$")
DEFAULT_CORRECTION_FILE = Path(__file__).parent / "data" / "probes" / "optimal_correction.json"

# Class encoding
NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2

# One-hot encoding
_BASE_TO_IDX = {b: i for i, b in enumerate("NACGT")}
_ALLOWED_BASES = frozenset("NACGT")
_IN_MAP = torch.tensor([
    [0, 0, 0, 0],   # N
    [1, 0, 0, 0],   # A
    [0, 1, 0, 0],   # C
    [0, 0, 1, 0],   # G
    [0, 0, 0, 1],   # T
], dtype=torch.float32)


def one_hot_encode(seq: str) -> torch.Tensor:
    """One-hot encode a DNA sequence to a (4, L) float tensor.

    Accepts RNA: U is treated as T. Case-insensitive. N encodes to all-zeros.
    Raises ValueError on any other character (IUPAC ambiguity codes,
    whitespace, gaps, digits, ...) rather than silently coercing it to N,
    which would corrupt predictions without warning.
    """
    seq = seq.upper().replace("U", "T")  # accept RNA input; U -> T
    unexpected = set(seq) - _ALLOWED_BASES
    if unexpected:
        examples = ", ".join(
            f"{ch!r} (first at index {seq.index(ch)})" for ch in sorted(unexpected)
        )
        raise ValueError(
            f"one_hot_encode received unexpected base(s): {examples}. "
            "Allowed input is A, C, G, T, U (U is read as T) and N. Check the "
            "sequence for non-DNA characters such as IUPAC ambiguity codes, "
            "whitespace, gaps ('-'), or digits."
        )
    idx = torch.tensor([_BASE_TO_IDX[b] for b in seq], dtype=torch.long)
    return _IN_MAP[idx].T.contiguous()


# ---------------------------------------------------------------------------
# Probe & Hook Logic
# ---------------------------------------------------------------------------

PROBE_LAYERS = ("skip", "resblock_15", "resblock_11", "resblock_7", "resblock_3", "resblock_1")

def parse_probe_layers(probe_layer):
    if isinstance(probe_layer, str):
        layers = probe_layer.split("+")
    else:
        layers = list(probe_layer)
    for layer in layers:
        if layer not in PROBE_LAYERS:
            raise ValueError(f"probe_layer must be from {PROBE_LAYERS}, got {layer}")
    return layers


def make_probe(kernel_size=1, hidden_dim=None, in_channels=32):
    pad = kernel_size // 2
    if hidden_dim:
        return nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 3, kernel_size, padding=pad),
        )
    return nn.Conv1d(in_channels, 3, kernel_size, padding=pad)


def load_probe(probe_path, device):
    blob = torch.load(probe_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    
    layers = parse_probe_layers(cfg["probe_layer"])
    in_channels = 32 * len(layers)
    
    # Add DNA channels if the model was trained with them
    if cfg.get("include_sequence", False):
        in_channels += 4
        
    probe = make_probe(
        kernel_size=cfg["kernel_size"], 
        hidden_dim=cfg["hidden_dim"], 
        in_channels=in_channels
    ).to(device)
    
    probe.load_state_dict(blob["state_dict"])
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe, cfg


def load_frozen_pangolin(weights_path, device):
    model = Pangolin(L, W, AR)
    map_loc = device if device.type == "cuda" else torch.device("cpu")
    model.load_state_dict(torch.load(weights_path, map_location=map_loc))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def attach_hooks(model, probe_layers):
    handles = {}
    for layer in probe_layers:
        cache = {}
        if layer == "skip":
            def hook(_module, inputs, _cache=cache):
                _cache["activations"] = inputs[0]
            model.conv_last1.register_forward_pre_hook(hook)
            is_cropped = True
        else:
            idx = int(layer.split("_")[1])
            def hook(_module, _inputs, output, _cache=cache):
                _cache["activations"] = output
            model.resblocks[idx].register_forward_hook(hook)
            is_cropped = False
        handles[layer] = {"cache": cache, "is_cropped": is_cropped}
    return handles


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

@dataclass
class BiPangolinResult:
    pangolin_prob: torch.Tensor
    probe_none: torch.Tensor
    probe_acceptor: torch.Tensor
    probe_donor: torch.Tensor
    tissues: tuple
    # Pangolin tissue-specific PSI/usage predictions, shape (n_tissues, L).
    # ONLY populated when the runner was constructed with `use_psi_models=True`,
    # which loads the PSI-tuned weight files separately. Reading the PSI
    # channel from a P-tuned model gives a misleading side-output that
    # correlates with P rather than with true PSI, so we refuse to do that
    # silently — `None` here means "we don't know, ask explicitly".
    pangolin_psi: "torch.Tensor | None" = None
    metadata: dict = field(default_factory=dict)
    # Optional, only populated when runner was constructed with
    # `per_tissue_probes=True`. Shape (3, n_tissues, L) — channel order
    # NONE / ACC / DON, tissue order matches `tissues`. Each per-tissue
    # probe is averaged across the 3 folds for that tissue only. These
    # probes are attached to the P-tuned Pangolin models.
    probe_per_tissue: "torch.Tensor | None" = None
    # Per-tissue probe outputs from probes attached to the PSI-tuned Pangolin
    # models. Same shape (3, n_tissues, L) and channel/tissue ordering as
    # `probe_per_tissue`. Populated only when the runner was constructed with
    # BOTH `use_psi_models=True` AND `per_tissue_probes=True`, AND probe files
    # for the PSI-tuned model_nums (1, 3, 5, 7) are present in probe_dir.
    # Use this when comparing probe outputs against Pangolin PSI — the
    # P-tuned-probe version mixes activations from two networks.
    probe_per_tissue_psi: "torch.Tensor | None" = None

    def __len__(self):
        return self.probe_none.shape[0]

    @property
    def raw(self):
        parts = [self.pangolin_prob]
        if self.pangolin_psi is not None:
            parts.append(self.pangolin_psi)
        parts += [
            self.probe_none.unsqueeze(0),
            self.probe_acceptor.unsqueeze(0),
            self.probe_donor.unsqueeze(0),
        ]
        return torch.cat(parts, dim=0)

    def four_track_per_tissue(self):
        """Return a 4 x n_tissues x L tissue-specific donor/acceptor matrix.

        Channel order is:
            0: donor PSI
            1: donor P(spliced)
            2: acceptor PSI
            3: acceptor P(spliced)

        Pangolin values are routed into donor or acceptor channels using the
        probe's three-way argmax. Positions classified as None remain zero.
        """
        labels = torch.stack([
            self.probe_none,
            self.probe_acceptor,
            self.probe_donor,
        ], dim=0).argmax(dim=0)

        out = torch.zeros(
            4,
            self.pangolin_prob.shape[0],
            self.pangolin_prob.shape[1],
            dtype=self.pangolin_prob.dtype,
            device=self.pangolin_prob.device,
        )
        if self.pangolin_psi is None:
            raise RuntimeError(
                "four_track_per_tissue() needs pangolin_psi but it is None. "
                "Re-run with BiPangolinRunner(use_psi_models=True) to load the "
                "PSI-tuned weight files."
            )
        donor_mask = labels == DON_CLASS
        acceptor_mask = labels == ACC_CLASS

        out[0, :, donor_mask] = self.pangolin_psi[:, donor_mask]
        out[1, :, donor_mask] = self.pangolin_prob[:, donor_mask]
        out[2, :, acceptor_mask] = self.pangolin_psi[:, acceptor_mask]
        out[3, :, acceptor_mask] = self.pangolin_prob[:, acceptor_mask]
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _mps_is_healthy() -> bool:
    """Probe whether the MPS backend handles the ops Pangolin relies on.

    PyTorch's Metal (Apple Silicon) backend has historically mis-handled
    high-dilation 1D convolutions (Pangolin uses atrous rates up to 25) and
    boolean-mask indexing (used in four_track_per_tissue) — sometimes returning
    NaNs or raising deep in the forward pass. We run a tiny representative
    computation and confirm the result is finite before trusting MPS.
    """
    try:
        dev = torch.device("mps")
        x = torch.randn(1, 4, 256, device=dev)
        conv = nn.Conv1d(4, 8, kernel_size=3, dilation=25, padding=25).to(dev)
        y = conv(x)
        mask = y[0, 0] > 0
        _ = y[..., mask]  # boolean indexing along last dim
        return bool(torch.isfinite(y).all().item())
    except Exception:
        return False


def _resolve_device(device):
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device)

    if resolved.type == "mps" and not _mps_is_healthy():
        print(
            "biPangolin: WARNING — the MPS (Apple Silicon) backend failed a "
            "health check (high-dilation conv1d / boolean indexing errored or "
            "produced NaN). Falling back to CPU.",
            file=sys.stderr,
        )
        resolved = torch.device("cpu")
    return resolved


class BiPangolinRunner:
    def __init__(self, pangolin_model_dir=None, probe_dir=None, device="auto",
                 ensemble=True, probe_tag=None, correction_k=None,
                 correction_file=None, tissue="all_tissues",
                 per_tissue_probes=False, use_psi_models=False):
        """Load Pangolin + probes for inference.

        use_psi_models: if True, additionally load Pangolin's PSI-tuned weight
            files (final.{fold}.{1,3,5,7}.3.v2) and read PSI from those.
            If False (default), pangolin_psi is left None on the result —
            because reading the PSI channel from a P-tuned model gives a
            misleading side-output that correlates with P, not real PSI.
            Costs ~2× Pangolin inference compute.
        """
        self.device = _resolve_device(device)
        self.pangolin_model_dir = (
            Path(pangolin_model_dir) if pangolin_model_dir else resolve_pangolin_weights()
        )
        self.probe_dir = Path(probe_dir) if probe_dir else resolve_probe_weights()
        self.tissue = tissue
        self.ensemble = ensemble
        self.per_tissue_probes = per_tissue_probes
        self.use_psi_models = use_psi_models
        self._pair_specs = []
        # Lazily-built caches of loaded (model, probe, handles, cfg, tissue_idx)
        # tuples. Populated on first scoring call and reused for the runner's
        # lifetime so we don't reload weights from disk on every sequence.
        self._pair_cache = None
        self._psi_cache = None
        # PSI-tuned model specs: (pangolin_path, probe_path_or_None, tissue_idx).
        # probe_path is populated only when the user has trained probes for the
        # PSI-tuned models (probes for model_nums 1,3,5,7); otherwise None and
        # we just read Pangolin PSI from the model without computing a probe.
        self._psi_specs = []
        self._psi_has_probes = False
        self.correction_k = self._resolve_correction_k(correction_k, correction_file)

        valid_tissues = ("all_tissues",) + TISSUE_NAMES
        if tissue not in valid_tissues:
            raise ValueError(f"tissue must be one of {valid_tissues}, got {tissue!r}")

        candidates = []
        psi_candidates = []
        for p in sorted(self.pangolin_model_dir.glob("final.*.v2")):
            m_p = PANGOLIN_FILE_RE.match(p.name)
            m_psi = PANGOLIN_PSI_FILE_RE.match(p.name)
            if m_p:
                tissue_idx = int(m_p.group(2)) // 2
                if tissue == "all_tissues" or TISSUE_NAMES[tissue_idx] == tissue:
                    candidates.append((p, tissue_idx))
            elif m_psi and use_psi_models:
                tissue_idx = (int(m_psi.group(2)) - 1) // 2
                if tissue == "all_tissues" or TISSUE_NAMES[tissue_idx] == tissue:
                    psi_candidates.append((p, tissue_idx))

        if not candidates:
            raise FileNotFoundError(f"No Pangolin P-tuned models found in {self.pangolin_model_dir}")
        if not ensemble:
            candidates = candidates[:1]
            psi_candidates = psi_candidates[:1]

        for pangolin_path, tissue_idx in candidates:
            pattern = f"probe.{pangolin_path.name}.*.pt"
            probe_paths = sorted(self.probe_dir.glob(pattern))
            if probe_tag is not None:
                probe_paths = [p for p in probe_paths if probe_tag in p.name]
            if not probe_paths:
                raise FileNotFoundError(f"No probe matched for {pangolin_path.name} in {self.probe_dir}")
            self._pair_specs.append((pangolin_path, probe_paths[-1], tissue_idx)) # Load latest match

        if use_psi_models:
            if not psi_candidates:
                raise FileNotFoundError(
                    f"use_psi_models=True but no PSI-tuned Pangolin models "
                    f"(final.*.[1357].3.v2) found in {self.pangolin_model_dir}"
                )
            # Defensive: drop PSI candidates for tissues we don't have probes
            # (P-tuned models) for, since the result tissue dimension follows
            # the probe set. With a full Pangolin install this filter is a no-op.
            probe_tissues = {t for _, _, t in self._pair_specs}
            psi_candidates = [(p, t) for (p, t) in psi_candidates if t in probe_tissues]
            # Try to find matching PSI-tuned probes (probes trained against
            # PSI-tuned model activations). If found, attach them; otherwise
            # store None — runner still loads the PSI-tuned model for Pangolin
            # PSI predictions, just won't compute a PSI-side probe output.
            self._psi_specs = []
            n_with_probe = 0
            for pangolin_path, tissue_idx in psi_candidates:
                pattern = f"probe.{pangolin_path.name}.*.pt"
                probe_paths = sorted(self.probe_dir.glob(pattern))
                if probe_tag is not None:
                    probe_paths = [p for p in probe_paths if probe_tag in p.name]
                probe_path = probe_paths[-1] if probe_paths else None
                if probe_path is not None:
                    n_with_probe += 1
                self._psi_specs.append((pangolin_path, probe_path, tissue_idx))
            self._psi_has_probes = (n_with_probe == len(self._psi_specs)
                                    and n_with_probe > 0)
            if n_with_probe and not self._psi_has_probes:
                print(f"biPangolin: WARNING — found PSI-tuned probes for "
                      f"{n_with_probe}/{len(self._psi_specs)} PSI-tuned models; "
                      f"PSI-side per-tissue probe outputs will not be computed "
                      f"(need probes for all to ensemble cleanly).")

        self.tissues_present = sorted({t for _, _, t in self._pair_specs})
        self.tissue_names = tuple(TISSUE_NAMES[t] for t in self.tissues_present)
        print(f"biPangolin: {len(self._pair_specs)} model+probe pairs ready on {self.device}")
        if use_psi_models:
            msg = f"biPangolin: + {len(self._psi_specs)} PSI-tuned models for PSI predictions"
            if self._psi_has_probes:
                msg += " (+ matching PSI-side probes attached)"
            print(msg)
        if self.correction_k is not None and self.correction_k != 1.0:
            print(f"biPangolin: correction k={self.correction_k:.1f}")

    def _resolve_correction_k(self, correction_k, correction_file):
        if correction_k is not None:
            return float(correction_k)

        if correction_file is not None:
            correction_path = Path(correction_file)
        else:
            probe_correction_path = self.probe_dir / "optimal_correction.json"
            correction_path = probe_correction_path if probe_correction_path.exists() else DEFAULT_CORRECTION_FILE

        if not correction_path.exists():
            return None

        with open(correction_path) as f:
            correction = json.load(f)
        try:
            return float(correction["empirical_sweep"]["best_k"])
        except KeyError as e:
            raise KeyError(
                f"{correction_path} must contain empirical_sweep.best_k "
                "for biPangolin's default correction."
            ) from e

    def _apply_correction(self, probe_probs):
        if self.correction_k is None or self.correction_k == 1.0:
            return probe_probs

        corrected = probe_probs.clone()
        corrected[NONE_CLASS] *= self.correction_k
        return corrected / corrected.sum(dim=0, keepdim=True).clamp_min(1e-12)

    def _build_pair_cache(self):
        cache = []
        for pangolin_path, probe_path, tissue_idx in self._pair_specs:
            pangolin_model = load_frozen_pangolin(pangolin_path, self.device)
            probe, cfg = load_probe(probe_path, self.device)
            layers = parse_probe_layers(cfg["probe_layer"])
            handles = attach_hooks(pangolin_model, layers)
            cache.append((pangolin_model, probe, handles, cfg, tissue_idx))
        return cache

    def _iter_pairs(self):
        # Models + probes are loaded from disk once and cached on the runner for
        # its lifetime. Previously this reloaded every weight file (and moved it
        # to the device) on *every* score_sequence() call — and the VCF path
        # calls score_sequence twice per variant — which dominated runtime at
        # scale. Hooks are attached exactly once when the cache is built;
        # re-attaching per call against a persistent model would stack them.
        if self._pair_cache is None:
            self._pair_cache = self._build_pair_cache()
        yield from self._pair_cache

    def _build_psi_cache(self):
        cache = []
        for pangolin_path, probe_path, tissue_idx in self._psi_specs:
            pangolin_model = load_frozen_pangolin(pangolin_path, self.device)
            probe = None
            handles = None
            cfg = None
            if probe_path is not None:
                probe, cfg = load_probe(probe_path, self.device)
                layers = parse_probe_layers(cfg["probe_layer"])
                handles = attach_hooks(pangolin_model, layers)
            cache.append((pangolin_model, probe, handles, cfg, tissue_idx))
        return cache

    def _iter_psi_models(self):
        """Yield (pangolin_model, probe_or_None, handles_or_None, cfg_or_None, tissue_idx)
        for the PSI-tuned Pangolin files. probe/handles/cfg are None when no
        PSI-side probe was found. Only populated if `use_psi_models=True`.
        Like `_iter_pairs`, models are loaded once and cached for the runner's
        lifetime."""
        if self._psi_cache is None:
            self._psi_cache = self._build_psi_cache()
        yield from self._psi_cache

    @torch.no_grad()
    def _forward_one(self, pangolin_model, probe, handles, cfg, seq_tensor):
        out = pangolin_model(seq_tensor)[0]       
        L_out = out.shape[-1]
        
        gathered_acts = []
        # 1. Gather all requested network layers
        for layer_name, h in handles.items():
            acts = h["cache"]["activations"]
            if not h["is_cropped"]:
                acts = acts[..., PANGOLIN_CROP:PANGOLIN_CROP + L_out]
            gathered_acts.append(acts)
            
        # 2. Extract and append raw sequence to the centre channel
        if cfg.get("include_sequence", False):
            cropped_seq = seq_tensor[..., PANGOLIN_CROP:PANGOLIN_CROP + L_out]
            gathered_acts.append(cropped_seq)
            
        combined_acts = torch.cat(gathered_acts, dim=1)
        
        probe_logits = probe(combined_acts)               
        probe_probs = torch.softmax(probe_logits, dim=1)[0]  
        return out, probe_probs

    @torch.no_grad()
    def score_sequence(self, seq):
        L_out = len(seq)
        if L_out == 0:
            raise ValueError("Sequence is empty")
        if L_out > USABLE_LEN:
            raise ValueError(
                f"Sequence length {L_out} exceeds single-window max {USABLE_LEN}. "
                "Use score_long_sequence() for longer inputs.")

        padded = "N" * PANGOLIN_CROP + seq + "N" * (WINDOW_LEN - PANGOLIN_CROP - L_out)
        seq_t = one_hot_encode(padded).unsqueeze(0).to(self.device)

        prob_sums = {t: torch.zeros(L_out) for t in self.tissues_present}
        psi_sums  = {t: torch.zeros(L_out) for t in self.tissues_present} if self.use_psi_models else None
        psi_counts = {t: 0 for t in self.tissues_present} if self.use_psi_models else None
        probe_sum = torch.zeros(3, L_out)
        probe_sums_per_tissue = (
            {t: torch.zeros(3, L_out) for t in self.tissues_present}
            if self.per_tissue_probes else None
        )
        # PSI-side per-tissue probe sums. Populated only when use_psi_models
        # AND per_tissue_probes AND PSI-tuned probes are present.
        compute_psi_probes = (self.use_psi_models and self.per_tissue_probes
                              and self._psi_has_probes)
        probe_sums_per_tissue_psi = (
            {t: torch.zeros(3, L_out) for t in self.tissues_present}
            if compute_psi_probes else None
        )

        # Pass 1: P-tuned models — collect P + probe outputs.
        for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_pairs():
            pangolin_out, probe_probs = self._forward_one(
                pangolin_model, probe, handles, cfg, seq_t)
            prob_sums[tissue_idx] += pangolin_out[PROB_CHANNEL_PER_TISSUE[tissue_idx], :L_out].cpu()
            p = probe_probs[:, :L_out].cpu()
            probe_sum += p
            if probe_sums_per_tissue is not None:
                probe_sums_per_tissue[tissue_idx] += p

        # Pass 2 (optional): PSI-tuned models — collect PSI + (optional) PSI-side probe.
        if self.use_psi_models:
            for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_psi_models():
                if probe is not None and compute_psi_probes:
                    pangolin_out, probe_probs = self._forward_one(
                        pangolin_model, probe, handles, cfg, seq_t)
                    p = probe_probs[:, :L_out].cpu()
                    probe_sums_per_tissue_psi[tissue_idx] += p
                else:
                    with torch.no_grad():
                        pangolin_out = pangolin_model(seq_t)[0]
                psi_sums[tissue_idx]  += pangolin_out[PSI_CHANNEL_PER_TISSUE[tissue_idx], :L_out].cpu()
                psi_counts[tissue_idx] += 1

        return self._assemble_result(
            prob_sums,
            psi_sums,
            probe_sum,
            psi_counts=psi_counts,
            metadata={"length": L_out, "tiled": False},
            probe_sums_per_tissue=probe_sums_per_tissue,
            probe_sums_per_tissue_psi=probe_sums_per_tissue_psi,
        )

    @torch.no_grad()
    def score_long_sequence(self, seq, overlap=TILE_OVERLAP):
        L_out = len(seq)
        if L_out == 0:
            raise ValueError("Sequence is empty")
        if L_out <= USABLE_LEN:
            return self.score_sequence(seq)
        if not (0 <= overlap < USABLE_LEN):
            raise ValueError(f"overlap must be in [0, {USABLE_LEN}), got {overlap}")

        stride = USABLE_LEN - overlap
        starts = list(range(0, max(1, L_out - USABLE_LEN + 1), stride))
        if starts[-1] + USABLE_LEN < L_out:
            starts.append(L_out - USABLE_LEN)

        blend = self._triangular_blend(USABLE_LEN, overlap)
        padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
        padded_t = one_hot_encode(padded).to(self.device)

        prob_sums = {t: torch.zeros(L_out) for t in self.tissues_present}
        psi_sums = (
            {t: torch.zeros(L_out) for t in self.tissues_present}
            if self.use_psi_models else None
        )
        probe_sum = torch.zeros(3, L_out)
        weight_sum = torch.zeros(L_out)
        tissue_weight_sum = {t: torch.zeros(L_out) for t in self.tissues_present}
        psi_tissue_weight_sum = (
            {t: torch.zeros(L_out) for t in self.tissues_present}
            if self.use_psi_models else None
        )
        probe_sums_per_tissue = (
            {t: torch.zeros(3, L_out) for t in self.tissues_present}
            if self.per_tissue_probes else None
        )
        compute_psi_probes = (self.use_psi_models and self.per_tissue_probes
                              and self._psi_has_probes)
        probe_sums_per_tissue_psi = (
            {t: torch.zeros(3, L_out) for t in self.tissues_present}
            if compute_psi_probes else None
        )

        # Pass 1: P-tuned models — P + probe.
        for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_pairs():
            for start in starts:
                window = padded_t[:, start:start + WINDOW_LEN].unsqueeze(0)
                pangolin_out, probe_probs = self._forward_one(
                    pangolin_model, probe, handles, cfg, window)

                lo = start
                hi = min(start + USABLE_LEN, L_out)
                n = hi - lo
                w = blend[:n]

                prob_sums[tissue_idx][lo:hi] += (
                    pangolin_out[PROB_CHANNEL_PER_TISSUE[tissue_idx], :n].cpu() * w)
                p_w = probe_probs[:, :n].cpu() * w.unsqueeze(0)
                probe_sum[:, lo:hi] += p_w
                if probe_sums_per_tissue is not None:
                    probe_sums_per_tissue[tissue_idx][:, lo:hi] += p_w
                tissue_weight_sum[tissue_idx][lo:hi] += w
                weight_sum[lo:hi] += w

        # Pass 2 (optional): PSI-tuned models — PSI + (optional) PSI-side probe.
        if self.use_psi_models:
            for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_psi_models():
                for start in starts:
                    window = padded_t[:, start:start + WINDOW_LEN].unsqueeze(0)
                    lo = start
                    hi = min(start + USABLE_LEN, L_out)
                    n = hi - lo
                    w = blend[:n]
                    if probe is not None and compute_psi_probes:
                        pangolin_out, probe_probs = self._forward_one(
                            pangolin_model, probe, handles, cfg, window)
                        p_w = probe_probs[:, :n].cpu() * w.unsqueeze(0)
                        probe_sums_per_tissue_psi[tissue_idx][:, lo:hi] += p_w
                    else:
                        with torch.no_grad():
                            pangolin_out = pangolin_model(window)[0]
                    psi_sums[tissue_idx][lo:hi] += (
                        pangolin_out[PSI_CHANNEL_PER_TISSUE[tissue_idx], :n].cpu() * w)
                    psi_tissue_weight_sum[tissue_idx][lo:hi] += w

        for tissue_idx in self.tissues_present:
            weights = tissue_weight_sum[tissue_idx].clamp_min(1e-9)
            prob_sums[tissue_idx] = prob_sums[tissue_idx] / weights
            if probe_sums_per_tissue is not None:
                probe_sums_per_tissue[tissue_idx] = (
                    probe_sums_per_tissue[tissue_idx] / weights.unsqueeze(0))
            if self.use_psi_models:
                psi_weights = psi_tissue_weight_sum[tissue_idx].clamp_min(1e-9)
                psi_sums[tissue_idx] = psi_sums[tissue_idx] / psi_weights
                if probe_sums_per_tissue_psi is not None:
                    probe_sums_per_tissue_psi[tissue_idx] = (
                        probe_sums_per_tissue_psi[tissue_idx] / psi_weights.unsqueeze(0))
        probe_sum = probe_sum / weight_sum.clamp_min(1e-9).unsqueeze(0)

        return self._assemble_result(
            prob_sums,
            psi_sums,
            probe_sum,
            metadata={
                "length": L_out,
                "tiled": True,
                "n_windows": len(starts),
                "overlap": overlap,
            },
            skip_pair_normalisation=True,
            probe_sums_per_tissue=probe_sums_per_tissue,
            probe_sums_per_tissue_psi=probe_sums_per_tissue_psi,
        )

    @torch.no_grad()
    def score_region(self, fasta_path, chrom, start, end, **kwargs):
        try:
            import pyfastx
        except ImportError as e:
            raise ImportError("score_region requires pyfastx: pip install pyfastx") from e

        fasta = pyfastx.Fasta(str(fasta_path))
        if chrom not in fasta:
            raise KeyError(f"Chromosome {chrom!r} not in FASTA")

        start, end = int(start), int(end)
        if not (0 <= start < end <= len(fasta[chrom])):
            raise ValueError(f"Bad coords [{start}, {end}) for {chrom} "
                             f"(length {len(fasta[chrom])})")

        seq = fasta[chrom][start:end].seq
        result = score_sequence_or_long_sequence(self, seq) if not kwargs else (
            self.score_sequence(seq) if len(seq) <= USABLE_LEN
            else self.score_long_sequence(seq, **kwargs)
        )
        result.metadata.update({
            "chrom": chrom,
            "start": start,
            "end": end,
            "fasta": str(fasta_path),
        })
        return result

    def score_variant(self, fasta_path, chrom, pos, ref, alt, distance=50):
        try:
            import pyfastx
        except ImportError as e:
            raise ImportError("score_variant requires pyfastx: pip install pyfastx") from e

        from ._variants import score_variant as _score_variant
        fasta = pyfastx.Fasta(str(fasta_path))
        return _score_variant(self, fasta, chrom, pos, ref, alt, distance=distance)

    def score_vcf(self, vcf_in, vcf_out, fasta_path, distance=50,
                  tissue_for_info=None, progress=True):
        from ._variants import score_vcf as _score_vcf
        return _score_vcf(
            self,
            vcf_in,
            vcf_out,
            fasta_path,
            distance=distance,
            tissue_for_info=tissue_for_info,
            progress=progress,
        )

    @staticmethod
    def _triangular_blend(usable_len, overlap):
        if overlap == 0:
            return torch.ones(usable_len)
        ramp = torch.linspace(0, 1, overlap + 2)[1:-1]
        weights = torch.ones(usable_len)
        weights[:overlap] = ramp
        weights[-overlap:] = ramp.flip(0)
        return weights

    def _assemble_result(self, prob_sums, psi_sums, probe_sum, metadata=None,
                         skip_pair_normalisation=False,
                         probe_sums_per_tissue=None,
                         probe_sums_per_tissue_psi=None,
                         psi_counts=None):
        if not skip_pair_normalisation:
            pair_tissues = [t for _, _, t in self._pair_specs]
            psi_tissues  = ([t for _, _, t in self._psi_specs]
                            if self.use_psi_models else [])
            for tissue_idx in self.tissues_present:
                n_tissue = pair_tissues.count(tissue_idx)
                prob_sums[tissue_idx] = prob_sums[tissue_idx] / max(n_tissue, 1)
                if probe_sums_per_tissue is not None:
                    probe_sums_per_tissue[tissue_idx] = (
                        probe_sums_per_tissue[tissue_idx] / max(n_tissue, 1))
                if psi_sums is not None:
                    n_psi = psi_counts[tissue_idx] if psi_counts is not None else psi_tissues.count(tissue_idx)
                    psi_sums[tissue_idx] = psi_sums[tissue_idx] / max(n_psi, 1)
                if probe_sums_per_tissue_psi is not None:
                    n_psi = psi_counts[tissue_idx] if psi_counts is not None else psi_tissues.count(tissue_idx)
                    probe_sums_per_tissue_psi[tissue_idx] = (
                        probe_sums_per_tissue_psi[tissue_idx] / max(n_psi, 1))
            probe_sum = probe_sum / len(self._pair_specs)

        probe_sum = self._apply_correction(probe_sum)

        probe_per_tissue = None
        if probe_sums_per_tissue is not None:
            # (3, n_tissues, L), correction applied per-tissue.
            corrected = [self._apply_correction(probe_sums_per_tissue[t])
                         for t in self.tissues_present]
            probe_per_tissue = torch.stack(corrected, dim=1)

        probe_per_tissue_psi = None
        if probe_sums_per_tissue_psi is not None:
            corrected = [self._apply_correction(probe_sums_per_tissue_psi[t])
                         for t in self.tissues_present]
            probe_per_tissue_psi = torch.stack(corrected, dim=1)

        pangolin_psi_t = (
            torch.stack([psi_sums[t] for t in self.tissues_present])
            if psi_sums is not None else None
        )

        return BiPangolinResult(
            pangolin_prob=torch.stack([prob_sums[t] for t in self.tissues_present]),
            pangolin_psi=pangolin_psi_t,
            probe_none=probe_sum[NONE_CLASS],
            probe_acceptor=probe_sum[ACC_CLASS],
            probe_donor=probe_sum[DON_CLASS],
            tissues=self.tissue_names,
            metadata=metadata or {},
            probe_per_tissue=probe_per_tissue,
            probe_per_tissue_psi=probe_per_tissue_psi,
        )


# ---------------------------------------------------------------------------
# Calibration Test
# ---------------------------------------------------------------------------

_CALIBRATION_SEQ = (
    "cacagcaccggcggcatggacgagctgtacaaggactacaaggacgatgatgacaagtgataaacaaatggt"
    "aaggaagggcacatcaatctttgcttaattgtcctttactctaaagatgtattttatcatactgaatgctaa"
    "acttgatatctccttttaggtcattgatgtccttcaccccgggaaggcgacagtgcctaagacagaaattcgg"
).upper()


CALIBRATION_SEQ = _CALIBRATION_SEQ


def selftest(pangolin_model_dir=None, probe_dir=None, device="auto",
             ensemble=False, tissue="all_tissues"):
    runner = BiPangolinRunner(
        pangolin_model_dir, probe_dir, device=device, ensemble=ensemble, tissue=tissue)
    result = runner.score_sequence(_CALIBRATION_SEQ)

    don_argmax = int(result.probe_donor.argmax())
    acc_argmax = int(result.probe_acceptor.argmax())
    print(f"\nCalibration sequence ({len(_CALIBRATION_SEQ)} bp)")
    print(f"  Annotated donor:    69   probe argmax: {don_argmax}  "
          f"(P={result.probe_donor[don_argmax]:.3f}, P@69={result.probe_donor[69]:.3f})")
    print(f"  Annotated acceptor: 163  probe argmax: {acc_argmax}  "
          f"(P={result.probe_acceptor[acc_argmax]:.3f}, P@163={result.probe_acceptor[163]:.3f})")
    return result


def score_sequence_or_long_sequence(runner, seq):
    if len(seq) <= USABLE_LEN:
        return runner.score_sequence(seq)
    return runner.score_long_sequence(seq)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="biPangolin self-test")
    p.add_argument("pangolin_model_dir")
    p.add_argument("probe_dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--ensemble", action="store_true")
    args = p.parse_args()
    selftest(args.pangolin_model_dir, args.probe_dir,
             device=args.device, ensemble=args.ensemble)
