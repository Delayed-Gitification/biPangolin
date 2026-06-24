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
import sys
import torch
import torch.nn as nn

from .model import Pangolin, L, W, AR
from ._weights import resolve_pangolin_weights, resolve_probe_weights


# Geometry constants
#
# The Pangolin model internally crops PANGOLIN_CROP positions from each side of
# its output (model.py: CL = 2*sum(AR*(W-1)) = 10000, F.pad(skip, (-CL//2, ...))).
# That crop is exactly the model's receptive-field radius, so EVERY position the
# model emits has its full ±PANGOLIN_CROP bp of real context — the prediction for
# a given position is identical regardless of which window produced it. As a
# result, long sequences can be tiled with zero overlap and no blending: each
# tile yields (window_len - 2*PANGOLIN_CROP) usable positions that abut the next
# tile's exactly. (The old triangular-blend overlap scheme was a no-op averaging
# identical values.)
PANGOLIN_CROP = 5000
# Default per-tile model input length. Larger => fewer forward passes (the fixed
# 2*CROP overhead is amortised over more usable output) but more activation
# memory. Configurable per-runner via BiPangolinRunner(window_len=...).
DEFAULT_WINDOW_LEN = 50000
WINDOW_LEN = DEFAULT_WINDOW_LEN              # module-level default (back-compat)
USABLE_LEN = WINDOW_LEN - 2 * PANGOLIN_CROP  # usable output positions per tile

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

    def __getattr__(self, name):
        """Friendly per-tissue accessors for the routed Pangolin tracks:

            result.brain_P                  # (2, L) for brain: acceptor row 0, donor row 1
            result.brain_PSI                # same, PSI metric (needs PSI models)
            result.all_tissue_average_P     # mean over tissues (needs all tissues)
            result.all_tissue_average_PSI

        Each returns a (2, L) tensor: row 0 = acceptor, row 1 = donor. This is
        the SAME channel order used everywhere else in biPangolin (routed_tracks,
        the CLI bedGraphs, the VCF deltas, the probe class constants) — acceptor
        always comes first. Asking for something that was not computed raises a
        descriptive error telling you how to get it. Only reached when normal
        attribute lookup fails, so the real dataclass fields (pangolin_prob,
        probe_acceptor, ...) are untouched.
        """
        # Never intercept dunder / private lookups (pickle, copy, etc.).
        if name.startswith("_"):
            raise AttributeError(name)

        if name.endswith("_PSI"):
            tissue, metric = name[:-4], "PSI"
        elif name.endswith("_P"):
            tissue, metric = name[:-2], "P"
        else:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}")

        if metric == "P":
            if self.metadata.get("psi_only"):
                raise AttributeError(
                    f"`{name}` is unavailable: this result was produced in "
                    "psi-only mode, so the P-tuned models were never run. Re-run "
                    "with `--psi` (CLI) or use_psi_models=True instead of "
                    "psi_only=True to get P tracks alongside PSI."
                )
        else:  # PSI
            if self.pangolin_psi is None:
                raise AttributeError(
                    f"`{name}` is unavailable: PSI was not computed. Build the "
                    "runner with use_psi_models=True (Python) or pass `--psi` / "
                    "`--psi-only` (CLI), then re-score."
                )

        prob_routed, psi_routed = self.routed_tracks()
        routed = prob_routed if metric == "P" else psi_routed  # (2, n_tissues, L)

        if tissue == "all_tissue_average":
            if len(self.tissues) != len(TISSUE_NAMES):
                raise AttributeError(
                    f"`{name}` needs all {len(TISSUE_NAMES)} tissues "
                    f"{TISSUE_NAMES}, but this result only has {tuple(self.tissues)}. "
                    "Build the runner with tissue='all_tissues' (the default) to "
                    "average across every tissue."
                )
            pair = routed.mean(dim=1)  # (2, L)
        else:
            if tissue not in self.tissues:
                raise AttributeError(
                    f"Tissue {tissue!r} is not in this result. Available tissues: "
                    f"{tuple(self.tissues)}. (Use one of those, or "
                    "'all_tissue_average'.) If you wanted a tissue not listed, "
                    "build the runner with that tissue (or tissue='all_tissues')."
                )
            ti = self.tissues.index(tissue)
            pair = routed[:, ti, :]  # (2, L), row 0 acceptor / row 1 donor

        # Same convention as routed_tracks/CLI/VCF: row 0 acceptor, row 1 donor.
        return pair.contiguous()

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

    def _routing_masks(self, double_val_floor=0.01, double_val_ratio=0.1):
        """Decide, per position, which of {acceptor, donor} column(s) get the
        Pangolin value. Returns (acc_col_mask, don_col_mask), each a boolean
        (L,) tensor; a "both" position is True in both.

        Routing is driven by the corrected probe acceptor/donor probabilities
        (the `none` class is deliberately ignored — every position is routed):

          * single column = argmax(acceptor, donor)               (the default)
          * BOTH columns   when the probe is genuinely ambiguous:
                min(acc, don) >= double_val_floor                 (floor)
                AND  min(acc, don) / max(acc, don) >= double_val_ratio  (ratio)

        The floor rejects two near-zero values (e.g. 0.001/0.001 -> argmax,
        not "both"); the ratio rejects a clear winner with a marginal loser
        (e.g. don=0.99, acc=0.02 -> donor only, ratio 0.02 < 0.1). Both must
        hold for a position to light up both columns.
        """
        acc = self.probe_acceptor
        don = self.probe_donor
        mn = torch.minimum(acc, don)
        mx = torch.maximum(acc, don)
        both = (mn >= double_val_floor) & (mn >= double_val_ratio * mx)
        acc_wins = acc >= don                       # ties -> acceptor
        acc_col_mask = both | (~both & acc_wins)
        don_col_mask = both | (~both & ~acc_wins)
        return acc_col_mask, don_col_mask

    def routed_tracks(self, double_val_floor=0.01, double_val_ratio=0.1):
        """Route Pangolin P (and PSI, if available) into acceptor/donor tracks.

        This is biPangolin's default user-facing output: a SpliceAI-style pair
        of tracks where the *value* is always the Pangolin metric and the probe
        only decides which column it lands in. The unrouted column is exactly
        0.0, so in practice most intronic positions read e.g. acceptor=0,
        donor=1.2e-5 (one hard zero, one near-zero).

        Returns (prob_routed, psi_routed):
            prob_routed : (2, n_tissues, L) — channel 0 acceptor, 1 donor
            psi_routed  : same shape, or None if the runner had no PSI models

        One identity, one router. A base has a single splice identity — it
        either IS a donor site or IS an acceptor site; that is a property of the
        sequence, not of which Pangolin head (P vs PSI) you read. So a SINGLE
        routing decision is computed (from `_routing_masks`) and applied to BOTH
        P and PSI. We deliberately do NOT route PSI by a separate PSI-side probe:
        doing so could send the same base's P to the donor column and its PSI to
        the acceptor column — a self-contradictory "donor for P, acceptor for
        PSI" result — and would decouple the donor-P / donor-PSI tracks that
        otherwise light up together. The router used here is the P-side probe,
        which is also the one the none-class correction (`correction_k`) was
        calibrated against. The lone exception is psi_only mode (the P-tuned
        models are never run), where routing necessarily falls back to the
        PSI-side probes; at borderline sites that can disagree slightly with a
        --psi run, which is expected and unavoidable.
        """
        acc_col_mask, don_col_mask = self._routing_masks(
            double_val_floor=double_val_floor, double_val_ratio=double_val_ratio)

        def _route(values, fill_value=0.0):
            routed = torch.full(
                (2, values.shape[0], values.shape[1]), fill_value,
                dtype=values.dtype, device=values.device)
            routed[0, :, acc_col_mask] = values[:, acc_col_mask]
            routed[1, :, don_col_mask] = values[:, don_col_mask]
            return routed

        output_unscaled = self.metadata.get("output_unscaled_values", False)
        if output_unscaled:
            prob_shifted = self.pangolin_prob
            prob_fill = 0.05
        else:
            prob_shifted = torch.clamp((self.pangolin_prob - 0.05) / 0.9, min=0.0, max=1.0)
            prob_fill = 0.0

        prob_routed = _route(prob_shifted, fill_value=prob_fill)
        psi_routed = _route(self.pangolin_psi, fill_value=0.0) if self.pangolin_psi is not None else None
        return prob_routed, psi_routed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _mps_is_healthy() -> bool:
    """Probe whether the MPS backend handles the ops Pangolin relies on.

    PyTorch's Metal (Apple Silicon) backend has historically mis-handled
    high-dilation 1D convolutions (Pangolin uses atrous rates up to 25) and
    boolean-mask indexing (used when routing tracks) — sometimes returning
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
                 per_tissue_probes=False, use_psi_models=False,
                 window_len=None, n_models_per_tissue=None,
                 output_unscaled_values=False):
        """Load Pangolin + probes for inference.

        window_len: per-tile model input length (default 50000). Each tile
            yields window_len - 2*PANGOLIN_CROP usable output positions; long
            sequences are tiled with no overlap. Lower this if a machine OOMs
            on the forward pass (e.g. CPU/MPS); raise it to cut the number of
            forward passes on a GPU.

        use_psi_models: if True, additionally load Pangolin's PSI-tuned weight
            files (final.{fold}.{1,3,5,7}.3.v2) and read PSI from those.
            If False (default), pangolin_psi is left None on the result —
            because reading the PSI channel from a P-tuned model gives a
            misleading side-output that correlates with P, not real PSI.
            Costs ~2× Pangolin inference compute.
        """
        self.device = _resolve_device(device)
        self.output_unscaled_values = output_unscaled_values
        self.window_len = int(window_len) if window_len else DEFAULT_WINDOW_LEN
        if self.window_len <= 2 * PANGOLIN_CROP:
            raise ValueError(
                f"window_len must exceed 2*PANGOLIN_CROP = {2 * PANGOLIN_CROP} "
                f"(otherwise no usable output remains after the model's crop), "
                f"got {self.window_len}")
        self.usable_len = self.window_len - 2 * PANGOLIN_CROP
        if n_models_per_tissue is not None and n_models_per_tissue not in (1, 2, 3):
            raise ValueError(
                f"n_models_per_tissue must be 1, 2, or 3 (the number of trained "
                f"folds per tissue), got {n_models_per_tissue}")
        self.n_models_per_tissue = n_models_per_tissue
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
        # Fast modes: keep only the first n folds per tissue (1, 2, or 3). Fewer
        # folds => proportionally less inference, at some loss of ensemble
        # robustness. Routing is unaffected (it still averages whatever folds
        # remain before deciding).
        if n_models_per_tissue is not None:
            candidates = self._limit_per_tissue(candidates, n_models_per_tissue)
            psi_candidates = self._limit_per_tissue(psi_candidates, n_models_per_tissue)
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

    @staticmethod
    def _limit_per_tissue(candidates, n):
        """Keep at most n (path, tissue_idx) entries per tissue_idx, preserving
        order (folds are encountered fold-1, fold-2, fold-3 in sorted name
        order, so this keeps the lowest-numbered folds deterministically)."""
        kept_per_tissue = {}
        out = []
        for path, tissue_idx in candidates:
            c = kept_per_tissue.get(tissue_idx, 0)
            if c < n:
                out.append((path, tissue_idx))
                kept_per_tissue[tissue_idx] = c + 1
        return out

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

    def _require_psi_routing(self):
        """Validate that --psi-only style routing is possible: we need the
        PSI-tuned models loaded AND PSI-side probes (probes trained against the
        PSI-tuned model activations, i.e. probe files for model_nums 1,3,5,7)
        for every PSI model, so the donor/acceptor routing can be computed
        without running the P-tuned models at all."""
        if not self.use_psi_models:
            raise ValueError(
                "psi_only requires use_psi_models=True (the PSI-tuned Pangolin "
                "models supply both the PSI values and the routing probe).")
        if not self._psi_has_probes:
            raise ValueError(
                "psi_only routing needs PSI-side probes — probe files matching "
                "the PSI-tuned models (model_nums 1,3,5,7) — for every PSI "
                "model in probe_dir, but they were not all found. Train/place "
                "those probes, or use --psi (which routes with the P-side "
                "probes) instead.")

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
    def score_sequence(self, seq, psi_only=False):
        L_out = len(seq)
        if L_out == 0:
            raise ValueError("Sequence is empty")
        if L_out > self.usable_len:
            raise ValueError(
                f"Sequence length {L_out} exceeds single-window max {self.usable_len}. "
                "Use score_long_sequence() for longer inputs.")
        if psi_only:
            self._require_psi_routing()

        # Pad with exactly the receptive-field flank on each side: the model
        # crops PANGOLIN_CROP off each end, so a (L_out + 2*PANGOLIN_CROP) input
        # yields exactly L_out output positions. We deliberately do NOT pad out
        # to a fixed window_len — that would waste compute (e.g. a 10kb variant
        # window padded to 50kb is 5x the work for the same result).
        padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
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

        # Pass 1: P-tuned models — collect P + probe outputs. Skipped entirely
        # in psi_only mode (no P value, routing comes from the PSI-side probes).
        if not psi_only:
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
                # Run the PSI-side probe when we need per-tissue PSI probes OR
                # when psi_only routing requires the flat probe from PSI models.
                need_probe = probe is not None and (compute_psi_probes or psi_only)
                if need_probe:
                    pangolin_out, probe_probs = self._forward_one(
                        pangolin_model, probe, handles, cfg, seq_t)
                    p = probe_probs[:, :L_out].cpu()
                    if compute_psi_probes:
                        probe_sums_per_tissue_psi[tissue_idx] += p
                    if psi_only:
                        probe_sum += p
                        if probe_sums_per_tissue is not None:
                            probe_sums_per_tissue[tissue_idx] += p
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
            metadata={"length": L_out, "tiled": False, "psi_only": psi_only},
            probe_sums_per_tissue=probe_sums_per_tissue,
            probe_sums_per_tissue_psi=probe_sums_per_tissue_psi,
            probe_from_psi=psi_only,
        )

    @torch.no_grad()
    def score_long_sequence(self, seq, psi_only=False):
        """Score a sequence longer than one window by gap-free, overlap-free tiling.

        Each tile feeds window_len bases to the model and gets back usable_len
        output positions, each with full ±PANGOLIN_CROP context (guaranteed by
        the model's internal crop). Tiles abut exactly, so there's nothing to
        blend — within a single model+probe pair every output position is
        written exactly once (a clamped final tile may re-cover some positions;
        we skip the already-covered prefix so each position is counted once).
        Ensemble averaging across pairs is then handled by _assemble_result,
        identically to the single-window path.
        """
        L_out = len(seq)
        if L_out == 0:
            raise ValueError("Sequence is empty")
        if psi_only:
            self._require_psi_routing()
        if L_out <= self.usable_len:
            return self.score_sequence(seq, psi_only=psi_only)

        tile_out = self.usable_len
        # Window output-start positions (in sequence coordinates). Stride ==
        # tile_out so consecutive usable regions abut. A final clamped start
        # ensures the tail is covered; it can only ever re-cover (never skip).
        starts = list(range(0, max(1, L_out - tile_out + 1), tile_out))
        if starts[-1] + tile_out < L_out:
            starts.append(L_out - tile_out)

        # Pad the whole sequence with one receptive-field flank each side. A
        # window starting at sequence position `start` reads padded[start :
        # start+window_len] and outputs sequence positions [start, start+tile_out).
        padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
        padded_t = one_hot_encode(padded).to(self.device)

        prob_sums = {t: torch.zeros(L_out) for t in self.tissues_present}
        psi_sums  = {t: torch.zeros(L_out) for t in self.tissues_present} if self.use_psi_models else None
        psi_counts = {t: 0 for t in self.tissues_present} if self.use_psi_models else None
        probe_sum = torch.zeros(3, L_out)
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

        # Pass 1: P-tuned models — P + probe. Accumulate each pair once per pos.
        # Skipped in psi_only mode (routing comes from the PSI-side probes).
        if not psi_only:
            for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_pairs():
                covered_to = 0
                for start in starts:
                    window = padded_t[:, start:start + self.window_len].unsqueeze(0)
                    pangolin_out, probe_probs = self._forward_one(
                        pangolin_model, probe, handles, cfg, window)
                    lo = max(start, covered_to)
                    hi = min(start + tile_out, L_out)
                    if hi <= lo:
                        continue
                    off = lo - start                      # offset into this tile's output
                    prob_sums[tissue_idx][lo:hi] += (
                        pangolin_out[PROB_CHANNEL_PER_TISSUE[tissue_idx], off:off + (hi - lo)].cpu())
                    p = probe_probs[:, off:off + (hi - lo)].cpu()
                    probe_sum[:, lo:hi] += p
                    if probe_sums_per_tissue is not None:
                        probe_sums_per_tissue[tissue_idx][:, lo:hi] += p
                    covered_to = hi

        # Pass 2 (optional): PSI-tuned models — PSI + (optional) PSI-side probe.
        if self.use_psi_models:
            for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_psi_models():
                covered_to = 0
                # Run the PSI-side probe when we need per-tissue PSI probes OR
                # when psi_only routing requires the flat probe from PSI models.
                need_probe = probe is not None and (compute_psi_probes or psi_only)
                for start in starts:
                    window = padded_t[:, start:start + self.window_len].unsqueeze(0)
                    lo = max(start, covered_to)
                    hi = min(start + tile_out, L_out)
                    if hi <= lo:
                        continue
                    off = lo - start
                    if need_probe:
                        pangolin_out, probe_probs = self._forward_one(
                            pangolin_model, probe, handles, cfg, window)
                        p = probe_probs[:, off:off + (hi - lo)].cpu()
                        if compute_psi_probes:
                            probe_sums_per_tissue_psi[tissue_idx][:, lo:hi] += p
                        if psi_only:
                            probe_sum[:, lo:hi] += p
                            if probe_sums_per_tissue is not None:
                                probe_sums_per_tissue[tissue_idx][:, lo:hi] += p
                    else:
                        with torch.no_grad():
                            pangolin_out = pangolin_model(window)[0]
                    psi_sums[tissue_idx][lo:hi] += (
                        pangolin_out[PSI_CHANNEL_PER_TISSUE[tissue_idx], off:off + (hi - lo)].cpu())
                    covered_to = hi
                psi_counts[tissue_idx] += 1

        return self._assemble_result(
            prob_sums,
            psi_sums,
            probe_sum,
            psi_counts=psi_counts,
            metadata={"length": L_out, "tiled": True, "n_windows": len(starts),
                      "psi_only": psi_only},
            probe_sums_per_tissue=probe_sums_per_tissue,
            probe_sums_per_tissue_psi=probe_sums_per_tissue_psi,
            probe_from_psi=psi_only,
        )

    @torch.no_grad()
    def score_region(self, fasta_path, chrom, start, end, psi_only=False):
        import pyfastx
        fasta = pyfastx.Fasta(str(fasta_path))
        if chrom not in fasta:
            raise KeyError(f"Chromosome {chrom!r} not in FASTA")

        start, end = int(start), int(end)
        if not (0 <= start < end <= len(fasta[chrom])):
            raise ValueError(f"Bad coords [{start}, {end}) for {chrom} "
                             f"(length {len(fasta[chrom])})")

        seq = fasta[chrom][start:end].seq
        result = score_sequence_or_long_sequence(self, seq, psi_only=psi_only)
        result.metadata.update({
            "chrom": chrom,
            "start": start,
            "end": end,
            "fasta": str(fasta_path),
        })
        return result

    def score_variant(self, fasta_path, chrom, pos, ref, alt, distance=50):
        import pyfastx
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

    def _assemble_result(self, prob_sums, psi_sums, probe_sum, metadata=None,
                         skip_pair_normalisation=False,
                         probe_sums_per_tissue=None,
                         probe_sums_per_tissue_psi=None,
                         psi_counts=None,
                         probe_from_psi=False):
        # probe_from_psi=True (psi_only mode): the flat routing probe and the
        # per-tissue probe were accumulated from the PSI-tuned models, so they
        # must be normalised by the PSI model counts, not the P-pair counts.
        if not skip_pair_normalisation:
            pair_tissues = [t for _, _, t in self._pair_specs]
            psi_tissues  = ([t for _, _, t in self._psi_specs]
                            if self.use_psi_models else [])
            for tissue_idx in self.tissues_present:
                n_tissue = pair_tissues.count(tissue_idx)
                prob_sums[tissue_idx] = prob_sums[tissue_idx] / max(n_tissue, 1)
                if probe_sums_per_tissue is not None:
                    n_probe_tissue = (
                        (psi_counts[tissue_idx] if psi_counts is not None
                         else psi_tissues.count(tissue_idx))
                        if probe_from_psi else n_tissue)
                    probe_sums_per_tissue[tissue_idx] = (
                        probe_sums_per_tissue[tissue_idx] / max(n_probe_tissue, 1))
                if psi_sums is not None:
                    n_psi = psi_counts[tissue_idx] if psi_counts is not None else psi_tissues.count(tissue_idx)
                    psi_sums[tissue_idx] = psi_sums[tissue_idx] / max(n_psi, 1)
                if probe_sums_per_tissue_psi is not None:
                    n_psi = psi_counts[tissue_idx] if psi_counts is not None else psi_tissues.count(tissue_idx)
                    probe_sums_per_tissue_psi[tissue_idx] = (
                        probe_sums_per_tissue_psi[tissue_idx] / max(n_psi, 1))
            n_probe = len(self._psi_specs) if probe_from_psi else len(self._pair_specs)
            probe_sum = probe_sum / max(n_probe, 1)

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

        meta = metadata or {}
        meta["output_unscaled_values"] = getattr(self, "output_unscaled_values", False)

        return BiPangolinResult(
            pangolin_prob=torch.stack([prob_sums[t] for t in self.tissues_present]),
            pangolin_psi=pangolin_psi_t,
            probe_none=probe_sum[NONE_CLASS],
            probe_acceptor=probe_sum[ACC_CLASS],
            probe_donor=probe_sum[DON_CLASS],
            tissues=self.tissue_names,
            metadata=meta,
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


def score_sequence_or_long_sequence(runner, seq, psi_only=False):
    if len(seq) <= runner.usable_len:
        return runner.score_sequence(seq, psi_only=psi_only)
    return runner.score_long_sequence(seq, psi_only=psi_only)


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
