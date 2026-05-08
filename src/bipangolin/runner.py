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
from dataclasses import dataclass
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
USABLE_LEN = WINDOW_LEN_DEFAULT - 2 * PANGOLIN_CROP
TILE_OVERLAP = 2000          

# Pangolin output channel mapping
PROB_CHANNEL_PER_TISSUE = [1, 4, 7, 10]   # P(spliced)
PSI_CHANNEL_PER_TISSUE  = [2, 5, 8, 11]   # PSI / usage
TISSUE_NAMES = ("heart", "liver", "brain", "testis")

PANGOLIN_FILE_RE = re.compile(r"^final\.([1-3])\.([0246])\.3\.v2$")
DEFAULT_CORRECTION_FILE = Path(__file__).parent / "data" / "probes" / "optimal_correction.json"

# Class encoding
NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2

# One-hot encoding
_BASE_TO_IDX = {b: i for i, b in enumerate("NACGT")}
_IN_MAP = torch.tensor([
    [0, 0, 0, 0],   # N
    [1, 0, 0, 0],   # A
    [0, 1, 0, 0],   # C
    [0, 0, 1, 0],   # G
    [0, 0, 0, 1],   # T
], dtype=torch.float32)


def one_hot_encode(seq: str) -> torch.Tensor:
    seq = seq.upper()
    idx = torch.tensor([_BASE_TO_IDX.get(b, 0) for b in seq], dtype=torch.long)
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
    pangolin_psi: torch.Tensor
    probe_none: torch.Tensor
    probe_acceptor: torch.Tensor
    probe_donor: torch.Tensor
    tissues: tuple

    def __len__(self):
        return self.probe_none.shape[0]

    @property
    def raw(self):
        return torch.cat([
            self.pangolin_prob,
            self.pangolin_psi,
            self.probe_none.unsqueeze(0),
            self.probe_acceptor.unsqueeze(0),
            self.probe_donor.unsqueeze(0),
        ], dim=0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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


class BiPangolinRunner:
    def __init__(self, pangolin_model_dir=None, probe_dir=None, device="auto",
                 ensemble=True, probe_tag=None, correction_k=None,
                 correction_file=None, tissue="all_tissues"):
        self.device = _resolve_device(device)
        self.pangolin_model_dir = (
            Path(pangolin_model_dir) if pangolin_model_dir else resolve_pangolin_weights()
        )
        self.probe_dir = Path(probe_dir) if probe_dir else resolve_probe_weights()
        self._pair_specs = []
        self.correction_k = self._resolve_correction_k(correction_k, correction_file)

        valid_tissues = ("all_tissues",) + TISSUE_NAMES
        if tissue not in valid_tissues:
            raise ValueError(f"tissue must be one of {valid_tissues}, got {tissue!r}")

        candidates = []
        for p in sorted(self.pangolin_model_dir.glob("final.*.v2")):
            m = PANGOLIN_FILE_RE.match(p.name)
            if not m:
                continue
            tissue_idx = int(m.group(2)) // 2
            if tissue != "all_tissues" and TISSUE_NAMES[tissue_idx] != tissue:
                continue
            candidates.append((p, tissue_idx))

        if not candidates:
            raise FileNotFoundError(f"No Pangolin models found in {self.pangolin_model_dir}")
        if not ensemble:
            candidates = candidates[:1]

        for pangolin_path, tissue_idx in candidates:
            pattern = f"probe.{pangolin_path.name}.*.pt"
            probe_paths = sorted(self.probe_dir.glob(pattern))
            if probe_tag is not None:
                probe_paths = [p for p in probe_paths if probe_tag in p.name]
            if not probe_paths:
                raise FileNotFoundError(f"No probe matched for {pangolin_path.name} in {self.probe_dir}")
            self._pair_specs.append((pangolin_path, probe_paths[-1], tissue_idx)) # Load latest match

        self.tissues_present = sorted({t for _, _, t in self._pair_specs})
        self.tissue_names = tuple(TISSUE_NAMES[t] for t in self.tissues_present)
        print(f"biPangolin: {len(self._pair_specs)} model+probe pairs ready on {self.device}")
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
        return float(correction["empirical_sweep"]["best_k"])

    def _apply_correction(self, probe_probs):
        if self.correction_k is None or self.correction_k == 1.0:
            return probe_probs

        corrected = probe_probs.clone()
        corrected[NONE_CLASS] *= self.correction_k
        return corrected / corrected.sum(dim=0, keepdim=True).clamp_min(1e-12)

    def _iter_pairs(self):
        for pangolin_path, probe_path, tissue_idx in self._pair_specs:
            pangolin_model = load_frozen_pangolin(pangolin_path, self.device)
            probe, cfg = load_probe(probe_path, self.device)
            
            layers = parse_probe_layers(cfg["probe_layer"])
            handles = attach_hooks(pangolin_model, layers)
            
            try:
                yield pangolin_model, probe, handles, cfg, tissue_idx
            finally:
                del pangolin_model, probe
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

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
        padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
        seq_t = one_hot_encode(padded).unsqueeze(0).to(self.device)
        L_out = len(seq)

        prob_sums = {t: torch.zeros(L_out, device=self.device) for t in self.tissues_present}
        psi_sums  = {t: torch.zeros(L_out, device=self.device) for t in self.tissues_present}
        count     = {t: 0 for t in self.tissues_present}
        probe_sum = torch.zeros(3, L_out, device=self.device)
        n_pairs   = 0

        for pangolin_model, probe, handles, cfg, tissue_idx in self._iter_pairs():
            pangolin_out, probe_probs = self._forward_one(
                pangolin_model, probe, handles, cfg, seq_t)
            prob_sums[tissue_idx] += pangolin_out[PROB_CHANNEL_PER_TISSUE[tissue_idx]]
            psi_sums[tissue_idx]  += pangolin_out[PSI_CHANNEL_PER_TISSUE[tissue_idx]]
            count[tissue_idx] += 1
            probe_sum += probe_probs
            n_pairs += 1

        prob_stack = torch.stack([prob_sums[t] / count[t] for t in self.tissues_present])
        psi_stack  = torch.stack([psi_sums[t]  / count[t] for t in self.tissues_present])
        probe_avg  = self._apply_correction(probe_sum / n_pairs)

        return BiPangolinResult(
            pangolin_prob=prob_stack.cpu(),
            pangolin_psi=psi_stack.cpu(),
            probe_none=probe_avg[NONE_CLASS].cpu(),
            probe_acceptor=probe_avg[ACC_CLASS].cpu(),
            probe_donor=probe_avg[DON_CLASS].cpu(),
            tissues=self.tissue_names,
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
