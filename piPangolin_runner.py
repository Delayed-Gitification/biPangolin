import torch
import torch.nn as nn
import re
from pathlib import Path
from dataclasses import dataclass
from pangolin.model import Pangolin, L, W, AR

# Constants
PANGOLIN_CROP = 5000
WINDOW_LEN_DEFAULT = 20000
TISSUE_NAMES = ("heart", "liver", "brain", "testis")
PROB_CHANNEL_MAP = [1, 4, 7, 10]
PSI_CHANNEL_MAP  = [2, 5, 8, 11]
NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2

PANGOLIN_FILE_RE = re.compile(r"^final\.([1-3])\.([0246])\.3\.v2$")

_BASE_TO_IDX = {b: i for i, b in enumerate("NACGT")}
_IN_MAP = torch.tensor([
    [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
], dtype=torch.float32)

def one_hot_encode(seq: str) -> torch.Tensor:
    idx = torch.tensor([_BASE_TO_IDX.get(b.upper(), 0) for b in seq], dtype=torch.long)
    return _IN_MAP[idx].T.contiguous()

@dataclass
class BiPangolinResult:
    pangolin_prob: torch.Tensor
    pangolin_psi: torch.Tensor
    probe_none: torch.Tensor
    probe_acceptor: torch.Tensor
    probe_donor: torch.Tensor
    tissues: tuple

class BiPangolinRunner:
    def __init__(self, pangolin_model_dir, probe_dir, device="auto", 
                 ensemble=True, tissue="all_tissues"):
        
        self.pangolin_model_dir = Path(pangolin_model_dir)
        self.probe_dir = Path(probe_dir)
        
        # Assertions for user-friendliness
        assert self.pangolin_model_dir.exists(), f"Pangolin models not found at {pangolin_model_dir}"
        assert self.probe_dir.exists(), f"Probes not found at {probe_dir}"
        valid_tissues = ["all_tissues"] + list(TISSUE_NAMES)
        assert tissue in valid_tissues, f"Tissue must be one of {valid_tissues}"

        self.device = self._resolve_device(device)
        self._pair_specs = []

        candidates = []
        for p in sorted(self.pangolin_model_dir.glob("final.*.v2")):
            m = PANGOLIN_FILE_RE.match(p.name)
            if not m: continue
            
            t_idx = int(m.group(2)) // 2
            if tissue != "all_tissues" and TISSUE_NAMES[t_idx] != tissue:
                continue
            candidates.append((p, t_idx))

        assert len(candidates) > 0, f"No Pangolin models found for tissue '{tissue}'"
        if not ensemble: candidates = candidates[:1]

        # Match Probes - Updated to be more flexible
        for p_path, t_idx in candidates:
            # Look for the latest version of the probe for this specific model
            probe_matches = sorted(self.probe_dir.glob(f"probe.{p_path.name}.*.pt"))
            
            if probe_matches:
                self._pair_specs.append((p_path, probe_matches[-1], t_idx))
            else:
                # Just skip it and let the user know, rather than crashing
                print(f"  Note: No probe found for {p_path.name}, skipping this tissue.")

        assert len(self._pair_specs) > 0, (
            f"No valid model+probe pairs found in {self.probe_dir}. "
            "Ensure your probes are named 'probe.final.X.Y.3.v2...'"
        )

        self.tissues_present = sorted({t for _, _, t in self._pair_specs})
        self.tissue_names = tuple(TISSUE_NAMES[t] for t in self.tissues_present)
        print(f"biPangolin: {len(self._pair_specs)} pairs ({tissue}) ready on {self.device}")

    def _resolve_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _attach_hooks(self, model, layers):
        handles = {}
        for layer in layers:
            cache = {}
            if layer == "skip":
                model.conv_last1.register_forward_pre_hook(lambda m, i, c=cache: c.update({"acts": i[0]}))
                is_cropped = True
            else:
                idx = int(layer.split("_")[1])
                model.resblocks[idx].register_forward_hook(lambda m, i, o, c=cache: c.update({"acts": o}))
                is_cropped = False
            handles[layer] = {"cache": cache, "is_cropped": is_cropped}
        return handles

    @torch.no_grad()
    def score_sequence(self, seq):
        padded = "N" * PANGOLIN_CROP + seq + "N" * PANGOLIN_CROP
        seq_t = one_hot_encode(padded).unsqueeze(0).to(self.device)
        L_out = len(seq)

        # Storage for ensemble results
        p_sums = {t: torch.zeros(L_out) for t in self.tissues_present}
        probe_sum = torch.zeros(3, L_out)

        for p_path, pr_path, t_idx in self._pair_specs:
            # 1. Load Pangolin
            m = Pangolin(L, W, AR)
            m.load_state_dict(torch.load(p_path, map_location=self.device))
            m.to(self.device).eval()
            
            # 2. Load Probe and Config
            blob = torch.load(pr_path, map_location=self.device)
            cfg = blob["config"]
            
            # Robustly handle if probe_layer is a string or a tuple/list
            layers = cfg["probe_layer"]
            if isinstance(layers, str):
                layers = layers.split("+")
            
            # 3. Reconstruct Probe Architecture to match training
            in_ch = 32 * len(layers) + (4 if cfg.get("include_sequence") else 0)
            k = cfg["kernel_size"]
            pad = k // 2
            
            if cfg.get("hidden_dim"):
                probe = nn.Sequential(
                    nn.Conv1d(in_ch, cfg["hidden_dim"], k, padding=pad),
                    nn.ReLU(),
                    nn.Conv1d(cfg["hidden_dim"], 3, k, padding=pad)
                )
            else:
                probe = nn.Conv1d(in_ch, 3, k, padding=pad)
            
            probe.load_state_dict(blob["state_dict"])
            probe.to(self.device).eval()

            # 4. Run Inference
            handles = self._attach_hooks(m, layers)
            p_out = m(seq_t)[0]
            
            acts = []
            for l_name in layers:
                val = handles[l_name]["cache"]["acts"]
                if not handles[l_name]["is_cropped"]: 
                    val = val[..., PANGOLIN_CROP:PANGOLIN_CROP+L_out]
                acts.append(val)
            
            if cfg.get("include_sequence"):
                acts.append(seq_t[..., PANGOLIN_CROP:PANGOLIN_CROP+L_out])
            
            # 5. Accumulate Results
            pr_logits = probe(torch.cat(acts, dim=1))
            pr_probs = torch.softmax(pr_logits, dim=1)[0]
            
            p_sums[t_idx] += p_out[PROB_CHANNEL_MAP[t_idx]].cpu()
            probe_sum += pr_probs.cpu()
            
            del m, probe # Clean up VRAM

        # 6. Final Averaging
        pair_t_indices = [spec[2] for spec in self._pair_specs]
        
        return BiPangolinResult(
            pangolin_prob=torch.stack([
                p_sums[t] / pair_t_indices.count(t) for t in self.tissues_present
            ]),
            pangolin_psi=torch.zeros(1, L_out), 
            probe_none=probe_sum[NONE_CLASS] / len(self._pair_specs),
            probe_acceptor=probe_sum[ACC_CLASS] / len(self._pair_specs),
            probe_donor=probe_sum[DON_CLASS] / len(self._pair_specs),
            tissues=self.tissue_names
        )

# --- TEST EXECUTION BLOCK ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("models")
    parser.add_argument("probes")
    parser.add_argument("--tissue", default="all_tissues")
    args = parser.parse_args()

    CALIBRATION_SEQ = (
        "cacagcaccggcggcatggacgagctgtacaaggactacaaggacgatgatgacaagtgataaacaaatggt"
        "aaggaagggcacatcaatctttgcttaattgtcctttactctaaagatgtattttatcatactgaatgctaa"
        "acttgatatctccttttaggtcattgatgtccttcaccccgggaaggcgacagtgcctaagacagaaattcgg"
    ).upper()

    runner = BiPangolinRunner(args.models, args.probes, tissue=args.tissue)
    result = runner.score_sequence(CALIBRATION_SEQ)

    print(f"\nCalibration Result ({len(CALIBRATION_SEQ)} bp):")
    print(f"  Donor (Exp 69):    {result.probe_donor.argmax()} (P={result.probe_donor.max():.3f})")
    print(f"  Acceptor (Exp 163): {result.probe_acceptor.argmax()} (P={result.probe_acceptor.max():.3f})")