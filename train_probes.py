"""
Train a donor/acceptor/none probe on Pangolin's penultimate (skip) activations.

Pipeline per Pangolin model file:
  1. Parse GTF for + and - strand splice sites and gene extents.
  2. For each gene, build a region from
        [min_splice_site - 5000, max_splice_site + 5000]
     and tile that region into overlapping windows.
  3. Within each window, label ALL relevant-strand splice sites that fall in
     its usable region — including those from neighbouring/overlapping genes
     so the probe doesn't see real sites mislabelled as 'none'.
  4. Run frozen Pangolin once per window; cache activations at donor + acceptor
     positions and a subsample of "none" positions. Optionally also cache the
     raw one-hot sequence in a K_MAX-wide window around each labelled position,
     concatenated with the activation channels.
  5. Train Conv1d(in_channels, 3, kernel_size) on the cached tensors.

Train/val/test split is by chromosome (mirrors Pangolin's own held-out
chromosomes for the test set).
"""
from bisect import bisect_left, bisect_right
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyfastx
from pangolin.model import Pangolin, L, W, AR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

WINDOW_LEN = 20000
PANGOLIN_CROP = 5000
USABLE_LEN = WINDOW_LEN - 2 * PANGOLIN_CROP   # = 10000
DEFAULT_OVERLAP = 2000
GENE_FLANK = 5000

K_MAX_RADIUS = 0
K_MAX = 2 * K_MAX_RADIUS + 1   

# Class encoding
NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2

TRAIN_CHROMS = {f"chr{c}" for c in [2, 4, 5, 6, 8] + list(range(10, 23))}
VAL_CHROMS = {"chr3", "chr7"}
TEST_CHROMS = {"chr1", "chr9"}


# ---------------------------------------------------------------------------
# Sequence Utilities & One-hot encoding
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]

_BASE_TO_IDX = {b: i for i, b in enumerate("NACGT")}
IN_MAP = torch.tensor([
    [0, 0, 0, 0],   # N
    [1, 0, 0, 0],   # A
    [0, 1, 0, 0],   # C
    [0, 0, 1, 0],   # G
    [0, 0, 0, 1],   # T
], dtype=torch.float32)


def one_hot_encode(seq: str) -> torch.Tensor:
    seq = seq.upper()
    idx = torch.tensor([_BASE_TO_IDX.get(b, 0) for b in seq], dtype=torch.long)
    return IN_MAP[idx].T.contiguous()        # (4, len)


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

def parse_gtf(gtf_path, chroms=None, strand_to_parse="+"):
    """Parse a GTF for exons on the specified strand.

    Uses transcript-level exon ordering to exclude:
      - The acceptor position of each transcript's first exon (TSS, not a real acceptor)
      - The donor position of each transcript's last exon (TTS, not a real donor)

    Returns:
      sites: dict[chrom] -> dict[pos_0based] -> ACC_CLASS | DON_CLASS
      genes: dict[chrom] -> list[(gene_id, set_of_splice_site_positions)]
    """
    # Pass 1: collect all exons grouped by transcript
    transcript_exons = {}
    transcript_gene = {}     # transcript_id -> gene_id
    transcript_chrom = {}    # transcript_id -> chrom

    n_lines = 0
    n_exons = 0
    with open(gtf_path) as fh:
        for line in fh:
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                print(f"    parsed {n_lines:,} GTF lines, {n_exons:,} {strand_to_parse} strand exons so far")
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts[:9]
            if feature != "exon" or strand != strand_to_parse:
                continue
            if chroms is not None and chrom not in chroms:
                continue
            n_exons += 1

            start, end = int(start), int(end)
            transcript_id = _extract_attr(attrs, "transcript_id")
            gene_id = _extract_attr(attrs, "gene_id")
            key = (chrom, transcript_id)
            transcript_exons.setdefault(key, []).append((start, end))
            transcript_gene[transcript_id] = gene_id
            transcript_chrom[transcript_id] = chrom

    # Pass 2: for each transcript, sort exons and assign splice site labels,
    # skipping the acceptor of the first exon and donor of the last exon.
    raw = {}       # chrom -> {pos: class | "CONFLICT"}
    gene_sites = {}  # chrom -> {gene_id: set(positions)}

    n_transcripts = 0
    n_acc_skipped = 0   # first-exon acceptors skipped
    n_don_skipped = 0   # last-exon donors skipped

    for (chrom, transcript_id), exons in transcript_exons.items():
        # Sort exons in reverse order for negative strand to process from 5' to 3'
        exons_sorted = sorted(exons, key=lambda e: e[0], reverse=(strand_to_parse == "-"))
        gene_id = transcript_gene[transcript_id]
        n_transcripts += 1

        chrom_raw = raw.setdefault(chrom, {})
        chrom_genes = gene_sites.setdefault(chrom, {})
        site_set = chrom_genes.setdefault(gene_id, set())

        for i, (start, end) in enumerate(exons_sorted):
            if strand_to_parse == "+":
                acc_pos = start - 1   # first exonic base (0-based)
                don_pos = end - 1     # last exonic base (0-based)
            else:
                acc_pos = end - 1     # first exonic base for negative strand
                don_pos = start - 1   # last exonic base for negative strand

            is_first = (i == 0)
            is_last  = (i == len(exons_sorted) - 1)

            # Always add to gene region (for window tiling)
            site_set.add(acc_pos)
            site_set.add(don_pos)

            # Label acceptor — skip for first exon (TSS, not a real splice acceptor)
            if not is_first:
                prev = chrom_raw.get(acc_pos)
                if prev is None:
                    chrom_raw[acc_pos] = ACC_CLASS
                elif prev != ACC_CLASS:
                    chrom_raw[acc_pos] = "CONFLICT"
            else:
                n_acc_skipped += 1

            # Label donor — skip for last exon (TTS, not a real splice donor)
            if not is_last:
                prev = chrom_raw.get(don_pos)
                if prev is None:
                    chrom_raw[don_pos] = DON_CLASS
                elif prev != DON_CLASS:
                    chrom_raw[don_pos] = "CONFLICT"
            else:
                n_don_skipped += 1

    sites = {chrom: {p: c for p, c in d.items() if c != "CONFLICT"}
             for chrom, d in raw.items()}
    genes = {chrom: list(d.items()) for chrom, d in gene_sites.items()}
    n_sites_total = sum(len(d) for d in sites.values())
    n_genes_total = sum(len(d) for d in genes.values())
    print(f"    {n_exons:,} {strand_to_parse} strand exons across {n_transcripts:,} transcripts -> "
          f"{n_sites_total:,} clean splice sites in {n_genes_total:,} genes "
          f"across {len(sites)} chroms "
          f"(skipped {n_acc_skipped:,} TSS acceptors, {n_don_skipped:,} TTS donors)")
    return sites, genes


def _extract_attr(attrs, key):
    needle = key + ' "'
    i = attrs.find(needle)
    if i < 0:
        return None
    j = attrs.find('"', i + len(needle))
    return attrs[i + len(needle):j] if j > 0 else None


# ---------------------------------------------------------------------------
# Window tiling
# ---------------------------------------------------------------------------

def tile_windows(sites_by_chrom, genes_by_chrom, fasta,
                 none_subsample_ratio=10, overlap=DEFAULT_OVERLAP,
                 seed=0, max_genes=None, strand="+"):
    assert overlap < USABLE_LEN, "overlap must be < usable region length"
    rng = random.Random(seed)
    stride = WINDOW_LEN - overlap
    records = []

    sorted_sites = {chrom: sorted(d.keys()) for chrom, d in sites_by_chrom.items()}

    if max_genes is not None:
        flat = [(chrom, gene_id, site_set)
                for chrom, gene_list in genes_by_chrom.items()
                for gene_id, site_set in gene_list]
        if len(flat) > max_genes:
            flat = rng.sample(flat, max_genes)
        genes_by_chrom = {}
        for chrom, gene_id, site_set in flat:
            genes_by_chrom.setdefault(chrom, []).append((gene_id, site_set))
        print(f"    sampled {sum(len(v) for v in genes_by_chrom.values())} genes "
              f"across {len(genes_by_chrom)} chroms (max_genes={max_genes})")

    seen_window_starts = {}

    for chrom, gene_list in genes_by_chrom.items():
        if chrom not in fasta:
            continue
        chrom_len = len(fasta[chrom])
        chrom_sites = sites_by_chrom.get(chrom, {})
        chrom_sorted = sorted_sites.get(chrom, [])
        chrom_seen = seen_window_starts.setdefault(chrom, set())

        for _gene_id, site_set in tqdm(gene_list, desc=f"  tiling {chrom}",
                                       leave=False, unit="gene"):
            if not site_set:
                continue
            region_start = max(0, min(site_set) - GENE_FLANK)
            region_end = min(chrom_len, max(site_set) + GENE_FLANK)

            w_starts = list(range(region_start,
                                  max(region_start + 1, region_end - WINDOW_LEN + 1),
                                  stride))
            last_start = max(region_start, region_end - WINDOW_LEN)
            if not w_starts or w_starts[-1] < last_start:
                w_starts.append(last_start)

            for w_start in w_starts:
                if w_start in chrom_seen:
                    continue
                if w_start + WINDOW_LEN > chrom_len:
                    continue
                chrom_seen.add(w_start)

                usable_start = w_start + PANGOLIN_CROP
                usable_end = usable_start + USABLE_LEN

                lo = bisect_left(chrom_sorted, usable_start)
                hi = bisect_right(chrom_sorted, usable_end - 1)
                pos_labels = [(p - w_start, chrom_sites[p])
                              for p in chrom_sorted[lo:hi]]

                if not pos_labels:
                    continue

                n_pos = len(pos_labels)
                taken = {p for p, _ in pos_labels}
                candidate_offsets = [off for off in range(PANGOLIN_CROP, PANGOLIN_CROP + USABLE_LEN)
                                     if off not in taken]
                n_neg = min(none_subsample_ratio * n_pos, len(candidate_offsets))
                if n_neg > 0:
                    for off in rng.sample(candidate_offsets, n_neg):
                        pos_labels.append((off, NONE_CLASS))

                centers = torch.tensor([p for p, _ in pos_labels], dtype=torch.long)
                labels = torch.tensor([c for _, c in pos_labels], dtype=torch.long)
                records.append((chrom, w_start, centers, labels, strand))

    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, records, fasta_path):
        self.records = records
        self.fasta_path = fasta_path
        self._fasta = None

    def _get_fasta(self):
        if self._fasta is None:
            self._fasta = pyfastx.Fasta(self.fasta_path)
        return self._fasta

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        chrom, w_start, centers, labels, strand = self.records[idx]
        fasta = self._get_fasta()
        seq = fasta[chrom][w_start:w_start + WINDOW_LEN].seq
        
        if strand == "-":
            seq = reverse_complement(seq)
            centers = WINDOW_LEN - 1 - centers

        return one_hot_encode(seq), centers, labels


def collate_windows(batch):
    seqs = torch.stack([item[0] for item in batch])
    centers_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    return seqs, centers_list, labels_list


# ---------------------------------------------------------------------------
# Pangolin loading & hooks
# ---------------------------------------------------------------------------

def load_frozen_pangolin(weights_path, device):
    model = Pangolin(L, W, AR)
    map_loc = device if device.type == "cuda" else torch.device("cpu")
    model.load_state_dict(torch.load(weights_path, map_location=map_loc))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


PROBE_LAYERS = ("skip", "resblock_15", "resblock_11", "resblock_7", "resblock_3", "resblock_1")


def parse_probe_layers(probe_layer):
    """Parse a "+"-separated string or list of layer names into a list."""
    if isinstance(probe_layer, str):
        layers = probe_layer.split("+")
    else:
        layers = list(probe_layer)
    for layer in layers:
        if layer not in PROBE_LAYERS:
            raise ValueError(f"probe_layer must be from {PROBE_LAYERS}, got {layer}")
    return layers


def attach_hooks(model, probe_layers):
    """Register hooks for one or more layers.

    Returns dict[layer_name] -> {"cache": ..., "is_cropped": ...}.
    """
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
            assert 0 <= idx < len(model.resblocks), f"resblock index {idx} out of range"
            def hook(_module, _inputs, output, _cache=cache):
                _cache["activations"] = output
            model.resblocks[idx].register_forward_hook(hook)
            is_cropped = False
        handles[layer] = {"cache": cache, "is_cropped": is_cropped}
    return handles


def attach_hook(model, probe_layer="skip"):
    """Backward-compat single-layer wrapper."""
    layers = parse_probe_layers(probe_layer)
    if len(layers) != 1:
        raise ValueError("attach_hook only handles one layer; use attach_hooks for multi-layer")
    h = attach_hooks(model, layers)[layers[0]]
    return h["cache"], h["is_cropped"]


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(pangolin, handles, loader, device, desc="extracting",
                         include_sequence=False):
    """Gather activation slices (and optionally raw sequence) at each labelled position.

    For N activation layers this returns:
        include_sequence=False: (N_labels, 32*N,     K_MAX)
        include_sequence=True:  (N_labels, 32*N + 4, K_MAX)

    The 4 extra channels are the one-hot encoded K_MAX-nt window of the input
    sequence centred on each labelled position. This lets the probe use raw
    local sequence as an additional feature alongside Pangolin's representations.

    seqs is (B, 4, WINDOW_LEN); labelled positions are window offsets so we
    index directly into seqs without any crop adjustment.
    """
    import time
    all_acts, all_labels = [], []
    n_labels = 0
    n_windows = 0
    t_start = time.time()
    pbar = tqdm(loader, desc=desc, unit="batch")
    layer_names = list(handles.keys())
    n_act_channels = 32 * len(layer_names)
    if include_sequence:
        n_act_channels += 4

    for seqs, centers_list, labels_list in pbar:
        seqs = seqs.to(device)                                # (B, 4, WINDOW_LEN)
        _ = pangolin(seqs)

        # Normalise each activation layer to (B, 32, USABLE_LEN) and pad
        padded_per_layer = []
        for layer in layer_names:
            acts = handles[layer]["cache"]["activations"]     # (B, 32, ?)
            if not handles[layer]["is_cropped"]:
                acts = acts[..., PANGOLIN_CROP:PANGOLIN_CROP + USABLE_LEN]
            padded_per_layer.append(F.pad(acts, (K_MAX_RADIUS, K_MAX_RADIUS)))

        # Pad seqs along sequence dim for safe K_MAX slicing (coords are in WINDOW_LEN space)
        if include_sequence:
            seqs_padded = F.pad(seqs, (K_MAX_RADIUS, K_MAX_RADIUS))   # (B, 4, WINDOW_LEN+K_MAX-1)

        for b, (centers, labels) in enumerate(zip(centers_list, labels_list)):
            # centers are window offsets [PANGOLIN_CROP, PANGOLIN_CROP+USABLE_LEN)
            # For activation layers: shift into USABLE_LEN space
            act_idx = centers.to(device) - PANGOLIN_CROP
            act_base = act_idx + K_MAX_RADIUS
            act_offsets = act_base.unsqueeze(1) + torch.arange(
                -K_MAX_RADIUS, K_MAX_RADIUS + 1, device=device).unsqueeze(0)

            per_layer = []
            for padded in padded_per_layer:
                slices = padded[b, :, act_offsets]                       # (32, n, K_MAX)
                per_layer.append(slices.permute(1, 0, 2).contiguous())   # (n, 32, K_MAX)

            if include_sequence:
                # centers are WINDOW_LEN offsets; +K_MAX_RADIUS shifts into padded seq coords
                seq_base = centers.to(device) + K_MAX_RADIUS
                seq_offsets = seq_base.unsqueeze(1) + torch.arange(
                    -K_MAX_RADIUS, K_MAX_RADIUS + 1, device=device).unsqueeze(0)
                seq_slices = seqs_padded[b, :, seq_offsets]              # (4, n, K_MAX)
                per_layer.append(seq_slices.permute(1, 0, 2).contiguous())  # (n, 4, K_MAX)

            stacked = torch.cat(per_layer, dim=1)                        # (n, 32*N[+4], K_MAX)
            all_acts.append(stacked.cpu())
            all_labels.append(labels)
            n_labels += labels.numel()

        n_windows += seqs.size(0)
        if hasattr(pbar, "set_postfix"):
            elapsed = time.time() - t_start
            rate = n_windows / max(elapsed, 1e-6)
            pbar.set_postfix(windows=n_windows, labels=n_labels,
                             win_per_s=f"{rate:.2f}", ch=n_act_channels)
    elapsed = time.time() - t_start
    print(f"    {desc}: {n_windows} windows in {elapsed:.1f}s "
          f"({n_windows/max(elapsed,1e-6):.2f} win/s); {n_labels:,} labels; "
          f"{len(layer_names)} layers {'+ seq ' if include_sequence else ''}"
          f"-> {n_act_channels} channels")
    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def make_probe(kernel_size=1, hidden_dim=None, in_channels=32):
    """Conv1d probe. in_channels = 32*N_layers [+ 4 if include_sequence]."""
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert kernel_size <= K_MAX, f"kernel_size must be <= K_MAX={K_MAX}"
    pad = kernel_size // 2
    if hidden_dim:
        return nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 3, kernel_size, padding=pad),
        )
    return nn.Conv1d(in_channels, 3, kernel_size, padding=pad)


def slice_for_kernel(acts, kernel_size):
    """Trim (N, C, K_MAX) -> (N, C, kernel_size), centered."""
    if kernel_size == K_MAX:
        return acts
    pad = (K_MAX - kernel_size) // 2
    return acts[:, :, pad:pad + kernel_size]


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe_on_cache(train_acts, train_labels, val_acts, val_labels,
                         device, epochs=50, lr=1e-3, batch_size=4096,
                         kernel_size=1, hidden_dim=None, patience=5):
    """Train probe. in_channels detected automatically from train_acts.shape[1]."""
    train_acts = slice_for_kernel(train_acts, kernel_size)
    val_acts = slice_for_kernel(val_acts, kernel_size)

    in_channels = train_acts.shape[1]
    n_layers = in_channels // 32
    has_seq = (in_channels % 32 == 4)
    probe = make_probe(kernel_size=kernel_size, hidden_dim=hidden_dim,
                       in_channels=in_channels).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(train_acts, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_acts_d = val_acts.to(device)
    val_labels_d = val_labels.to(device)
    center = kernel_size // 2

    n_params = sum(p.numel() for p in probe.parameters())
    print(f"  probe: in_channels={in_channels} ({n_layers} act layers"
          f"{' + seq' if has_seq else ''}) "
          f"kernel_size={kernel_size} hidden_dim={hidden_dim} "
          f"params={n_params} patience={patience}")

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0
        for acts, labels in train_loader:
            acts, labels = acts.to(device), labels.to(device)
            logits = probe(acts)[:, :, center]
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        probe.eval()
        with torch.no_grad():
            logits = probe(val_acts_d)[:, :, center]
            val_loss = loss_fn(logits, val_labels_d).item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            full_acc = (preds == val_labels_d).float().mean().item()
            per_class = []
            for c, name in ((NONE_CLASS, "none"), (ACC_CLASS, "acc"), (DON_CLASS, "don")):
                c_mask = val_labels_d == c
                if c_mask.any():
                    per_class.append(f"{name}:{(preds[c_mask] == c).float().mean():.3f}")

            mask_filt = probs[:, NONE_CLASS] < 0.95
            if mask_filt.any():
                f_logits = logits[mask_filt]
                f_labels = val_labels_d[mask_filt]
                bin_metrics = []
                for c_true, c_alt, name in [(ACC_CLASS, DON_CLASS, "acc>don"),
                                            (DON_CLASS, ACC_CLASS, "don>acc")]:
                    m = f_labels == c_true
                    if m.any():
                        pref = (f_logits[m, c_true] > f_logits[m, c_alt]).float().mean()
                        bin_metrics.append(f"{name}:{pref:.3f}")
                bin_str = f"FILTERED (n={int(mask_filt.sum())}) [{', '.join(bin_metrics)}]"
            else:
                bin_str = "FILTERED: No sites passed threshold"

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            epochs_no_improve = 0
            improved_marker = " *"
        else:
            epochs_no_improve += 1
            improved_marker = f" (no improvement {epochs_no_improve}/{patience})"

        print(f"  epoch {epoch+1:3d}/{epochs}  train_loss={epoch_loss/max(n_batches,1):.4f}  "
              f"val_loss={val_loss:.4f}{improved_marker}")
        print(f"    FULL (acc={full_acc:.4f}): [{', '.join(per_class)}]")
        print(f"    {bin_str}")

        if epochs_no_improve >= patience:
            print(f"  early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return probe


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_loaders(fasta_path, gtf_path, none_subsample_ratio, overlap, batch_size,
                  max_genes_train=None, max_genes_val=None, max_genes_test=None):
    fasta = pyfastx.Fasta(fasta_path)

    print("Parsing GTF and tiling windows...")
    train_recs, val_recs, test_recs = [], [], []
    
    for strand in ["+", "-"]:
        print(f"--- Processing {strand} strand ---")
        train_sites, train_genes = parse_gtf(gtf_path, chroms=TRAIN_CHROMS, strand_to_parse=strand)
        val_sites, val_genes = parse_gtf(gtf_path, chroms=VAL_CHROMS, strand_to_parse=strand)
        test_sites, test_genes = parse_gtf(gtf_path, chroms=TEST_CHROMS, strand_to_parse=strand)

        train_recs += tile_windows(train_sites, train_genes, fasta, none_subsample_ratio, overlap,
                                   max_genes=max_genes_train, strand=strand)
        val_recs += tile_windows(val_sites, val_genes, fasta, none_subsample_ratio, overlap,
                                 max_genes=max_genes_val, strand=strand)
        test_recs += tile_windows(test_sites, test_genes, fasta, none_subsample_ratio, overlap,
                                  max_genes=max_genes_test, strand=strand)
        
    print(f"  Total train: {len(train_recs)} windows")
    print(f"  Total val:   {len(val_recs)} windows")
    print(f"  Total test:  {len(test_recs)} windows")

    def make_loader(recs):
        return DataLoader(WindowDataset(recs, fasta_path),
                          batch_size=batch_size, shuffle=False,
                          collate_fn=collate_windows)

    return make_loader(train_recs), make_loader(val_recs), make_loader(test_recs)


def run_one_model(weights_path, train_loader, val_loader, test_loader,
                  device, cache_dir=None, probe_layer="skip",
                  kernel_size=1, hidden_dim=None, include_sequence=False):
    layers = parse_probe_layers(probe_layer)
    layer_tag = "+".join(layers)
    if include_sequence:
        layer_tag += "+seq"

    pangolin = load_frozen_pangolin(weights_path, device)
    handles = attach_hooks(pangolin, layers)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"acts.{Path(weights_path).name}.{layer_tag}.pt"

    if cache_path and cache_path.exists():
        print(f"  loading cached activations from {cache_path}")
        blob = torch.load(cache_path)
        train_acts, train_labels = blob["train_acts"], blob["train_labels"]
        val_acts, val_labels = blob["val_acts"], blob["val_labels"]
        test_acts, test_labels = blob["test_acts"], blob["test_labels"]
    else:
        print(f"  extracting activations: layers={layers} include_sequence={include_sequence}")
        train_acts, train_labels = extract_activations(
            pangolin, handles, train_loader, device, desc="  train",
            include_sequence=include_sequence)
        val_acts, val_labels = extract_activations(
            pangolin, handles, val_loader, device, desc="  val",
            include_sequence=include_sequence)
        test_acts, test_labels = extract_activations(
            pangolin, handles, test_loader, device, desc="  test",
            include_sequence=include_sequence)
        if cache_path:
            torch.save({"train_acts": train_acts, "train_labels": train_labels,
                        "val_acts": val_acts, "val_labels": val_labels,
                        "test_acts": test_acts, "test_labels": test_labels},
                       cache_path)

    del pangolin
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def fmt_dist(labels):
        counts = torch.bincount(labels, minlength=3).tolist()
        return f"none={counts[NONE_CLASS]:,} acc={counts[ACC_CLASS]:,} don={counts[DON_CLASS]:,}"
    print(f"  train labels: {fmt_dist(train_labels)}")
    print(f"  val   labels: {fmt_dist(val_labels)}")
    print(f"  test  labels: {fmt_dist(test_labels)}")

    probe = train_probe_on_cache(train_acts, train_labels, val_acts, val_labels, device,
                                 kernel_size=kernel_size, hidden_dim=hidden_dim)

    probe.eval()
    test_acts_sliced = slice_for_kernel(test_acts, kernel_size)
    center = kernel_size // 2
    with torch.no_grad():
        preds = probe(test_acts_sliced.to(device))[:, :, center].argmax(dim=1)
        acc = (preds == test_labels.to(device)).float().mean().item()
        print(f"  TEST acc={acc:.4f}")
    return probe


def main(model_dir, fasta_path, gtf_path, out_dir,
         cache_dir=None, batch_size=4,
         none_subsample_ratio=10, overlap=DEFAULT_OVERLAP,
         max_genes_train=None, max_genes_val=None, max_genes_test=None,
         probe_layer="skip", kernel_size=1, hidden_dim=None,
         include_sequence=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print(f"device:           {device}")
    print(f"torch:            {torch.__version__}")
    print(f"window_len:       {WINDOW_LEN}  (usable {USABLE_LEN}, crop {PANGOLIN_CROP})")
    print(f"overlap:          {overlap}")
    print(f"batch_size:       {batch_size}")
    print(f"none_subsample:   {none_subsample_ratio}x positive")
    print(f"max_genes:        train={max_genes_train} val={max_genes_val} test={max_genes_test}")
    print(f"probe_layer:      {probe_layer}")
    print(f"include_sequence: {include_sequence}")
    print(f"kernel_size:      {kernel_size}  (cache K_MAX={K_MAX})")
    print(f"hidden_dim:       {hidden_dim}")
    print(f"train chroms:     {sorted(TRAIN_CHROMS)}")
    print(f"val chroms:       {sorted(VAL_CHROMS)}")
    print(f"test chroms:      {sorted(TEST_CHROMS)}")
    print(f"fasta:            {fasta_path}")
    print(f"gtf:              {gtf_path}")
    print(f"model_dir:        {model_dir}")
    print(f"cache_dir:        {cache_dir}")
    print(f"out_dir:          {out_dir}")
    print("=" * 60)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(
        fasta_path, gtf_path, none_subsample_ratio, overlap, batch_size,
        max_genes_train=max_genes_train,
        max_genes_val=max_genes_val,
        max_genes_test=max_genes_test)

    for mf in sorted(Path(model_dir).glob("final.*.v2")):
        print(f"=== {mf.name} ===")
        probe = run_one_model(mf, train_loader, val_loader, test_loader,
                              device, cache_dir, probe_layer=probe_layer,
                              kernel_size=kernel_size, hidden_dim=hidden_dim,
                              include_sequence=include_sequence)
        seq_tag = "+seq" if include_sequence else ""
        tag = f"{probe_layer}{seq_tag}.k{kernel_size}.h{hidden_dim}"
        out_path = out_dir / f"probe.{mf.name}.{tag}.pt"
        torch.save({
            "state_dict": probe.state_dict(),
            "config": {
                "probe_layer": probe_layer,
                "kernel_size": kernel_size,
                "hidden_dim": hidden_dim,
                "include_sequence": include_sequence,
                "pangolin_model_file": mf.name,
            },
        }, out_path)


if __name__ == "__main__":
    model_dir = "/camp/home/wilkino/home/POSTDOC/software/biPangolin/Pangolin/pangolin/models/"
    fasta_path = "/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa"
    gtf_path = "/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf"
    out_dir = "./bipangolin_probes"
    cache_dir = "./bipangolin_cache"

    print(f"Starting biPangolin Extraction and Training...")

    main(
        model_dir=model_dir,
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        out_dir=out_dir,
        cache_dir=cache_dir,
        batch_size=32,
        none_subsample_ratio=100,
        max_genes_train=None,
        max_genes_val=None,
        max_genes_test=None,
        probe_layer=PROBE_LAYERS,
        include_sequence=True,
        kernel_size=1,
        hidden_dim=64,
    )