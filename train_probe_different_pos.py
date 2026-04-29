"""
Train a donor/acceptor/none probe on Pangolin's penultimate (skip) activations.

Pipeline per Pangolin model file:
  1. Parse GTF for + strand splice sites and gene extents (forward strand only).
  2. For each + strand gene, build a region from
        [min_splice_site - 5000, max_splice_site + 5000]
     and tile that region into overlapping windows.
  3. Within each window, label ALL forward-strand splice sites that fall in
     its usable region — including those from neighbouring/overlapping genes
     so the probe doesn't see real sites mislabelled as 'none'.
  4. Run frozen Pangolin once per window; cache 32-d skip activations at
     donor + acceptor positions and a subsample of "none" positions.
  5. Train Linear(32, 3) on the cached tensors.

Train/val/test split is by chromosome (mirrors Pangolin's own held-out
chromosomes for the test set).
"""
from bisect import bisect_left, bisect_right
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pyfastx
from pangolin.model import Pangolin, L, W, AR

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):                       # no-op fallback
        return it


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

WINDOW_LEN = 20000              # full input length per Pangolin forward pass
PANGOLIN_CROP = 5000            # bases stripped from each end of skip
USABLE_LEN = WINDOW_LEN - 2 * PANGOLIN_CROP   # = 10000
DEFAULT_OVERLAP = 2000          # overlap between adjacent windows in a gene region
GENE_FLANK = 5000               # bases upstream/downstream of first/last splice site

# Class encoding
NONE_CLASS, ACC_CLASS, DON_CLASS = 0, 1, 2

# Chromosome split — mirrors Pangolin's training/test split, with held-out
# test chromosomes carved from Pangolin's test set so the probe has truly
# unseen test data even though Pangolin saw the underlying sequences.
TRAIN_CHROMS = {f"chr{c}" for c in [2, 4, 5, 6, 8] + list(range(10, 23))}
VAL_CHROMS = {"chr3", "chr7"}
TEST_CHROMS = {"chr1", "chr9"}


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

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

def parse_gtf(gtf_path, chroms=None):
    """Parse a GTF for + strand exons.

    Returns:
      sites: dict[chrom] -> dict[pos_0based] -> ACC_CLASS | DON_CLASS
      genes: dict[chrom] -> list[(gene_id, set_of_splice_site_positions)]

    Conflict policy: if the same genomic position is annotated as both donor
    and acceptor (e.g. across overlapping transcripts), it is dropped from the
    site labels but its position is retained in the relevant gene's set so the
    gene's region still spans it.
    """
    raw = {}                           # chrom -> {pos: class | "CONFLICT"}
    gene_sites = {}                    # chrom -> {gene_id: set(positions)}

    n_lines = 0
    n_exons = 0
    with open(gtf_path) as fh:
        for line in fh:
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                print(f"    parsed {n_lines:,} GTF lines, {n_exons:,} + strand exons so far")
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts[:9]
            if feature != "exon" or strand != "+":
                continue
            if chroms is not None and chrom not in chroms:
                continue
            n_exons += 1

            start, end = int(start), int(end)
            acc_pos = start - 1        # first exonic base (0-based)
            don_pos = end - 1          # last exonic base (0-based)

            gene_id = _extract_attr(attrs, "gene_id")
            chrom_genes = gene_sites.setdefault(chrom, {})
            site_set = chrom_genes.setdefault(gene_id, set())
            site_set.add(acc_pos); site_set.add(don_pos)

            chrom_raw = raw.setdefault(chrom, {})
            for pos, cls in ((acc_pos, ACC_CLASS), (don_pos, DON_CLASS)):
                prev = chrom_raw.get(pos)
                if prev is None:
                    chrom_raw[pos] = cls
                elif prev != cls:
                    chrom_raw[pos] = "CONFLICT"

    sites = {chrom: {p: c for p, c in d.items() if c != "CONFLICT"}
             for chrom, d in raw.items()}
    genes = {chrom: list(d.items()) for chrom, d in gene_sites.items()}
    n_sites_total = sum(len(d) for d in sites.values())
    n_genes_total = sum(len(d) for d in genes.values())
    print(f"    {n_exons:,} + strand exons -> "
          f"{n_sites_total:,} unique sites in {n_genes_total:,} genes "
          f"across {len(sites)} chroms")
    return sites, genes


def _extract_attr(attrs, key):
    """Pull a single attribute value from a GTF attribute string."""
    needle = key + ' "'
    i = attrs.find(needle)
    if i < 0:
        return None
    j = attrs.find('"', i + len(needle))
    return attrs[i + len(needle):j] if j > 0 else None


# ---------------------------------------------------------------------------
# Window tiling (gene-anchored, with cross-gene labelling)
# ---------------------------------------------------------------------------

def tile_windows(sites_by_chrom, genes_by_chrom, fasta,
                 none_subsample_ratio=10, overlap=DEFAULT_OVERLAP,
                 seed=0, max_genes=None):
    """For each gene on the + strand, tile its splice-site region into windows.

    Cross-gene labelling: every window's labels include ALL + strand splice
    sites from `sites_by_chrom` that fall in its usable region, regardless of
    which gene they came from. This avoids accidentally training the probe to
    treat a real splice site as 'none' just because it sits inside another
    gene's window.

    If max_genes is set, sample that many genes (uniformly at random across
    chromosomes) instead of using all of them.

    Returns list of (chrom, w_start, centers_in_window, labels) tuples.
    centers are 0-based offsets into the WINDOW_LEN input.
    """
    assert overlap < USABLE_LEN, "overlap must be < usable region length"
    rng = random.Random(seed)
    stride = WINDOW_LEN - overlap
    records = []

    # Pre-sort splice site positions per chromosome for fast range lookup
    sorted_sites = {chrom: sorted(d.keys()) for chrom, d in sites_by_chrom.items()}

    # Optionally subsample genes globally across chromosomes
    if max_genes is not None:
        flat = [(chrom, gene_id, site_set)
                for chrom, gene_list in genes_by_chrom.items()
                for gene_id, site_set in gene_list]
        if len(flat) > max_genes:
            flat = rng.sample(flat, max_genes)
        # Rebuild as per-chrom dict so the rest of the function is unchanged
        genes_by_chrom = {}
        for chrom, gene_id, site_set in flat:
            genes_by_chrom.setdefault(chrom, []).append((gene_id, site_set))
        print(f"    sampled {sum(len(v) for v in genes_by_chrom.values())} genes "
              f"across {len(genes_by_chrom)} chroms (max_genes={max_genes})")

    seen_window_starts = {}            # chrom -> set, to dedupe overlapping genes

    for chrom, gene_list in genes_by_chrom.items():
        if chrom not in fasta:
            continue
        chrom_len = len(fasta[chrom])
        chrom_sites = sites_by_chrom.get(chrom, {})
        chrom_sorted = sorted_sites.get(chrom, [])
        chrom_seen = seen_window_starts.setdefault(chrom, set())

        for _gene_id, site_set in tqdm(gene_list,
                                       desc=f"  tiling {chrom}",
                                       leave=False, unit="gene"):
            if not site_set:
                continue
            region_start = max(0, min(site_set) - GENE_FLANK)
            region_end = min(chrom_len, max(site_set) + GENE_FLANK)

            # Tile gene region with overlap. We allow the last window to start
            # earlier than region_end so the region's tail still gets covered.
            w_starts = list(range(region_start, max(region_start + 1, region_end - WINDOW_LEN + 1), stride))
            # Ensure final window covers up to region_end
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

                # All + strand splice sites in this window's usable region
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
                records.append((chrom, w_start, centers, labels))

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
        chrom, w_start, centers, labels = self.records[idx]
        fasta = self._get_fasta()
        seq = fasta[chrom][w_start:w_start + WINDOW_LEN].seq
        return one_hot_encode(seq), centers, labels


def collate_windows(batch):
    seqs = torch.stack([item[0] for item in batch])
    centers_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    return seqs, centers_list, labels_list


# ---------------------------------------------------------------------------
# Pangolin loading & hook
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


PROBE_LAYERS = ("skip", "resblock_15", "resblock_11", "resblock_7", "resblock_3")


def attach_hook(model, probe_layer="skip"):
    """Register a hook to capture activations from one of several layers.

    Returns (cache_dict, is_cropped) where:
      cache_dict["activations"] is populated on every forward pass
      is_cropped tells the indexer whether to subtract PANGOLIN_CROP from centers.
        skip is post-crop (USABLE_LEN), resblock outputs are pre-crop (WINDOW_LEN).
    """
    if probe_layer not in PROBE_LAYERS:
        raise ValueError(f"probe_layer must be one of {PROBE_LAYERS}, got {probe_layer}")

    cache = {}

    if probe_layer == "skip":
        # Pre-hook on conv_last1: captures the cropped skip tensor (B, 32, USABLE_LEN)
        def hook(_module, inputs):
            cache["activations"] = inputs[0]
        model.conv_last1.register_forward_pre_hook(hook)
        is_cropped = True
    else:
        # Forward hook on a residual block: captures its output (B, 32, WINDOW_LEN)
        idx = int(probe_layer.split("_")[1])
        assert 0 <= idx < len(model.resblocks), f"resblock index {idx} out of range"
        def hook(_module, _inputs, output):
            cache["activations"] = output
        model.resblocks[idx].register_forward_hook(hook)
        is_cropped = False

    return cache, is_cropped


# Backward-compat alias (one call site uses it)
def attach_skip_hook(model):
    cache, _ = attach_hook(model, probe_layer="skip")
    return cache


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(pangolin, cache, loader, device, desc="extracting", is_cropped=True):
    import time
    all_acts, all_labels = [], []
    n_labels = 0
    n_windows = 0
    t_start = time.time()
    pbar = tqdm(loader, desc=desc, unit="batch")
    for seqs, centers_list, labels_list in pbar:
        seqs = seqs.to(device)
        _ = pangolin(seqs)
        skip = cache["activations"]
        # skip shape: (B, 32, USABLE_LEN) if cropped, (B, 32, WINDOW_LEN) if not.
        # centers are window offsets in [0, WINDOW_LEN). For cropped activations
        # subtract PANGOLIN_CROP to map into [0, USABLE_LEN).
        for b, (centers, labels) in enumerate(zip(centers_list, labels_list)):
            idx = centers.to(device)
            if is_cropped:
                idx = idx - PANGOLIN_CROP
            acts = skip[b, :, idx].T               # (n_labelled, 32)
            all_acts.append(acts.cpu())
            all_labels.append(labels)
            n_labels += labels.numel()
        n_windows += seqs.size(0)
        if hasattr(pbar, "set_postfix"):
            elapsed = time.time() - t_start
            rate = n_windows / max(elapsed, 1e-6)
            pbar.set_postfix(windows=n_windows, labels=n_labels,
                             win_per_s=f"{rate:.2f}")
    elapsed = time.time() - t_start
    print(f"    {desc}: {n_windows} windows in {elapsed:.1f}s "
          f"({n_windows/max(elapsed,1e-6):.2f} win/s); {n_labels:,} labels")
    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe_on_cache(train_acts, train_labels, val_acts, val_labels,
                         device, epochs=500, lr=1e-3, batch_size=4096):
    probe = nn.Linear(32, 3).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(train_acts, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_acts_d = val_acts.to(device)
    val_labels_d = val_labels.to(device)

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0
        for acts, labels in train_loader:
            acts, labels = acts.to(device), labels.to(device)
            loss = loss_fn(probe(acts), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        probe.eval()
        with torch.no_grad():
            logits = probe(val_acts_d)
            probs = torch.softmax(logits, dim=1)  # Convert to probabilities
            
            # Filter: only positions where P(Don) + P(Acc) >= 0.05
            # (Which is equivalent to P(None) < 0.95)
            mask_filt = probs[:, NONE_CLASS] < 0.95
            
            if mask_filt.any():
                f_logits = logits[mask_filt]
                f_labels = val_labels_d[mask_filt]
                f_preds = f_logits.argmax(dim=1)
                
                filt_acc = (f_preds == f_labels).float().mean().item()
                
                # Per-class accuracy within the filtered set
                f_per_class = []
                for c, name in ((NONE_CLASS, "none"), (ACC_CLASS, "acc"), (DON_CLASS, "don")):
                    c_mask = f_labels == c
                    if c_mask.any():
                        acc_val = (f_preds[c_mask] == c).float().mean().item()
                        f_per_class.append(f"{name}:{acc_val:.3f}")
                
                # Binary preference: of the filtered sites, how often is the correct 
                # splice identity preferred over the wrong one?
                bin_metrics = []
                for c_true, c_alt, name in [(ACC_CLASS, DON_CLASS, "acc>don"), 
                                            (DON_CLASS, ACC_CLASS, "don>acc")]:
                    m = f_labels == c_true
                    if m.any():
                        pref = (f_logits[m, c_true] > f_logits[m, c_alt]).float().mean().item()
                        bin_metrics.append(f"{name}:{pref:.3f}")
                
                filter_str = (f"FILTERED (n={mask_filt.sum():,}, acc={filt_acc:.4f}): "
                              f"[{', '.join(f_per_class)}] [{', '.join(bin_metrics)}]")
            else:
                filter_str = "FILTERED: No sites passed the >5% confidence threshold."

        print(f"  epoch {epoch+1:3d}/{epochs}  train_loss={epoch_loss/max(n_batches,1):.4f}")
        print(f"    {filter_str}")
    return probe


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_loaders(fasta_path, gtf_path, none_subsample_ratio, overlap, batch_size,
                  max_genes_train=None, max_genes_val=None, max_genes_test=None):
    fasta = pyfastx.Fasta(fasta_path)

    print("Parsing GTF...")
    train_sites, train_genes = parse_gtf(gtf_path, chroms=TRAIN_CHROMS)
    val_sites, val_genes = parse_gtf(gtf_path, chroms=VAL_CHROMS)
    test_sites, test_genes = parse_gtf(gtf_path, chroms=TEST_CHROMS)

    print("Tiling windows...")
    train_recs = tile_windows(train_sites, train_genes, fasta, none_subsample_ratio, overlap,
                              max_genes=max_genes_train)
    val_recs = tile_windows(val_sites, val_genes, fasta, none_subsample_ratio, overlap,
                            max_genes=max_genes_val)
    test_recs = tile_windows(test_sites, test_genes, fasta, none_subsample_ratio, overlap,
                             max_genes=max_genes_test)
    print(f"  train: {len(train_recs)} windows")
    print(f"  val:   {len(val_recs)} windows")
    print(f"  test:  {len(test_recs)} windows")

    def make_loader(recs):
        return DataLoader(WindowDataset(recs, fasta_path),
                          batch_size=batch_size, shuffle=False,
                          collate_fn=collate_windows)

    return make_loader(train_recs), make_loader(val_recs), make_loader(test_recs)


def run_one_model(weights_path, train_loader, val_loader, test_loader,
                  device, cache_dir=None, probe_layer="skip"):
    pangolin = load_frozen_pangolin(weights_path, device)
    cache, is_cropped = attach_hook(pangolin, probe_layer=probe_layer)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache file is keyed by both model file and probe layer so different
        # layers don't overwrite each other's caches.
        cache_path = cache_dir / f"acts.{Path(weights_path).name}.{probe_layer}.pt"

    if cache_path and cache_path.exists():
        print(f"  loading cached activations from {cache_path}")
        blob = torch.load(cache_path)
        train_acts, train_labels = blob["train_acts"], blob["train_labels"]
        val_acts, val_labels = blob["val_acts"], blob["val_labels"]
        test_acts, test_labels = blob["test_acts"], blob["test_labels"]
    else:
        print(f"  extracting activations from {probe_layer} (cropped={is_cropped})")
        train_acts, train_labels = extract_activations(pangolin, cache, train_loader, device,
                                                       desc="  train", is_cropped=is_cropped)
        val_acts, val_labels = extract_activations(pangolin, cache, val_loader, device,
                                                   desc="  val", is_cropped=is_cropped)
        test_acts, test_labels = extract_activations(pangolin, cache, test_loader, device,
                                                     desc="  test", is_cropped=is_cropped)
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

    probe = train_probe_on_cache(train_acts, train_labels, val_acts, val_labels, device)

    # Final test set evaluation (only computed at end; never used for selection)
    probe.eval()
    with torch.no_grad():
        preds = probe(test_acts.to(device)).argmax(dim=1)
        labels_d = test_labels.to(device)
        acc = (preds == labels_d).float().mean().item()
        print(f"  TEST acc={acc:.4f}")
    return probe


def main(model_dir, fasta_path, gtf_path, out_dir,
         cache_dir=None, batch_size=4,
         none_subsample_ratio=10, overlap=DEFAULT_OVERLAP,
         max_genes_train=None, max_genes_val=None, max_genes_test=None,
         probe_layer="skip"):
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
                              device, cache_dir, probe_layer=probe_layer)
        torch.save(probe.state_dict(),
                   out_dir / f"probe.{mf.name}.{probe_layer}.pt")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _make_dummy_data(tmp_dir):
    """Synthetic FASTA + GTF with two overlapping + strand genes on chr2."""
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    fasta_path = tmp / "dummy.fa"
    gtf_path = tmp / "dummy.gtf"

    rng = random.Random(0)
    chroms = {"chr1": 80000, "chr2": 80000}     # chr1 = test, chr2 = train
    with open(fasta_path, "w") as fh:
        for chrom, length in chroms.items():
            seq = "".join(rng.choices("ACGT", k=length))
            fh.write(f">{chrom}\n")
            for i in range(0, length, 80):
                fh.write(seq[i:i+80] + "\n")

    # On chr2: gene A with exons 20000-20100, 22000-22100, 24000-24100
    #          gene B with exons 23000-23080 (overlaps gene A's window)
    # On chr1: gene C with one pair of exons for the test set
    with open(gtf_path, "w") as fh:
        for chrom, gene, exons in [
            ("chr2", "geneA", [(20000, 20100), (22000, 22100), (24000, 24100)]),
            ("chr2", "geneB", [(23000, 23080), (23300, 23380)]),
            ("chr1", "geneC", [(15000, 15100), (17000, 17100)]),
        ]:
            for s, e in exons:
                fh.write(f"{chrom}\ttest\texon\t{s}\t{e}\t.\t+\t.\tgene_id \"{gene}\";\n")

    return str(fasta_path), str(gtf_path)


if __name__ == "__main__":
    # Define your local paths
    model_dir = "/Users/ogw/Documents/GitHub/Pangolin/pangolin/models/" # <-- Update this to your 12 weights files
    fasta_path = "/Users/ogw/Downloads/hg38.fa"
    gtf_path = "/Users/ogw/Downloads/gencode.v49.basic.annotation.gtf" # <-- Update this to your GTF
    out_dir = "./bipangolin_probes"
    cache_dir = "./bipangolin_cache"

    # Apple Silicon GPU setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Starting biPangolin Extraction and Training using device: {device}...")
    
    main(
        model_dir=model_dir,
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        out_dir=out_dir,
        cache_dir=cache_dir,
        batch_size=32,  # Keep this low for the M2 RAM
        none_subsample_ratio=10,
        max_genes_train=None,    # set to None for full run
        max_genes_val=None,
        max_genes_test=None,
        probe_layer="resblock_15",   # try: skip, resblock_15, resblock_11, resblock_7, resblock_3
    )

# if __name__ == "__main__":
#     import tempfile
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     with tempfile.TemporaryDirectory() as tmp:
#         fasta_path, gtf_path = _make_dummy_data(tmp)
#         fasta = pyfastx.Fasta(fasta_path)

#         # Manually parse all chroms together to inspect
#         sites, genes = parse_gtf(gtf_path)
#         print(f"Parsed {sum(len(d) for d in sites.values())} splice sites "
#               f"across {len(sites)} chroms")

#         # Cross-gene labelling check: gene A's window around 22000-24000 should
#         # also pick up gene B's sites at 22999, 23079, 23299, 23379
#         print(f"chr2 splice sites: {sorted(sites['chr2'].keys())}")

#         # Build records using only the train chroms to demo the split
#         train_sites, train_genes = parse_gtf(gtf_path, chroms={"chr2"})
#         recs = tile_windows(train_sites, train_genes, fasta,
#                             none_subsample_ratio=5, overlap=2000)
#         print(f"chr2 windows: {len(recs)}")
#         for chrom, w_start, centers, labels in recs:
#             n_pos = (labels != NONE_CLASS).sum().item()
#             n_neg = (labels == NONE_CLASS).sum().item()
#             site_genomic = [(int(c) + w_start) for c, l in zip(centers.tolist(), labels.tolist()) if l != NONE_CLASS]
#             print(f"  {chrom}:{w_start}-{w_start+WINDOW_LEN}  "
#                   f"sites={sorted(site_genomic)}  none={n_neg}")

#         # Run end-to-end smoke test with random Pangolin weights
#         ds = WindowDataset(recs, fasta_path)
#         loader = DataLoader(ds, batch_size=2, collate_fn=collate_windows)

#         pangolin = Pangolin(L, W, AR).to(device).eval()
#         for p in pangolin.parameters():
#             p.requires_grad_(False)
#         cache = attach_skip_hook(pangolin)

#         acts, labels = extract_activations(pangolin, cache, loader, device)
#         print(f"acts: {tuple(acts.shape)}, labels dist: {torch.bincount(labels).tolist()}")

#         probe = train_probe_on_cache(acts, labels, acts, labels, device, epochs=10)
#         print("OK")



