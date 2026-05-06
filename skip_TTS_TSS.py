(bipangolin) [wilkino@gl411 biPangolin]$ python
Python 3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:45:40) [GCC 14.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
[16]+  Stopped                 python
(bipangolin) [wilkino@gl411 biPangolin]$ python runner.py /camp/home/wilkino/home/POSTDOC/software/biPangolin/Pangolin/pangolin/models/ ./bipangolin_probes
python: can't open file '/nemo/lab/ulej/home/users/wilkino/POSTDOC/software/biPangolin/runner.py': [Errno 2] No such file or directory
(bipangolin) [wilkino@gl411 biPangolin]$ python biPangolin_runner.py /camp/home/wilkino/home/POSTDOC/software/biPangolin/Pangolin/pangolin/models/ ./bipangolin_probes
biPangolin: 1 model+probe pairs ready on cuda

Calibration sequence (217 bp)
  Annotated donor:    69   probe argmax: 69  (P=1.000, P@69=1.000)
  Annotated acceptor: 163  probe argmax: 163  (P=1.000, P@163=1.000)
(bipangolin) [wilkino@gl411 biPangolin]$ zless biPangolin_runner.py 

[17]+  Stopped                 zless biPangolin_runner.py
(bipangolin) [wilkino@gl411 biPangolin]$ zless train_
train_5_layers.py    train_probes_old.py  train_probes.py      
(bipangolin) [wilkino@gl411 biPangolin]$ zless 
another.py             bipangolin.py          direct_probe_train.py  fml.py                 skip_TTS_TSS.py        train_probes.py
bipangolin_cache/      biPangolin_runner.py   direct_probing.py      logs/                  train_5_layers.py      
bipangolin_probes/     data/                  environment.yml        Pangolin/              train_probes_old.py    
(bipangolin) [wilkino@gl411 biPangolin]$ zless skip_TTS_TSS.py 






























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
        none_subsample_ratio=10,
        max_genes_train=None,
        max_genes_val=None,
        max_genes_test=None,
        probe_layer=PROBE_LAYERS,
        include_sequence=True,   # also concatenate raw 11-nt one-hot window
        kernel_size=1,
        hidden_dim=64,
    )
