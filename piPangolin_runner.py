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






































            psi_sums[tissue_idx]  += pangolin_out[PSI_CHANNEL_PER_TISSUE[tissue_idx]]
            count[tissue_idx] += 1
            probe_sum += probe_probs
            n_pairs += 1

        prob_stack = torch.stack([prob_sums[t] / count[t] for t in self.tissues_present])
        psi_stack  = torch.stack([psi_sums[t]  / count[t] for t in self.tissues_present])
        probe_avg  = probe_sum / n_pairs

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


def selftest(pangolin_model_dir, probe_dir, device="auto", ensemble=False):
    runner = BiPangolinRunner(pangolin_model_dir, probe_dir, device=device, ensemble=ensemble)
    result = runner.score_sequence(_CALIBRATION_SEQ)

    don_argmax = int(result.probe_donor.argmax())
    acc_argmax = int(result.probe_acceptor.argmax())
    print(f"\nCalibration sequence ({len(_CALIBRATION_SEQ)} bp)")
    print(f"  Annotated donor:    69   probe argmax: {don_argmax}  "
          f"(P={result.probe_donor[don_argmax]:.3f}, P@69={result.probe_donor[69]:.3f})")
    print(f"  Annotated acceptor: 163  probe argmax: {acc_argmax}  "
          f"(P={result.probe_acceptor[acc_argmax]:.3f}, P@163={result.probe_acceptor[163]:.3f})")


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
