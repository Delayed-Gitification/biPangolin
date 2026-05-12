"""
spliceai_worker.py — persistent SpliceAI scoring subprocess.

Protocol (line-based JSON over stdin/stdout):

  parent -> worker:   {"seq_path": "/tmp/seq.txt", "out_path": "/tmp/out.npz", "context": 10000}
  worker -> parent:   {"status": "ok"}             OR   {"status": "error", "msg": "..."}

The worker emits "READY" on its first stdout line once models are loaded.
Sequence is read from a file (not stdin) to avoid encoding / size issues.
Scores are written to an .npz with arrays `acceptor`, `donor`.

Run directly:
    python spliceai_worker.py
then write JSON requests to its stdin, one per line.
"""
import json
import os
import sys

# Quiet TF startup banner
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import spliceai
from tensorflow.keras.models import load_model
from spliceai.utils import one_hot_encode


def main():
    models_dir = os.path.join(os.path.dirname(spliceai.__file__), "models")
    sys.stderr.write(f"[spliceai_worker] loading 5 models from {models_dir}\n")
    sys.stderr.flush()
    models = [load_model(os.path.join(models_dir, f"spliceai{i}.h5"))
              for i in range(1, 6)]
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            with open(req["seq_path"]) as f:
                seq = f.read().strip()
            ctx = int(req.get("context", 10000))
            padded = "N" * (ctx // 2) + seq + "N" * (ctx // 2)
            x = one_hot_encode(padded)[None, :]
            y = np.mean([m.predict(x, verbose=0) for m in models], axis=0)
            np.savez(req["out_path"], acceptor=y[0, :, 1], donor=y[0, :, 2])
            sys.stdout.write(json.dumps({"status": "ok"}) + "\n")
        except Exception as e:
            sys.stdout.write(json.dumps({"status": "error", "msg": repr(e)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
