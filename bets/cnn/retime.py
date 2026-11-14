#!/usr/bin/env python3
"""Re-time every FP32 .pte in bets/cnn/artifacts sequentially (one process, threadpool 4, batch 1).

The main benchmark ran two model queues in parallel on a shared laptop; this pass removes our own
cross-contention (other agents' load remains). 3 rounds x 30 runs, median of round-medians.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import export_tower  # noqa: E402,F401  (flatc/env setup)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from executorch.extension.pybindings import portable_lib as pl  # noqa: E402

torch.set_num_threads(3)
A = Path(__file__).resolve().parent / "artifacts"
out = {}
pl._unsafe_reset_threadpool(4)
for p in sorted(A.glob("*_fp32.pte")):
    res = int(p.stem.split("_")[-2])
    x = torch.rand(1, 3, res, res, generator=torch.Generator().manual_seed(0))
    m = pl._load_for_executorch(str(p))
    for _ in range(5):
        m.forward((x,))
    meds = []
    for _ in range(3):
        t = []
        for _ in range(30):
            s = time.perf_counter()
            m.forward((x,))
            t.append((time.perf_counter() - s) * 1000)
        meds.append(float(np.median(t)))
    out[p.name] = {"median_ms": round(float(np.median(meds)), 2), "round_medians": [round(v, 2) for v in meds]}
    print(p.name, out[p.name], flush=True)
(A.parent / "retime_fp32.json").write_text(json.dumps(out, indent=1))
