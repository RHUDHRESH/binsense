#!/usr/bin/env python3
"""Paired comparison (exact McNemar) of each CNN head vs the MobileCLIP2-S0 reference, per test image.

Reference per-image correctness is read from results/results_laptop_fp32_head.md (read-only).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
ref = {}
for line in (ROOT / "results" / "results_laptop_fp32_head.md").read_text().splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) >= 5 and c[0].endswith(".jpg"):
        ref[(c[1], c[0])] = c[4] == "yes"
f = json.loads((D / "artifacts" / "files.json").read_text())
keys = [(Path(p).parent.name, Path(p).name) for p in f["test"]]
y = np.array([f["classes"].index(k[0]) for k in keys])
r = np.array([ref[k] for k in keys])
print(f"reference acc on same 240: {r.mean():.4f}  (n matched {len(keys)})")
for p in sorted((D / "artifacts").glob("*_testpred.npy")):
    m = np.load(p) == y
    b, c = int((m & ~r).sum()), int((~m & r).sum())
    pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(f"{p.stem:45s} acc {m.mean():.4f}  wins {b:3d} losses {c:3d}  McNemar exact p={pv:.3g}")
