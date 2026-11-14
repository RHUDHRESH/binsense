#!/usr/bin/env python3
"""Collect experiments/cnn/result_*.json (+ retime_fp32.json if present) into one markdown table on stdout."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
rt = json.loads((D / "retime_fp32.json").read_text()) if (D / "retime_fp32.json").exists() else {}
rows = []
for p in sorted(D.glob("result_*.json")):
    r = json.loads(p.read_text())
    f, q = r["fp32"], r.get("int8", {})
    ms = rt.get(f["pte"], {}).get("median_ms", f["laptop_ms_t4"])
    rows.append((r["model"], r["res"], r["params_m"], f["size_mb"], q.get("size_mb"), ms,
                 q.get("conv_cos_vs_eager_mean"), q.get("scope", "-"),
                 f["head_bal"]["acc"], f["head_bal"]["ci95"], q.get("head_bal_fp32fit", {}).get("acc")))
rows.sort(key=lambda t: t[5])
print("| model | res | params M | FP32 MB | INT8 MB | FP32 ms (t=4) | INT8 cos (converted) | INT8 scope | acc FP32 .pte [95% CI] | acc INT8-conv test (FP32-fit head) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m, res, pm, fm, qm, ms, cos, sc, a, ci, aq in rows:
    print(f"| {m} | {res} | {pm} | {fm} | {qm} | {ms} | {cos} | {sc.split(':')[0]} | {a:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {aq} |")
