#!/usr/bin/env python3
"""Collect every results_*.json into one comparison table (SUMMARY.md) + energy estimate."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
rows = sorted((json.loads(p.read_text()) for p in out.glob("results_*.json")), key=lambda r: r["inference_ms"]["p50"])
# Published Pi 4B wall power under 4-core load (pidramble.com): 6.4 W. An estimate, not a measurement.
LOAD_W = 6.4
head = ["run", "size MB", "threads", "accuracy", "infer p50 ms", "infer p90 ms", "preproc p50 ms", "img/s", "peak RAM MB", "est. mJ/img", "temp", "throttled"]
lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
for r in rows:
    lines.append("| " + " | ".join(str(v) for v in [
        r["tag"], r["pte_size_mb"], r["threads"], f"{r['accuracy'] * 100:.1f}% ({r['correct']}/{r['scored']})",
        r["inference_ms"]["p50"], r["inference_ms"]["p90"], r["preprocess_ms"]["p50"], r["images_per_second"],
        r["peak_rss_mb"], round(LOAD_W * r["inference_ms"]["p50"]), r.get("temp_end") or "-", r.get("throttled") or "-"]) + " |")
th = out / "thermal.csv"
if th.exists():
    temps = [float(l.split("temp=")[1].split("'")[0]) for l in th.read_text().splitlines() if "temp=" in l]
    if temps:
        lines += ["", f"Thermal log: {len(temps)} s, min {min(temps):.1f} C, max {max(temps):.1f} C (throttling starts at 80 C)."]
(out / "SUMMARY.md").write_text("\n".join(lines))
print("\n".join(lines))
