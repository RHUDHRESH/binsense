#!/usr/bin/env python3
"""Classify a folder of labelled images and write the hackathon deliverables.

Images:  <images>/<label>/<file>   (or --csv file,label)
Outputs: results.md, results.csv, stats.json in --out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from binsense import Cascade, BinSense  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def vc(*a: str) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:  # noqa: BLE001  (not a Pi)
        return None


def peak_rss_mb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6
        except Exception:  # noqa: BLE001
            return None


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1, help="timing repeats per image")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--gate-pte", type=Path, default=None, help="fast gate model (enables two-stage mode)")
    ap.add_argument("--gate-pack", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    if args.csv:
        items = [(args.images / r[0], r[1]) for r in csv.reader(args.csv.open()) if r and r[0] != "file"]
    else:
        items = [(p, p.parent.name) for p in sorted(args.images.rglob("*")) if p.suffix.lower() in EXT]
    if not items:
        sys.exit("no images found")

    temp0 = vc("measure_temp")
    engine = BinSense(args.pte, args.pack, threads=args.threads or None)
    if args.gate_pte:
        engine = Cascade(BinSense(args.gate_pte, args.gate_pack), engine, args.threshold)
    for _ in range(args.warmup):
        engine.decide(items[0][0])
        if args.gate_pte:
            engine.full.decide(items[0][0])
    if args.gate_pte:
        engine.fast = engine.slow = 0

    rows, pre, inf = [], [], []
    wall0 = time.perf_counter()
    for path, truth in items:
        for _ in range(args.repeat):
            d = engine.decide(path)
            pre.append(d["ms"]["preprocess"])
            inf.append(d["ms"]["inference"])
        c = d["choice"]
        rows.append({"image": path.name, "label": truth, "predicted": c["answer"],
                     "confidence": round(c["confidence"], 3), "correct": c["answer"] == truth,
                     "multi_select": " + ".join(d["multi"]["labels"] + d["multi"]["tags"]),
                     "inference_ms": round(d["ms"]["inference"], 1), **({"path": d["path"]} if "path" in d else {}),
                     **{f"yes_{k}": round(v, 3) for k, v in d["noul"].items()},
                     **{f"score_{k}": round(v["level"], 2) for k, v in d["score"].items()}})
    wall = time.perf_counter() - wall0

    known = [r for r in rows if r["label"] in engine.labels]
    correct = sum(r["correct"] for r in known)
    stats = {
        "tag": args.tag, "model": engine.meta["model"], "pretrained": engine.meta["pretrained"], "pte": engine.pte.name,
        "pte_size_mb": round(engine.pte.stat().st_size / 1e6, 2), "resolution": engine.res, "threads": engine.threads,
        "images": len(rows), "scored": len(known), "correct": correct,
        "accuracy": round(correct / max(1, len(known)), 4),
        "load_ms": round(engine.load_ms, 1),
        "inference_ms": {"p50": round(pct(inf, .5), 1), "p90": round(pct(inf, .9), 1), "min": round(min(inf), 1), "max": round(max(inf), 1)},
        "preprocess_ms": {"p50": round(pct(pre, .5), 1), "p90": round(pct(pre, .9), 1)},
        "images_per_second": round(len(inf) / wall, 2),
        "peak_rss_mb": round(peak_rss_mb() or 0, 1),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "temp_start": temp0, "temp_end": vc("measure_temp"), "clock_arm": vc("measure_clock", "arm"),
        "throttled": vc("get_throttled"),
    }
    if args.gate_pte:
        n = engine.fast + engine.slow
        stats["two_stage"] = {"gate": args.gate_pte.name, "threshold": args.threshold,
                              "fast_path_share": round(engine.fast / max(1, n), 3),
                              "gate_size_mb": round(args.gate_pte.stat().st_size / 1e6, 2)}
        fast_rows = [r for r in known if r.get("path") == "fast"]
        stats["two_stage"]["fast_path_accuracy"] = round(sum(r["correct"] for r in fast_rows) / max(1, len(fast_rows)), 4)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"results{('_' + args.tag) if args.tag else ''}"
    with (args.out / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (args.out / f"{stem}.json").write_text(json.dumps(stats, indent=1))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("yes" if r.get(c) is True else "NO" if r.get(c) is False else str(r.get(c, "")) for c in cols) + " |" for r in rows]
    md += ["", f"**Correct: {correct} / {len(known)}  |  Accuracy: {stats['accuracy'] * 100:.1f}%**", "",
           "```json", json.dumps(stats, indent=1), "```"]
    (args.out / f"{stem}.md").write_text("\n".join(md))

    if len(rows) <= 40:
        print("\n".join(md[:len(rows) + 2]))
    print(f"\nCorrect {correct}/{len(known)}  accuracy {stats['accuracy'] * 100:.1f}%")
    print(json.dumps({k: stats[k] for k in ("pte", "pte_size_mb", "threads", "load_ms", "inference_ms", "preprocess_ms",
                                            "images_per_second", "peak_rss_mb", "temp_end", "throttled", "two_stage") if k in stats}, indent=1))


if __name__ == "__main__":
    main()
