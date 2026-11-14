#!/usr/bin/env python3
"""Trace ONE photo through BinSense, step by step: input, preprocessing, each model, probabilities, decision.

  python trace.py                      # first sample photo
  python trace.py images/plastic/plastic1.jpg
  python trace.py some_photo.jpg --careful-only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binsense import BinSense, softmax  # noqa: E402

C = sys.stdout.isatty()
B, D, G, Y, R, X = [(c if C else "") for c in ("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")]


def head(title):
    print(f"\n{B}{title}{X}\n" + "-" * 72)


def bars(labels, p, mark=None):
    for i in np.argsort(-p):
        bar = "#" * int(round(p[i] * 40))
        tag = f"  {G}<- top{X}" if i == int(p.argmax()) else ""
        print(f"    {labels[i]:<11}{p[i] * 100:6.2f}%  {bar}{tag}")


def run_model(name, model, path, role):
    head(name)
    print(f"  role          {role}")
    print(f"  file          {model.pte.name}  ({model.pte.stat().st_size / 1e6:.1f} MB)")
    print(f"  model         {model.meta['model']}")
    from PIL import Image
    w, h = Image.open(path).size
    t0 = time.perf_counter(); x = model.preprocess(path); t1 = time.perf_counter()
    print(f"\n  {B}preprocess{X}    {w}x{h} photo -> resize shortest side to {model.res}, centre crop {model.res}x{model.res}")
    print(f"                scale: (pixel/255 - {[round(float(v), 3) for v in model.mean.ravel()]}) / {[round(float(v), 3) for v in model.std.ravel()]}")
    print(f"  tensor in     shape {tuple(x.shape)}, {x.size:,} numbers, range {x.min():.2f} to {x.max():.2f}   [{(t1 - t0) * 1000:.1f} ms]")
    t2 = time.perf_counter(); e = model.embed(x); t3 = time.perf_counter()
    print(f"\n  {B}ExecuTorch{X}    forward() through the .pte with the XNNPACK delegate, {model.threads} threads   [{(t3 - t2) * 1000:.1f} ms]")
    print(f"  embedding out {len(e)} numbers, scaled to length 1. First 8: {[round(float(v), 3) for v in e[:8]]}")
    W, b = model.head
    logits = W @ e + b
    p = softmax(logits)
    print(f"\n  {B}decision table{X} {W.shape[0]} labels x {W.shape[1]} weights. Score per label = weights . embedding + offset")
    print(f"  raw scores    " + "  ".join(f"{l}={v:+.2f}" for l, v in zip(model.labels, logits)))
    print(f"  probabilities (softmax of the scores):")
    bars(model.labels, p)
    return p, (t3 - t2) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--pte", default="models/own_mobileclip2_s0_256_fp32.pte")
    ap.add_argument("--pack", default="packs/own_mobileclip2_s0_256.npz")
    ap.add_argument("--gate-pte", default="models/stu_mobilenetv2_100_96_fp32.pte")
    ap.add_argument("--gate-pack", default="packs/stu_mobilenetv2_100_96.npz")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--review", type=float, default=0.7)
    ap.add_argument("--careful-only", action="store_true")
    ap.add_argument("--both", action="store_true", help="run the careful model even when the fast one is sure")
    args = ap.parse_args()
    base = Path(__file__).parent
    path = Path(args.image) if args.image else sorted((base / "images").rglob("*.jpg"))[0]
    if not path.exists():
        sys.exit(f"no such file: {path}")

    head("STEP 0  INPUT")
    from PIL import Image
    im = Image.open(path)
    print(f"  photo         {path}")
    print(f"  size          {im.size[0]}x{im.size[1]} pixels, {path.stat().st_size / 1024:.0f} KB, {im.mode}")
    truth = path.parent.name
    full = BinSense(base / args.pte, base / args.pack)
    known = truth in full.labels
    print(f"  true label    {truth if known else 'unknown (folder name is not a label)'}")

    final, used, total_ms = None, None, 0.0
    if not args.careful_only and (base / args.gate_pte).exists():
        gate = BinSense(base / args.gate_pte, base / args.gate_pack)
        p, ms = run_model("STEP 1  FAST MODEL", gate, path, "quick first answer"); total_ms += ms
        top = float(p.max())
        head("STEP 2  IS THE FAST MODEL SURE ENOUGH?")
        ok = top >= args.threshold
        print(f"  top probability {top * 100:.1f}%   needs at least {args.threshold * 100:.0f}%   ->  {(G + 'YES, accept its answer') if ok else (Y + 'NO, ask the careful model')}{X}")
        if ok:
            final, used = p, "fast"
        if not ok or args.both:
            p2, ms = run_model("STEP 3  CAREFUL MODEL", full, path, "slower, more accurate second opinion"); total_ms += ms
            if not ok:
                final, used = p2, "careful"
    else:
        final, ms = run_model("STEP 1  CAREFUL MODEL", full, path, "single model"); total_ms += ms; used = "careful"

    head("FINAL  OUTPUT")
    i = int(final.argmax()); conf = float(final[i]); label = full.labels[i]
    c = {"answer": label, "confidence": conf, "probabilities": dict(zip(full.labels, map(float, final)))}
    multi = full.multi_from(c, {})
    if conf < args.review:
        print(f"  decision      {Y}REVIEW{X}  (best guess {label}, {conf * 100:.1f}% is below the {args.review * 100:.0f}% floor)")
    else:
        print(f"  decision      {B}{label}{X}   confidence {conf * 100:.1f}%   decided by the {used} model")
    print(f"  multi-select  " + " + ".join(multi["labels"] + multi["tags"]))
    print(f"  tag scores    " + "  ".join(f"{k}={v:.2f}" for k, v in multi["tag_probabilities"].items()))
    if known:
        print(f"  check         true label is {truth}  ->  {(G + 'CORRECT') if truth == label else (R + 'WRONG')}{X}")
    print(f"  model time    {total_ms:.1f} ms total inference")


if __name__ == "__main__":
    main()
