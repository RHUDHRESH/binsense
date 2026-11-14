#!/usr/bin/env python3
"""Fit a tiny linear head on frozen image-tower features and add it to a decision pack.

The backbone stays the untouched pretrained model; only an [labels x 512] matrix is learned.
Embeddings are cached per folder, so re-fitting for a new label set takes seconds.

  --map maps organiser labels to public-data class folders:
        {"recyclable": ["trashnet_pool/cardboard", ...], "non_recyclable": ["trashnet_pool/trash"]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "edge"))
from binsense import BinSense  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def embed_folder(engine: BinSense, folder: Path, cache: Path, limit: int) -> np.ndarray:
    key = cache / (str(folder).replace("\\", "_").replace("/", "_") + f"__{engine.pte.stem}.npy")
    if key.exists():
        return np.load(key)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)[:limit]
    e = np.stack([engine.embed(engine.preprocess(p)) for p in files])
    cache.mkdir(parents=True, exist_ok=True)
    np.save(key, e)
    print(f"embedded {len(files)} from {folder}")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True, help="pack from build_labels.py (labels define the order)")
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=600, help="max images per source folder")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--gate-spec", type=Path, default=None, help="build a head-only pack for the gate model")
    ap.add_argument("--weight", type=float, default=0.7, help="head weight vs zero-shot text match")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if args.gate_spec:  # gate pack = labels from the main pack + the gate's own preprocessing, no text side
        g = json.loads(args.gate_spec.read_text())
        base = json.loads(str(np.load(args.pack, allow_pickle=False)["meta"]))
        gm = {"model": g["model"], "pretrained": g["pretrained"], "preprocess": g["preprocess"],
              "logit_scale": 1.0, "labels": base["labels"]}
        args.pack = args.out
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, meta=np.array(json.dumps(gm)), head_w=np.zeros((len(gm["labels"]), 1), np.float32),
                 head_b=np.zeros(len(gm["labels"]), np.float32))
    engine = BinSense(args.pte, args.pack)
    mapping = json.loads(args.map.read_text())
    X, y = [], []
    for i, lab in enumerate(engine.labels):
        for src in mapping[lab]:
            e = embed_folder(engine, args.data / src, Path("artifacts/emb_cache"), args.limit)
            X.append(e)
            y += [i] * len(e)
    X, y = np.concatenate(X), np.asarray(y)
    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
    cv = cross_val_score(clf, X, y, cv=5)
    clf.fit(X, y)
    W, b = clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)
    if W.shape[0] == 1:  # binary: expand to two rows
        W, b = np.concatenate([-W, W]) / 2, np.concatenate([-b, b]) / 2

    z = dict(np.load(args.pack, allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    meta.update(head_weight=args.weight, head_cv_accuracy=round(float(cv.mean()), 4), head_train_images=int(len(y)))
    z.update(meta=np.array(json.dumps(meta)), head_w=W, head_b=b)
    np.savez(args.out, **z)
    print(f"head: {len(y)} images, 5-fold CV accuracy {cv.mean() * 100:.1f}% (+/- {cv.std() * 100:.1f}) -> {args.out}")


if __name__ == "__main__":
    main()
