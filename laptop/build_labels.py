#!/usr/bin/env python3
"""Turn a label file into the small 'decision pack' the Pi uses (no text model on the Pi).

labels.json:
{
  "labels": {"recyclable": ["a plastic bottle", "a cardboard box", ...], "organic": [...]},
  "noul":   {"recyclable": {"yes": ["..."], "no": ["..."]}},
  "score":  {"contamination": [["a clean ...", ...], ..., ["a filthy ...", ...]]}
}
Optionally --support <folder/label/*.jpg> adds few-shot prototypes per label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "a photo of {} thrown away as waste.",
             "{} in a bin.", "a picture of discarded {}."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True, help="*_spec.json written by export_tower.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--support", type=Path, default=None)
    ap.add_argument("--proto-weight", type=float, default=0.5)
    args = ap.parse_args()

    import open_clip

    spec = json.loads(args.spec.read_text())
    clip, _, preprocess = open_clip.create_model_and_transforms(spec["model"], pretrained=spec["pretrained"])
    clip.eval()
    tok = open_clip.get_tokenizer(spec["model"])
    cfg = json.loads(args.labels.read_text())

    @torch.inference_mode()
    def text(phrases: list[str]) -> np.ndarray:
        prompts = [t.format(p) for p in phrases for t in TEMPLATES]
        f = clip.encode_text(tok(prompts))
        f = f / f.norm(dim=-1, keepdim=True)
        m = f.mean(0)
        return (m / m.norm()).numpy().astype(np.float32)

    arrays = {"label_emb": np.stack([text(v) for v in cfg["labels"].values()])}
    for k, v in cfg.get("noul", {}).items():
        arrays[f"noul__{k}"] = np.stack([text(v["no"]), text(v["yes"])])
    for k, levels in cfg.get("score", {}).items():
        arrays[f"score__{k}"] = np.stack([text(l) for l in levels])

    meta = {"model": spec["model"], "pretrained": spec["pretrained"], "preprocess": spec["preprocess"],
            "logit_scale": float(clip.logit_scale.exp()), "labels": list(cfg["labels"]),
            "noul": {k: True for k in cfg.get("noul", {})}, "score": {k: len(v) for k, v in cfg.get("score", {}).items()}}

    if args.support:
        from PIL import Image

        protos = []
        with torch.inference_mode():
            for lab in cfg["labels"]:
                files = [p for p in (args.support / lab).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                f = clip.encode_image(torch.stack([preprocess(Image.open(p).convert("RGB")) for p in files]))
                f = (f / f.norm(dim=-1, keepdim=True)).mean(0)
                protos.append((f / f.norm()).numpy().astype(np.float32))
                print(f"prototype {lab}: {len(files)} photos")
        arrays["proto"] = np.stack(protos)
        meta["proto_weight"] = args.proto_weight

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, meta=np.array(json.dumps(meta)), **arrays)
    print("wrote", args.out, {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
