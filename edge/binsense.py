"""BinSense: an offline "System One" decision engine for images.

One ExecuTorch image tower (.pte) turns a photo into an embedding. Typed questions
are answered by comparing that embedding with pre-computed text embeddings:

  choice(img)        -> one label + probability per label        (which bin?)
  noul(img, name)    -> probability of "yes"                     (is it recyclable?)
  score(img, name)   -> expected level on an ordered scale       (how contaminated?)

Runs on the Raspberry Pi with only executorch + torch + numpy + pillow.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class BinSense:
    def __init__(self, pte: str | Path, pack: str | Path, threads: int | None = None):
        from executorch.extension.pybindings import portable_lib as pl

        if threads:
            pl._unsafe_reset_threadpool(int(threads))
        self.threads = pl._threadpool_get_thread_count()
        t0 = time.perf_counter()
        self.module = pl._load_for_executorch(str(pte))
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.pte = Path(pte)

        z = np.load(pack, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.res = int(self.meta["preprocess"]["size"])
        self.mean = np.asarray(self.meta["preprocess"]["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.asarray(self.meta["preprocess"]["std"], np.float32).reshape(3, 1, 1)
        self.scale = float(self.meta["logit_scale"])
        self.labels = list(self.meta["labels"])
        self.label_emb = z["label_emb"].astype(np.float32) if "label_emb" in z.files else None  # [L, D]
        self.noul = {k: z[f"noul__{k}"].astype(np.float32) for k in self.meta.get("noul", {})}
        self.scales = {k: z[f"score__{k}"].astype(np.float32) for k in self.meta.get("score", {})}
        # optional few-shot prototypes (built from support photos, never from test photos)
        self.proto = z["proto"].astype(np.float32) if "proto" in z.files else None
        self.proto_weight = float(self.meta.get("proto_weight", 0.5))
        # optional linear head fitted on frozen features (public data mapped to the labels)
        self.head = (z["head_w"].astype(np.float32), z["head_b"].astype(np.float32)) if "head_w" in z.files else None
        self.head_weight = float(self.meta.get("head_weight", 0.7))
        # label groups for multi-select output, e.g. {"recyclable": ["paper", "glass"], "dry_waste": [...]}
        self.groups = dict(self.meta.get("groups", {}))

    # ---- image -> tensor -------------------------------------------------
    def preprocess(self, path: str | Path) -> np.ndarray:
        from PIL import Image

        im = Image.open(path)
        im.draft("RGB", (self.res * 2, self.res * 2))  # JPEG: decode at reduced size in the DCT domain
        im = im.convert("RGB")
        w, h = im.size
        s = self.res / min(w, h)
        im = im.resize((max(self.res, round(w * s)), max(self.res, round(h * s))), Image.BILINEAR)
        w, h = im.size
        l, t = (w - self.res) // 2, (h - self.res) // 2
        im = im.crop((l, t, l + self.res, t + self.res))
        x = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None]

    def embed(self, x: np.ndarray) -> np.ndarray:
        import torch

        e = self.module.forward((torch.from_numpy(np.ascontiguousarray(x)),))[0].numpy()[0]
        return e / (np.linalg.norm(e) + 1e-12)

    # ---- typed answers ---------------------------------------------------
    def choice_from(self, e: np.ndarray) -> dict:
        if self.label_emb is None:  # head-only model (the fast gate has no text side)
            p = softmax(self.head[0] @ e + self.head[1])
        else:
            logits = self.scale * (self.label_emb @ e)
            if self.proto is not None:
                w = self.proto_weight
                logits = (1 - w) * logits + w * self.scale * (self.proto @ e)
            p = softmax(logits)
            if self.head is not None:  # blend: text match keeps unseen-looking items sane
                w = self.head_weight
                p = (1 - w) * p + w * softmax(self.head[0] @ e + self.head[1])
        i = int(p.argmax())
        return {"answer": self.labels[i], "confidence": float(p[i]),
                "probabilities": {k: float(v) for k, v in zip(self.labels, p)}}

    def multi_from(self, c: dict, nouls: dict, label_min: float = 0.25, tag_min: float = 0.5) -> dict:
        """Multi-select: every label that is plausibly present, plus every tag that applies.

        A carton that is half paper, half cardboard returns both; group tags (recyclable, dry_waste, ...)
        are the summed probability of their member labels; yes/no questions add their own tags.
        """
        probs = c["probabilities"]
        labels = [k for k, v in sorted(probs.items(), key=lambda kv: -kv[1]) if v >= label_min] or [c["answer"]]
        tags = {g: float(sum(probs.get(m, 0.0) for m in members)) for g, members in self.groups.items()}
        tags.update(nouls)
        return {"labels": labels, "tags": [k for k, v in tags.items() if v >= tag_min],
                "tag_probabilities": {k: round(v, 3) for k, v in tags.items()}}

    def noul_from(self, e: np.ndarray, name: str) -> float:
        return float(softmax(self.scale * (self.noul[name] @ e))[1])  # rows: [no, yes]

    def score_from(self, e: np.ndarray, name: str) -> dict:
        p = softmax(self.scale * (self.scales[name] @ e))
        return {"level": float((p * np.arange(1, len(p) + 1)).sum()), "probabilities": p.tolist()}

    def decide(self, path: str | Path) -> dict:
        t0 = time.perf_counter()
        x = self.preprocess(path)
        t1 = time.perf_counter()
        e = self.embed(x)
        t2 = time.perf_counter()
        c = self.choice_from(e)
        nouls = {k: self.noul_from(e, k) for k in self.noul}
        out = {"image": str(path), "choice": c, "noul": nouls, "multi": self.multi_from(c, nouls),
               "score": {k: self.score_from(e, k) for k in self.scales}}
        t3 = time.perf_counter()
        out["ms"] = {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "decision": (t3 - t2) * 1000}
        return out


class Cascade:
    """Two-stage decision: a tiny INT8 gate answers when it is sure, otherwise the full model decides.

    Mirrors fast/slow thinking: cheap reflex for obvious items, careful look for ambiguous ones.
    """

    def __init__(self, gate: BinSense, full: BinSense, threshold: float = 0.9):
        self.gate, self.full, self.threshold = gate, full, threshold
        self.labels, self.meta, self.pte, self.res = full.labels, full.meta, full.pte, full.res
        self.threads, self.load_ms = full.threads, gate.load_ms + full.load_ms
        self.fast = self.slow = 0

    def decide(self, path) -> dict:
        t0 = time.perf_counter()
        x = self.gate.preprocess(path)
        t1 = time.perf_counter()
        c = self.gate.choice_from(self.gate.embed(x))
        t2 = time.perf_counter()
        if c["confidence"] >= self.threshold:
            self.fast += 1
            return {"image": str(path), "choice": c, "noul": {}, "multi": self.gate.multi_from(c, {}), "score": {}, "path": "fast",
                    "ms": {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "decision": 0.0}}
        self.slow += 1
        d = self.full.decide(path)
        d["path"] = "careful"
        d["ms"]["preprocess"] += (t1 - t0) * 1000
        d["ms"]["inference"] += (t2 - t1) * 1000
        return d
