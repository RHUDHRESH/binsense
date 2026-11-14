#!/usr/bin/env python3
"""Leakage probes for the TrashNet pool/test split (reads cached .npy features; no model needed).

1. exact duplicates (md5) between pool and test
2. for each test image, its top-1 cosine neighbour in the pool: how often is it a filename-adjacent
   sibling shot (same class, |index diff| <= 3)?  High => split is not group-aware, accuracy optimistic.
3. 1-NN accuracy vs LR head, and LR head accuracy when every pool image within +-3 filename index of
   any test image (same class) is removed from the fit set ("sibling-purged" fit).
Usage: python -X utf8 bets/cnn/leak_probe.py <prefix e.g. mobilenetv4_conv_small_224_fp32>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
A = Path(__file__).resolve().parent / "artifacts"
prefix = sys.argv[1]
f = json.loads((A / "files.json").read_text())
pool, test, classes = f["pool"], f["test"], f["classes"]
cls = lambda p: Path(p).parent.name  # noqa: E731
idx = lambda p: int(re.findall(r"(\d+)", Path(p).stem)[-1])  # noqa: E731
ytr = np.array([classes.index(cls(p)) for p in pool])
yte = np.array([classes.index(cls(p)) for p in test])

md5 = lambda p: hashlib.md5((ROOT / p).read_bytes()).hexdigest()  # noqa: E731
hp = {md5(p) for p in pool}
dup = sum(md5(p) in hp for p in test)

n = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)  # noqa: E731
Xtr, Xte = n(np.load(A / f"{prefix}_pool.npy")), n(np.load(A / f"{prefix}_test.npy"))
S = Xte @ Xtr.T
nn = S.argmax(1)
nn_acc = float((ytr[nn] == yte).mean())
sib = [cls(pool[j]) == cls(t) and abs(idx(pool[j]) - idx(t)) <= 3 for t, j in zip(test, nn)]

R = int(sys.argv[2]) if len(sys.argv) > 2 else 1
near = {(cls(t), idx(t) + k) for t in test for k in range(-R, R + 1)}
keep = np.array([(cls(p), idx(p)) not in near for p in pool])


def lr(mask):
    c = LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=0).fit(Xtr[mask], ytr[mask])
    return float((c.predict(Xte) == yte).mean())


out = {"prefix": prefix, "exact_dup_test_in_pool": dup, "nn1_acc": round(nn_acc, 4),
       "nn1_is_sibling_shot_frac": round(float(np.mean(sib)), 4), "nn1_cos_median": round(float(np.median(S.max(1))), 4),
       "lr_full_acc": round(lr(np.ones(len(pool), bool)), 4),
       "purge_radius": R, "lr_sibling_purged_acc": round(lr(keep), 4), "purged_pool_n": int(keep.sum()),
       # size-matched control: drop the same number of pool images at random (5 seeds)
       "lr_random_drop_same_n_acc": round(float(np.mean([lr(np.random.default_rng(s).permutation(keep))
                                                         for s in range(5)])), 4)}
print(json.dumps(out))
(A.parent / f"leak_{prefix}.json").write_text(json.dumps(out, indent=1))
