"""Leakage audit: test (data/trashnet) vs training pool (data/trashnet_pool).

Checks: identical file names, identical SHA-256 bytes, and perceptual near-duplicates (64-bit dHash).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEST, POOL = ROOT / "data/trashnet", ROOT / "data/trashnet_pool"


def files(d):
    return sorted(p for p in d.rglob("*.jpg"))


def dhash(p):
    g = np.asarray(Image.open(p).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


test, pool = files(TEST), files(POOL)
names_t = {(p.parent.name, p.name) for p in test}
names_p = {(p.parent.name, p.name) for p in pool}
bare_t, bare_p = {p.name for p in test}, {p.name for p in pool}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sha_t, sha_p = {sha(p): p for p in test}, {sha(p): p for p in pool}
dh_t = np.stack([dhash(p) for p in test]); dh_p = np.stack([dhash(p) for p in pool])
bits = np.unpackbits(dh_t[:, None, :] ^ dh_p[None, :, :], axis=2).sum(2)  # (240, 2287) hamming
mins = bits.min(1)
near = [(str(test[i].relative_to(ROOT)), str(pool[int(bits[i].argmin())].relative_to(ROOT)), int(mins[i]))
        for i in np.argsort(mins)[:10]]
rep = {
    "n_test": len(test), "n_pool": len(pool),
    "same_class_and_name": len(names_t & names_p), "same_bare_name": len(bare_t & bare_p),
    "same_sha256": len(set(sha_t) & set(sha_p)),
    "dup_sha_within_pool": len(pool) - len(sha_p), "dup_sha_within_test": len(test) - len(sha_t),
    "dhash_min_hamming_hist": {f"<={k}": int((mins <= k).sum()) for k in (0, 2, 4, 6, 8, 10)},
    "closest_pairs": near,
}
(Path(__file__).parent / "leak_check.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
