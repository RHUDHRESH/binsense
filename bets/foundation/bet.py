#!/usr/bin/env python3
"""Foundation-backbone bet: export -> XNNPACK .pte -> features through the .pte -> heads.

  bet.py run <name> [<name> ...]   export (if needed) + latency + feature extraction per model
  bet.py eval                      zero-shot / 20-shot prototypes / logistic head on cached features

Everything is written under bets/foundation/. Test folder data/trashnet is never used for fitting.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import export_tower as et  # noqa: E402  (configures flatc for ExecuTorch on Windows)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
SEED = 0
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXT = {".jpg", ".jpeg", ".png", ".webp"}
PTE, FEATS, RES = HERE / "pte", HERE / "feats", HERE / "results"
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

CANDS = {
    "s0_ref": dict(kind="existing", pte=ROOT / "artifacts" / "mobileclip2-s0_256_fp32.pte",
                   model="MobileCLIP2-S0", pretrained="dfndr2b"),
    "dinov2_s14": dict(kind="timm", model="vit_small_patch14_dinov2.lvd142m"),
    "dinov2_s14_reg4": dict(kind="timm", model="vit_small_patch14_reg4_dinov2.lvd142m"),
    "mobileclip_s1": dict(kind="openclip", model="MobileCLIP-S1", pretrained="datacompdr"),
    "mobileclip2_s2": dict(kind="openclip", model="MobileCLIP2-S2", pretrained="dfndr2b"),
    "vitb32": dict(kind="openclip", model="ViT-B-32", pretrained="laion2b_s34b_b79k"),
}


POOL_CAP = 100  # CPU is saturated by other agents: seeded 100/class pool subset (trash has 97 -> all)


def files_of(split: str, cap: int | None = None):
    out = []
    rng = np.random.default_rng(SEED)
    for i, c in enumerate(CLASSES):
        fs = [(p, i) for p in sorted((ROOT / "data" / split / c).iterdir()) if p.suffix.lower() in EXT]
        if cap and len(fs) > cap:
            fs = [fs[j] for j in sorted(rng.choice(len(fs), cap, replace=False))]
        out += fs
    return out


def s0_cached_pool():
    """Full-pool S0 features already computed through the same .pte + same preprocess by src/fit_head.py (read-only)."""
    cache = ROOT / "artifacts" / "emb_cache"
    return np.concatenate([np.load(cache / f"data_trashnet_pool_{c}__mobileclip2-s0_256_fp32.npy") for c in CLASSES])


def preprocess(path: Path, res: int, mean, std) -> np.ndarray:
    """Identical to edge/jev_edge.py EdgeJev.preprocess (what the Pi runs): shortest-side bilinear + centre crop."""
    from PIL import Image

    im = Image.open(path)
    im.draft("RGB", (res * 2, res * 2))
    im = im.convert("RGB")
    w, h = im.size
    s = res / min(w, h)
    im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - res) // 2, (h - res) // 2
    im = im.crop((l, t, l + res, t + res))
    x = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
    m = np.asarray(mean, np.float32).reshape(3, 1, 1)
    sd = np.asarray(std, np.float32).reshape(3, 1, 1)
    # MUST be C-contiguous: the ExecuTorch pybinding ignores numpy strides (non-contiguous input -> scrambled pixels,
    # caught by cosine-vs-eager 0.12-0.36 on real images while torch.rand gave 1.0). jev_edge does the same in embed().
    return np.ascontiguousarray(((x - m) / sd)[None])


def build(name: str):
    """Return (eager module, res, mean, std, text_fn or None, params_m)."""
    c = CANDS[name]
    text_fn = None
    if c["kind"] == "timm":
        import timm

        m = timm.create_model(c["model"], pretrained=True, img_size=224, num_classes=0).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        res, mean, std = 224, cfg["mean"], cfg["std"]
        tower = m
    else:
        clip, _, cfg = et.load(c["model"], c["pretrained"])
        tower = et.ImageTower(clip).eval()
        res, mean, std = cfg["size"][-1] if isinstance(cfg["size"], (list, tuple)) else cfg["size"], cfg["mean"], cfg["std"]
        import open_clip

        tok = open_clip.get_tokenizer(c["model"])

        @torch.inference_mode()
        def text_fn(prompts):
            f = clip.encode_text(tok(prompts)).float()
            return (f / f.norm(dim=-1, keepdim=True)).numpy()
    params = round(sum(p.numel() for p in tower.parameters()) / 1e6, 2)
    return tower, int(res), tuple(mean), tuple(std), text_fn, params


def latency(pte: Path, x: torch.Tensor, runs: int = 20):
    from executorch.extension.pybindings import portable_lib as pl

    m = pl._load_for_executorch(str(pte))
    for _ in range(3):
        m.forward((x,))
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        m.forward((x,))
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t)), m


def run(name: str):
    from executorch.extension.pybindings import portable_lib as pl

    pl._unsafe_reset_threadpool(4)
    rep_path = RES / f"{name}_export.json"
    rep = {"name": name, **{k: str(v) for k, v in CANDS[name].items()}, "threadpool": pl._threadpool_get_thread_count()}
    t0 = time.time()
    tower, res, mean, std, text_fn, params = build(name)
    rep.update(res=res, mean=mean, std=std, params_m=params)
    test = files_of("trashnet")
    x_real = torch.from_numpy(preprocess(test[0][0], res, mean, std))
    with torch.inference_mode():
        ref = tower(x_real)
        s = time.perf_counter()
        for _ in range(5):
            tower(x_real)
        rep["eager_ms"] = round((time.perf_counter() - s) * 200, 1)
    rep["feat_dim"] = int(ref.shape[-1])

    if CANDS[name]["kind"] == "existing":
        pte = CANDS[name]["pte"]
        rep["pte"] = str(pte)
        rep["size_mb"] = round(pte.stat().st_size / 1e6, 2)
    else:
        pte = PTE / f"{name}_{res}_fp32.pte"
        t = time.time()
        exported = torch.export.export(tower, (x_real,))
        info = et.lower(exported, pte)
        rep.update(info)
        rep["pte"] = str(pte)
        rep["export_s"] = round(time.time() - t, 1)
    ms, mod = latency(pte, x_real)
    out = mod.forward((x_real,))[0]
    rep["pte_ms_median20"] = round(ms, 1)
    rep["cosine_vs_eager_real"] = round(float(torch.nn.functional.cosine_similarity(out.reshape(1, -1), ref.reshape(1, -1)).item()), 6)
    print(json.dumps(rep, indent=1), flush=True)
    rep_path.write_text(json.dumps(rep, indent=1))

    if text_fn is not None:
        np.save(FEATS / f"{name}_text.npy", text_fn([f"a photo of {c}, a type of waste" for c in CLASSES]))

    # features through the .pte, batch 1, raw (not normalised)
    per = []
    for split in ("trashnet", "trashnet_pool"):
        fs = files_of(split, POOL_CAP if split == "trashnet_pool" else None)
        tag = "test" if split == "trashnet" else "pool"
        if name == "s0_ref" and tag == "pool":  # reuse the cached full-pool .pte features, subset to the capped list
            full = files_of(split)
            idx = {p: k for k, (p, _) in enumerate(full)}
            allF = s0_cached_pool()
            assert len(allF) == len(full), (len(allF), len(full))
            np.save(FEATS / "s0_ref_poolfull.npy", allF)
            np.save(FEATS / "labels_poolfull.npy", np.array([y for _, y in full], np.int64))
            np.save(FEATS / "s0_ref_pool.npy", allF[[idx[p] for p, _ in fs]])
            np.save(FEATS / "labels_pool.npy", np.array([y for _, y in fs], np.int64))
            (FEATS / "files_pool.json").write_text(json.dumps([str(p.relative_to(ROOT)) for p, _ in fs]))
            continue
        F = np.zeros((len(fs), rep["feat_dim"]), np.float32)
        for i, (p, _) in enumerate(fs):
            xi = torch.from_numpy(preprocess(p, res, mean, std))
            s = time.perf_counter()
            F[i] = mod.forward((xi,))[0].numpy().reshape(-1)
            per.append((time.perf_counter() - s) * 1000)
        np.save(FEATS / f"{name}_{tag}.npy", F)
        np.save(FEATS / f"labels_{tag}.npy", np.array([y for _, y in fs], np.int64))
        (FEATS / f"files_{tag}.json").write_text(json.dumps([str(p.relative_to(ROOT)) for p, _ in fs]))
        print(name, split, F.shape, f"{time.time() - t0:.0f}s", flush=True)
    if per:
        rep["extract_ms_median"] = round(float(np.median(per)), 1)
        rep["extract_ms_p90"] = round(float(np.percentile(per, 90)), 1)
    rep["total_s"] = round(time.time() - t0, 1)
    rep_path.write_text(json.dumps(rep, indent=1))
    print("done", name, rep.get("extract_ms_median"), "ms/img", flush=True)


def l2(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def evaluate():
    from scipy.stats import binomtest
    from sklearn.linear_model import LogisticRegression

    ytr, yte = np.load(FEATS / "labels_pool.npy"), np.load(FEATS / "labels_test.npy")
    ftr, fte = json.loads((FEATS / "files_pool.json").read_text()), json.loads((FEATS / "files_test.json").read_text())
    # leakage hunt 1: exact byte duplicates across pool/test
    h = lambda f: hashlib.md5((ROOT / f).read_bytes()).hexdigest()  # noqa: E731
    pool_h = {h(f) for f in ftr}
    exact_dup = sum(h(f) in pool_h for f in fte)
    rows, preds = {}, {}
    for name in CANDS:
        if not (FEATS / f"{name}_pool.npy").exists():
            continue
        Xtr, Xte = l2(np.load(FEATS / f"{name}_pool.npy")), l2(np.load(FEATS / f"{name}_test.npy"))
        r = json.loads((RES / f"{name}_export.json").read_text())
        row = {k: r.get(k) for k in ("params_m", "size_mb", "delegated_nodes", "non_delegated_nodes", "cosine_vs_eager_real",
                                     "eager_ms", "pte_ms_median20", "extract_ms_median", "res", "feat_dim")}
        # zero-shot
        if (FEATS / f"{name}_text.npy").exists():
            T = np.load(FEATS / f"{name}_text.npy")
            row["zeroshot"] = round(100 * float(((Xte @ T.T).argmax(1) == yte).mean()), 1)
        # 20-shot nearest prototype, 50 random support draws from the pool
        accs = []
        for s in range(50):
            rng = np.random.default_rng(SEED + s)
            P = np.stack([Xtr[rng.choice(np.where(ytr == c)[0], 20, replace=False)].mean(0) for c in range(len(CLASSES))])
            accs.append(100 * float(((Xte @ l2(P).T).argmax(1) == yte).mean()))
        row["proto20_mean"], row["proto20_sd"] = round(float(np.mean(accs)), 1), round(float(np.std(accs)), 1)
        row["proto20_min"], row["proto20_max"] = round(min(accs), 1), round(max(accs), 1)
        # logistic head on all pool features
        clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr)
        p = clf.predict(Xte)
        preds[name] = p
        k = int((p == yte).sum())
        row["linear"] = round(100 * k / len(yte), 1)
        row["linear_ci95"] = wilson(k, len(yte))
        row["linear_per_class"] = {c: round(100 * float((p[yte == i] == i).mean()), 1) for i, c in enumerate(CLASSES)}
        # leakage hunt 2: near-duplicates (max cosine of each test image to any pool image, this backbone)
        mx = (Xte @ Xtr.T).max(1)
        row["test_maxcos_to_pool_median"] = round(float(np.median(mx)), 3)
        row["test_near_dup_ge_0.95"] = int((mx >= 0.95).sum())
        keep = mx < 0.95
        row["linear_excl_neardup"] = round(100 * float((p[keep] == yte[keep]).mean()), 1) if keep.any() else None
        rows[name] = row
    if "s0_ref" in preds:  # paired exact McNemar vs the reference on the same 240 test images
        for name, p in preds.items():
            if name == "s0_ref":
                continue
            a, b = preds["s0_ref"] == yte, p == yte
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
            rows[name]["vs_s0_linear"] = {"s0_only_right": n01, "new_only_right": n10,
                                          "mcnemar_p": round(float(binomtest(n10, n01 + n10, 0.5).pvalue), 4) if n01 + n10 else 1.0}
    if (FEATS / "s0_ref_poolfull.npy").exists():  # S0 on the FULL 2287 pool: shows what the 100/class cap costs
        Xf, yf = l2(np.load(FEATS / "s0_ref_poolfull.npy")), np.load(FEATS / "labels_poolfull.npy")
        Xte = l2(np.load(FEATS / "s0_ref_test.npy"))
        k = int((LogisticRegression(C=10, max_iter=2000).fit(Xf, yf).predict(Xte) == yte).sum())
        rows["s0_ref"]["linear_fullpool"] = round(100 * k / len(yte), 1)
        rows["s0_ref"]["linear_fullpool_ci95"] = wilson(k, len(yte))
    out = {"seed": SEED, "n_pool": int(len(ytr)), "pool_cap_per_class": POOL_CAP, "n_test": int(len(yte)),
           "exact_dup_test_in_pool": exact_dup, "models": rows}
    (RES / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    PTE.mkdir(exist_ok=True)
    FEATS.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    if sys.argv[1] == "run":
        for n in sys.argv[2:]:
            try:
                run(n)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
                print("FAILED", n, msg, flush=True)
                (RES / f"{n}_FAILED.txt").write_text(msg)
    else:
        evaluate()
