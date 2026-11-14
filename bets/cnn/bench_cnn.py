#!/usr/bin/env python3
"""Bet 'cnn': frozen ImageNet timm backbone -> ExecuTorch/XNNPACK FP32 + INT8 .pte -> linear head.

For one timm model:
  1. export raw pooled features (num_classes=0, NO L2 norm inside the graph) to FP32 .pte
  2. PT2E INT8 (XNNPACKQuantizer, per-channel symmetric) calibrated on trashnet_pool images;
     if cosine(int8, eager) on real images < 0.95, retry with head layers excluded from quantization
  3. latency of each .pte with the ExecuTorch threadpool forced to 4 threads (batch 1)
  4. features for pool (fit) + held-out test, extracted THROUGH each .pte at batch 1, cached to .npy
  5. LogisticRegression(C=10, max_iter=2000) on L2-normalised pool features -> test accuracy

Preprocessing matches edge/jev_edge.py: shortest side -> res (bilinear), centre crop, timm mean/std.
Usage:  python -X utf8 bets/cnn/bench_cnn.py --model mobilenetv3_small_100.lamb_in1k
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import export_tower as et  # noqa: E402  (configures flatc; provides lower())

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
torch.manual_seed(0)
SEED = 0
EXT = {".jpg", ".jpeg", ".png"}
HEAD_KEYS = ["conv_head", "norm_head", "head", "classifier", "global_pool", "pre_logits"]


class Features(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)  # raw features; L2 norm happens in numpy


def list_files(folder: Path):
    files, labels = [], []
    classes = sorted(d.name for d in folder.iterdir() if d.is_dir())
    for i, c in enumerate(classes):
        fs = sorted(p for p in (folder / c).iterdir() if p.suffix.lower() in EXT)
        files += fs
        labels += [i] * len(fs)
    return files, np.asarray(labels), classes


def load_u8(path: Path, res: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(path)
    im.draft("RGB", (res * 2, res * 2))
    im = im.convert("RGB")
    w, h = im.size
    s = res / min(w, h)
    im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - res) // 2, (h - res) // 2
    return np.asarray(im.crop((l, t, l + res, t + res)), np.uint8)


def to_tensor(u8: np.ndarray, mean, std) -> torch.Tensor:
    x = u8.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(((x - mean) / std)[None].copy())


def load_pte(path: Path):
    from executorch.extension.pybindings import portable_lib as pl

    pl._unsafe_reset_threadpool(4)
    return pl._load_for_executorch(str(path))


def time_pte(mod, x, warm=3, runs=30) -> float:
    for _ in range(warm):
        mod.forward((x,))
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        mod.forward((x,))
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t))


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def quantize(m, x, calib, filt=None):
    """PT2E INT8. Returns (converted torch module for quality checks, exported program for lowering)."""
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    if filt is not None:
        q.set_filter_function(filt)
    prepared = prepare_pt2e(torch.export.export(m, (x,)).module(), q)
    move_exported_model_to_eval(prepared)
    with torch.inference_mode():
        for b in calib:
            prepared(b)
    conv = convert_pt2e(prepared)
    move_exported_model_to_eval(conv)
    return conv, torch.export.export(conv, (x,))


def cosine_check(run, m, probes):
    """run: callable tensor->tensor (a .pte module or the converted PT2E module)."""
    cs = []
    with torch.inference_mode():
        for p in probes:
            cs.append(float(torch.nn.functional.cosine_similarity(run(p), m(p)).item()))
    return round(float(np.mean(cs)), 4), round(float(np.min(cs)), 4)


def pte_run(mod):
    return lambda t: mod.forward((t,))[0]


def extract(run, u8s, mean, std, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    with torch.inference_mode():
        feats = np.stack([run(to_tensor(u, mean, std)).numpy()[0] for u in u8s]).astype(np.float32)
    np.save(cache, feats)
    return feats


def head_acc(Xtr, ytr, Xte, yte, balanced: bool):
    from sklearn.linear_model import LogisticRegression

    n = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)  # noqa: E731
    clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced" if balanced else None, random_state=SEED)
    clf.fit(n(Xtr), ytr)
    pred = clf.predict(n(Xte))
    k = int((pred == yte).sum())
    return {"acc": round(k / len(yte), 4), "ci95": wilson(k, len(yte)), "correct": k}, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--num-calib", type=int, default=128)
    args = ap.parse_args()

    import timm

    tag = args.model.split(".")[0]
    d = OUT / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    rep = {"model": args.model, "seed": SEED, "torch": torch.__version__, "timm": timm.__version__}
    t0 = time.time()

    net = timm.create_model(args.model, pretrained=True, num_classes=0).eval()
    rep["params_m_raw"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 3)
    net = timm.utils.reparameterize_model(net)  # no-op for plain nets; fuses RepViT/MobileOne branches
    m = Features(net).eval()
    rep["params_m"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 3)
    cfg = timm.data.resolve_data_config({}, model=net)
    res = args.res or int(cfg["input_size"][-1])
    mean = np.asarray(cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.asarray(cfg["std"], np.float32).reshape(3, 1, 1)
    rep.update(res=res, mean=list(cfg["mean"]), std=list(cfg["std"]))
    x = torch.rand(1, 3, res, res)

    pool_f, ytr, classes = list_files(ROOT / "data" / "trashnet_pool")
    test_f, yte, classes_t = list_files(ROOT / "data" / "trashnet")
    assert classes == classes_t
    rep["classes"] = classes
    (d / "files.json").write_text(json.dumps({"pool": [str(p.relative_to(ROOT)) for p in pool_f],
                                             "test": [str(p.relative_to(ROOT)) for p in test_f],
                                             "classes": classes}))
    pool_u8 = [load_u8(p, res) for p in pool_f]
    test_u8 = [load_u8(p, res) for p in test_f]

    # calibration: evenly spaced pool images; cosine probes: different pool images (not in calib)
    step = max(1, len(pool_f) // args.num_calib)
    calib_idx = list(range(0, len(pool_f), step))[: args.num_calib]
    probe_idx = [i + step // 2 for i in calib_idx[:: max(1, len(calib_idx) // 16)]][:16]
    calib = [to_tensor(pool_u8[i], mean, std) for i in calib_idx]
    probes = [to_tensor(pool_u8[i], mean, std) for i in probe_idx]

    with torch.inference_mode():
        eager_probe = m(probes[0])
    rep["feat_dim"] = int(eager_probe.shape[-1])

    # ---- FP32 ----
    fp32_path = d / f"{tag}_{res}_fp32.pte"
    fp32 = et.lower(torch.export.export(m, (x,)), fp32_path)
    mod = load_pte(fp32_path)
    fp32["laptop_ms_t4"] = round(time_pte(mod, probes[0]), 2)
    fp32["cos_vs_eager_mean"], fp32["cos_vs_eager_min"] = cosine_check(pte_run(mod), m, probes)
    Xtr = extract(pte_run(mod), pool_u8, mean, std, d / f"{tag}_{res}_fp32_pool.npy")
    Xte = extract(pte_run(mod), test_u8, mean, std, d / f"{tag}_{res}_fp32_test.npy")
    fp32["head_bal"], pred_fp = head_acc(Xtr, ytr, Xte, yte, True)
    np.save(d / f"{tag}_{res}_fp32_testpred.npy", pred_fp)
    rep["fp32"] = fp32
    print("FP32", json.dumps({k: v for k, v in fp32.items() if k != "non_delegated_ops"}), flush=True)
    del mod

    # ---- INT8 ----
    # Quality is judged on the PT2E *converted* torch module: XNNPACK INT8 .pte outputs are known to be
    # wrong when executed on this Windows x86 laptop (coordinator evidence: mnv2 converted cos 0.95, .pte 0.33).
    # The .pte is still written for on-Pi testing; its laptop numbers are recorded only as *_unreliable.
    int8_path = d / f"{tag}_{res}_int8.pte"
    conv, ep = quantize(m, x, calib)
    scope = "all"
    cmean, cmin = cosine_check(conv, m, probes)
    if cmean < 0.95:
        print("INT8 full converted cosine", cmean, "-> retry excluding head layers", flush=True)
        rep["int8_full_attempt"] = {"conv_cos_mean": cmean, "conv_cos_min": cmin}

        def filt(node) -> bool:
            stack = " ".join(str(f) for f, _ in node.meta.get("nn_module_stack", {}).values())
            return not any(k in stack for k in HEAD_KEYS)

        conv, ep = quantize(m, x, calib, filt)
        cmean, cmin = cosine_check(conv, m, probes)
        scope = "all_except_head:" + ",".join(HEAD_KEYS)
    int8 = et.lower(ep, int8_path)
    int8.update(scope=scope, conv_cos_vs_eager_mean=cmean, conv_cos_vs_eager_min=cmin)
    mod = load_pte(int8_path)
    int8["pte_cos_on_laptop_unreliable"], _ = cosine_check(pte_run(mod), m, probes)
    int8["pte_ms_t4_on_laptop_unreliable"] = round(time_pte(mod, probes[0]), 2)
    del mod
    # Cost cut (shared CPU, converted module is batch-1 python): the head is fitted on FP32 .pte pool
    # features and applied to INT8 (converted) test features = laptop-trained head on a quantized backbone.
    Xte_q = extract(conv, test_u8, mean, std, d / f"{tag}_{res}_int8conv_test.npy")
    Xtr_fp = np.load(d / f"{tag}_{res}_fp32_pool.npy")
    int8["head_bal_fp32fit"], pred_q = head_acc(Xtr_fp, ytr, Xte_q, yte, True)
    np.save(d / f"{tag}_{res}_int8conv_testpred.npy", pred_q)
    int8["int8_broken"] = bool(cmean < 0.95)
    rep["int8"] = int8
    print("INT8", json.dumps({k: v for k, v in int8.items() if k != "non_delegated_ops"}), flush=True)

    rep["wall_s"] = round(time.time() - t0, 1)
    (OUT / f"result_{tag}_{res}.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / f"result_{tag}_{res}.json")


if __name__ == "__main__":
    main()
