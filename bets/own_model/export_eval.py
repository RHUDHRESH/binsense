"""Export the fine-tuned checkpoint to FP32 and INT8 XNNPACK .pte (Arm course PT2E flow) and evaluate
each .pte itself over the full held-out test set (data/trashnet, 240 images).

INT8: XNNPACKQuantizer per-channel symmetric, final classifier Linear kept FP32 (filter function),
move_exported_model_to_eval, calibration on 120 TRAIN-split pool images with the exact eval preprocessing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import export_tower  # noqa: E402  (configures flatc for ExecuTorch on Windows)
from train import CLASSES, EVAL_TF, HEAD, MEAN, STD, build, list_dir, metrics, split_pool  # noqa: E402


def load_tensors(items):
    from PIL import Image

    return torch.stack([EVAL_TF(Image.open(p).convert("RGB")) for p, _ in items]), np.array([y for _, y in items])


def run_pte_all(path: Path, X: torch.Tensor):
    from executorch.extension.pybindings import portable_lib as pl

    m = pl._load_for_executorch(str(path))
    outs = [m.forward((X[i:i + 1].contiguous(),))[0].clone() for i in range(len(X))]
    x0 = X[:1].contiguous()
    for _ in range(5):
        m.forward((x0,))
    t = []
    for _ in range(40):
        s = time.perf_counter(); m.forward((x0,)); t.append((time.perf_counter() - s) * 1000)
    return torch.cat(outs), float(np.median(t)), float(np.percentile(t, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v2", choices=list(HEAD))
    ap.add_argument("--num-calib", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    from executorch.extension.pybindings import portable_lib as pl
    pl._unsafe_reset_threadpool(4)  # before any .pte is loaded
    rep = {"arch": args.arch, "et_threadpool": int(pl._threadpool_get_thread_count())}

    model = build(args.arch, pretrained=False)
    model.load_state_dict(torch.load(HERE / f"{args.arch}_best.pt", map_location="cpu"))
    model.eval()
    te = list_dir(ROOT / "data/trashnet")
    Xte, yte = load_tensors(te)
    with torch.inference_mode():
        ref = torch.cat([model(Xte[i:i + 40]) for i in range(0, len(Xte), 40)])
        x0 = Xte[:1]
        for _ in range(3):
            model(x0)
        t = []
        for _ in range(20):
            s = time.perf_counter(); model(x0); t.append((time.perf_counter() - s) * 1000)
    rep["eager"] = {"test": metrics(ref.argmax(1).numpy(), yte), "laptop_median_ms_torch4": round(float(np.median(t)), 2)}
    print("eager acc", rep["eager"]["test"]["acc"], flush=True)
    example = (torch.randn(1, 3, 224, 224),)

    # FP32 .pte
    fp32_path = HERE / f"{args.arch}_fp32_xnnpack.pte"
    info = export_tower.lower(torch.export.export(model, example), fp32_path)
    out, med, p90 = run_pte_all(fp32_path, Xte)
    info.update({"test": metrics(out.argmax(1).numpy(), yte), "laptop_median_ms": round(med, 2), "laptop_p90_ms": round(p90, 2),
                 "max_abs_logit_diff_vs_eager": round(float((out - ref).abs().max()), 5),
                 "top1_agreement_vs_eager": round(float((out.argmax(1) == ref.argmax(1)).float().mean()), 4)})
    rep["fp32_pte"] = info
    print("fp32", json.dumps({k: v for k, v in info.items() if k != "test"}), "acc", info["test"]["acc"], flush=True)

    # INT8 .pte (Arm course PT2E flow)
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,
                                                                        get_symmetric_quantization_config)
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    head = HEAD[args.arch]

    def skip_classifier_head(node) -> bool:
        for fqn, _ in node.meta.get("nn_module_stack", {}).values():
            if head in str(fqn):
                return False
        return True

    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    q.set_filter_function(skip_classifier_head)
    prepared = prepare_pt2e(torch.export.export(model, example).module(), q)
    move_exported_model_to_eval(prepared)
    tr, _ = split_pool(args.seed)
    step = max(1, len(tr) // args.num_calib)
    calib = sorted(tr, key=lambda t: str(t[0]))[::step][:args.num_calib]  # evenly spaced over sorted train files
    Xc, yc = load_tensors(calib)
    with torch.inference_mode():
        for i in range(len(Xc)):
            prepared(Xc[i:i + 1])
    int8_path = HERE / f"{args.arch}_int8_xnnpack.pte"
    converted = convert_pt2e(prepared)
    # Trusted INT8 accuracy: the PT2E-converted (fake/reference-quantized) PyTorch module, run in Python.
    with torch.inference_mode():
        qout = torch.cat([converted(Xte[i:i + 1]) for i in range(len(Xte))])
    cosq = torch.nn.functional.cosine_similarity(qout, ref, dim=1)
    rep["int8_pt2e_converted"] = {
        "test": metrics(qout.argmax(1).numpy(), yte),
        "top1_agreement_vs_eager": round(float((qout.argmax(1) == ref.argmax(1)).float().mean()), 4),
        "logit_cosine_vs_eager_median": round(float(cosq.median()), 4),
        "logit_cosine_vs_eager_min": round(float(cosq.min()), 4)}
    print("int8 pt2e-converted acc", rep["int8_pt2e_converted"]["test"]["acc"], flush=True)
    np.save(HERE / f"{args.arch}_test_logits_int8pt2e.npy", qout.numpy())
    info = export_tower.lower(torch.export.export(converted, example), int8_path)
    out, med, p90 = run_pte_all(int8_path, Xte)
    cosp = torch.nn.functional.cosine_similarity(out, ref, dim=1)
    info.update({"NOTE": "INT8 .pte executed on Windows x86 runtime: NOT TRUSTED (coordinator reports wrong INT8 "
                         "outputs on this host); validate on the Pi. Latency here not meaningful.",
                 "test_windows_runtime_not_trusted": metrics(out.argmax(1).numpy(), yte),
                 "laptop_median_ms_not_meaningful": round(med, 2),
                 "logit_cosine_vs_eager_median_windows": round(float(cosp.median()), 4),
                 "top1_agreement_vs_eager_windows": round(float((out.argmax(1) == ref.argmax(1)).float().mean()), 4),
                 "num_calib": len(Xc), "calib_class_counts": np.bincount(yc, minlength=6).tolist(),
                 "calib_source": "train split of data/trashnet_pool (never val, never test)"})
    rep["int8_pte"] = info
    print("int8 pte", json.dumps({k: v for k, v in info.items() if k != "test_windows_runtime_not_trusted"}),
          "windows-runtime acc (untrusted)", info["test_windows_runtime_not_trusted"]["acc"], flush=True)
    np.save(HERE / f"{args.arch}_test_logits_int8pte.npy", out.numpy())

    (HERE / f"export_{args.arch}.json").write_text(json.dumps(rep, indent=1))
    spec_path = HERE / "model_spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    spec.update({
        "labels": CLASSES,
        "input": {"shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW", "channel_order": "RGB",
                  "preprocessing": {"resize_shorter_side": 256, "resize_interpolation": "bilinear",
                                    "center_crop": 224, "scale": "pixel/255", "mean": MEAN, "std": STD}},
        "output": "logits[1,6] in label order; softmax on device",
    })
    spec.setdefault("models", {})[args.arch] = {
        "fp32_pte": fp32_path.name, "int8_pte": int8_path.name,
        "test_acc": {"eager": rep["eager"]["test"]["acc"], "fp32_pte": rep["fp32_pte"]["test"]["acc"],
                     "int8_pt2e_converted_python": rep["int8_pt2e_converted"]["test"]["acc"],
                     "int8_pte": "validate on Pi (Windows INT8 runtime untrusted)"}}
    spec_path.write_text(json.dumps(spec, indent=1))
    print("wrote", spec_path, flush=True)


if __name__ == "__main__":
    main()
