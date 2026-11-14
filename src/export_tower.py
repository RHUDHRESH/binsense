#!/usr/bin/env python3
"""Export a CLIP-style image tower to an ExecuTorch XNNPACK .pte (FP32, optional INT8).

Runs on the development machine. Only the image tower goes to the Pi; label text
embeddings are computed here by build_labels.py and shipped as a small matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ExecuTorch shells out to flatc; make the venv's bundled copy visible (Windows + Linux).
_bin = Path(sys.executable).parent
for _name in ("flatc.exe", "flatc"):
    if (_bin / _name).exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(_bin / _name))
os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import torch


class ImageTower(torch.nn.Module):
    """image -> raw embedding."""

    def __init__(self, clip):
        super().__init__()
        self.visual = clip.visual

    def forward(self, x):
        return self.visual(x)  # L2-normalised on the device in numpy: keeps the quantizer away from the norm


def load(model_name: str, pretrained: str):
    import open_clip

    clip, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    clip.eval()
    try:  # fold multi-branch training blocks into single convs (MobileOne/FastViT style)
        from timm.utils import reparameterize_model

        clip = reparameterize_model(clip)
    except Exception as e:  # noqa: BLE001
        print("reparameterize skipped:", e)
    cfg = clip.visual.preprocess_cfg
    return clip, preprocess, cfg


def lower(exported, out: Path):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.devtools.backend_debug import get_delegation_info
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    info = get_delegation_info(edge.exported_program().graph_module)
    program = edge.to_executorch()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        program.write_to_file(f)
    df = info.get_operator_delegation_dataframe()
    left = df[(df["occurrences_in_non_delegated_graphs"] > 0) & (df["op_type"] != "Total")]
    return {
        "pte": out.name,
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "delegated_nodes": int(info.num_delegated_nodes),
        "non_delegated_nodes": int(info.num_non_delegated_nodes),
        "delegated_subgraphs": int(info.num_delegated_subgraphs),
        "non_delegated_ops": {r.op_type: int(r.occurrences_in_non_delegated_graphs) for r in left.itertuples()},
    }


def run_pte(path: Path, x: torch.Tensor, runs: int = 10):
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    m = _load_for_executorch(str(path))
    out = m.forward((x,))[0]
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        m.forward((x,))
        t.append((time.perf_counter() - s) * 1000)
    t.sort()
    return out, t[len(t) // 2]


def calibration_batches(folder: Path, preprocess, n: int):
    from PIL import Image

    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    step = max(1, len(files) // n)
    for p in files[::step][:n]:
        yield preprocess(Image.open(p).convert("RGB")).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MobileCLIP2-S0")
    ap.add_argument("--pretrained", default="dfndr2b")
    ap.add_argument("--res", type=int, default=0, help="override input resolution (0 = native)")
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--calib", type=Path, default=None, help="image folder for INT8 calibration")
    ap.add_argument("--num-calib", type=int, default=100)
    ap.add_argument("--int8-scope", default="", help="comma list of module-name fragments to quantize (empty = all)")
    ap.add_argument("--int8-name", default="int8")
    ap.add_argument("--skip-fp32", action="store_true")
    args = ap.parse_args()

    clip, preprocess, cfg = load(args.model, args.pretrained)
    res = args.res or cfg["size"][-1]
    tower = ImageTower(clip).eval()
    x = torch.rand(1, 3, res, res)
    tag = f"{args.model.lower()}_{res}"
    report = {"model": args.model, "pretrained": args.pretrained, "resolution": res,
              "image_tower_params_m": round(sum(p.numel() for p in tower.parameters()) / 1e6, 2)}

    with torch.inference_mode():
        ref = tower(x)
        s = time.perf_counter()
        for _ in range(5):
            tower(x)
        report["laptop_eager_ms"] = round((time.perf_counter() - s) * 200, 1)

    spec_path = args.out / f"{tag}_spec.json"
    if spec_path.exists():
        report = {**json.loads(spec_path.read_text()), **report}
    if not args.skip_fp32:
        exported = torch.export.export(tower, (x,))
        fp32 = lower(exported, args.out / f"{tag}_fp32.pte")
        out, ms = run_pte(args.out / fp32["pte"], x)
        fp32["laptop_ms"] = round(ms, 1)
        fp32["cosine_vs_eager"] = round(float(torch.nn.functional.cosine_similarity(out, ref).item()), 5)
        report["fp32"] = fp32
        print(json.dumps(fp32, indent=1))

    if args.calib:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        q = XNNPACKQuantizer()
        q.set_global(get_symmetric_quantization_config(is_per_channel=True))
        if args.int8_scope:
            # Mixed precision: quantize only the named conv stages; attention + projection stay FP32.
            keys = [k.strip() for k in args.int8_scope.split(",")]

            def in_scope(node) -> bool:
                stack = " ".join(str(f) for f, _ in node.meta.get("nn_module_stack", {}).values())
                return any(k in stack for k in keys)

            q.set_filter_function(in_scope)
        prepared = prepare_pt2e(torch.export.export(tower, (x,)).module(), q)
        from torchao.quantization.pt2e import move_exported_model_to_eval
        move_exported_model_to_eval(prepared)
        real = None
        with torch.inference_mode():
            for b in calibration_batches(args.calib, preprocess, args.num_calib):
                if b.shape[-1] != res:
                    b = torch.nn.functional.interpolate(b, size=(res, res), mode="bilinear")
                prepared(b)
                real = b
        converted = convert_pt2e(prepared)
        probe = real if real is not None else x
        with torch.no_grad():
            ref_real = tower(probe)
            conv_out = converted(probe)
        int8 = lower(torch.export.export(converted, (x,)), args.out / f"{tag}_{args.int8_name}.pte")
        # Quality of the quantized graph itself (trustworthy on any host). The .pte check below is only
        # meaningful on the target: INT8 XNNPACK kernels on Windows x86 gave wrong outputs in our tests.
        int8["converted_cosine_vs_eager"] = round(float(torch.nn.functional.cosine_similarity(conv_out, ref_real).item()), 5)
        out, ms = run_pte(args.out / int8["pte"], probe)
        int8["laptop_ms"] = round(ms, 1)
        int8["pte_cosine_on_this_host"] = round(float(torch.nn.functional.cosine_similarity(out, ref_real).item()), 5)
        int8["scope"] = args.int8_scope or "all"
        report[args.int8_name] = int8
        print(json.dumps(int8, indent=1))

    report["preprocess"] = {"size": res, "mean": list(cfg["mean"]), "std": list(cfg["std"]),
                            "interpolation": cfg.get("interpolation", "bicubic"),
                            "resize_mode": cfg.get("resize_mode", "shortest")}
    spec_path.write_text(json.dumps(report, indent=1))
    print("wrote", spec_path)


if __name__ == "__main__":
    main()
