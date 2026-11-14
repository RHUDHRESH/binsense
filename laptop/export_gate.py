#!/usr/bin/env python3
"""Export the fast 'gate' feature extractor (ImageNet-pretrained MobileNetV3-Small) to FP32 + INT8 .pte."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import export_tower as et  # sets up flatc, reuses lower()/run_pte()
import torch


class Features(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv3_small_100.lamb_in1k")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--num-calib", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    import timm
    from PIL import Image
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    net = timm.create_model(args.model, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=net)
    m = Features(net).eval()
    x = torch.rand(1, 3, args.res, args.res)
    tag = f"gate_mnv3s_{args.res}"
    report = {"model": args.model, "pretrained": "imagenet-1k", "resolution": args.res,
              "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
              "preprocess": {"size": args.res, "mean": list(cfg["mean"]), "std": list(cfg["std"])}}

    fp32 = et.lower(torch.export.export(m, (x,)), args.out / f"{tag}_fp32.pte")
    _, fp32["laptop_ms"] = et.run_pte(args.out / fp32["pte"], x)
    report["fp32"] = fp32

    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    files = sorted(p for p in args.calib.rglob("*.jpg"))
    files = files[:: max(1, len(files) // args.num_calib)][: args.num_calib]

    def load(p):
        im = Image.open(p).convert("RGB").resize((args.res, args.res), Image.BILINEAR)
        t = torch.from_numpy(__import__("numpy").asarray(im)).permute(2, 0, 1)[None].float() / 255
        return (t - mean) / std

    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(m, (x,)).module(), q)
    with torch.inference_mode():
        for p in files:
            prepared(load(p))
        probe = load(files[0])
        ref = m(probe)
    int8 = et.lower(torch.export.export(convert_pt2e(prepared), (x,)), args.out / f"{tag}_int8.pte")
    out, int8["laptop_ms"] = et.run_pte(args.out / int8["pte"], probe)
    int8["cosine_vs_eager_real_image"] = round(float(torch.nn.functional.cosine_similarity(out, ref).item()), 4)
    report["int8"] = int8
    (args.out / f"{tag}_spec.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in v if kk != "non_delegated_ops"} for k, v in report.items() if k in ("fp32", "int8")}, indent=1))


if __name__ == "__main__":
    main()
