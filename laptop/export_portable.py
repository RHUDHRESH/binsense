#!/usr/bin/env python3
"""Export the same image tower WITHOUT a backend delegate (portable kernels only): the 'no XNNPACK' baseline row."""
import sys
from pathlib import Path

import export_tower as et
import torch
from executorch.exir import EdgeCompileConfig, to_edge

model, pretrained, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
clip, _, _ = et.load(model, pretrained)
tower = et.ImageTower(clip).eval()
x = torch.rand(1, 3, res, res)
prog = to_edge(torch.export.export(tower, (x,)), compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
out = Path("../artifacts") / f"{model.lower()}_{res}_portable.pte"
with out.open("wb") as f:
    prog.write_to_file(f)
with torch.no_grad():
    ref = tower(x)
o, ms = et.run_pte(out, x, runs=3)
print(out.name, round(out.stat().st_size / 1e6, 2), "MB | laptop ms", round(ms, 1), "| cosine", round(float(torch.nn.functional.cosine_similarity(o, ref).item()), 5))
