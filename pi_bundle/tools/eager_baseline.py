#!/usr/bin/env python3
"""PyTorch eager baseline on the Pi (needs: pip install open_clip_torch timm). Same model, no ExecuTorch.
   python eager_baseline.py MobileCLIP2-S0 dfndr2b 192 [runs]
"""
import statistics
import sys
import time

import open_clip
import torch

model, pretrained, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 20
clip, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained)
clip.eval()
try:
    from timm.utils import reparameterize_model
    clip = reparameterize_model(clip)
except Exception:
    pass
x = torch.rand(1, 3, res, res)
t = []
with torch.inference_mode():
    for i in range(runs + 3):
        s = time.perf_counter(); clip.visual(x); d = (time.perf_counter() - s) * 1000
        if i >= 3: t.append(d)
t.sort()
print(f"eager {model} {res}px threads={torch.get_num_threads()}: median {statistics.median(t):.1f} ms, min {t[0]:.1f}, max {t[-1]:.1f}")
