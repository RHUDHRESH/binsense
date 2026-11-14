"""Download candidate weights into the HF cache (no export, no eval)."""
import sys
import time

import torch

torch.set_num_threads(2)
which = sys.argv[1:] or ["dinov2", "dinov2reg", "s1", "s2", "b32"]
for w in which:
    t = time.time()
    try:
        if w.startswith("dinov2"):
            import timm

            name = "vit_small_patch14_reg4_dinov2.lvd142m" if w == "dinov2reg" else "vit_small_patch14_dinov2.lvd142m"
            timm.create_model(name, pretrained=True, img_size=224, num_classes=0)
        else:
            import open_clip

            m, p = {"s1": ("MobileCLIP-S1", "datacompdr"), "s2": ("MobileCLIP2-S2", "dfndr2b"),
                    "b32": ("ViT-B-32", "laion2b_s34b_b79k")}[w]
            open_clip.create_model_and_transforms(m, pretrained=p)
        print(w, "ok", round(time.time() - t, 1), "s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(w, "FAIL", type(e).__name__, str(e)[:300], flush=True)
