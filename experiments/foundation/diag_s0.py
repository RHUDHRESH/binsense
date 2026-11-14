"""Why is S0 .pte vs eager cosine 0.41 on a real image when the spec says 1.0 on torch.rand?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bet  # noqa: E402  (sets threads, flatc)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from executorch.extension.pybindings import portable_lib as pl  # noqa: E402

pl._unsafe_reset_threadpool(4)
tower, res, mean, std, _, _ = bet.build("s0_ref")
m = pl._load_for_executorch(str(bet.CANDS["s0_ref"]["pte"]))
cache = bet.s0_cached_pool()
pool = bet.files_of("trashnet_pool")
torch.manual_seed(0)
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))  # noqa: E731
with torch.inference_mode():
    for tag, x in [("rand", torch.rand(1, 3, res, res))] + [
            (f"pool[{i}]", torch.from_numpy(bet.preprocess(pool[i][0], res, mean, std))) for i in (0, 500, 1500)]:
        e1, e2, p = tower(x), tower(x), m.forward((x,))[0]
        extra = f" cached_vs_pte={cos(torch.from_numpy(cache[int(tag[5:-1])]), p):.4f}" if tag != "rand" else ""
        print(f"{tag}: eager_vs_eager={cos(e1, e2):.4f} pte_vs_eager={cos(p, e1):.4f}{extra} "
              f"|eager|={float(e1.norm()):.2f} |pte|={float(p.norm()):.2f}", flush=True)
