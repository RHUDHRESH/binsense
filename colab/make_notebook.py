"""Generates colab/train_waste_models.ipynb (kept as a script so the notebook is reviewable as plain code)."""
import ast
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("# BinSense: fine-tune our own waste models on GPU and export to ExecuTorch\n"
   "Runtime > Change runtime type > **GPU**, then Runtime > **Run all**. "
   "The last cell downloads `binsense_models.zip`; put it in the project folder on the laptop.")

code(r'''
!pip -q install "torch==2.12.1" "torchvision==0.27.1" "executorch==1.3.1" "torchao==0.17.0" timm open_clip_torch huggingface_hub
import torch, torchvision; print(torch.__version__, torch.cuda.is_available())
''')

code(r'''
# Data: TrashNet (MIT). The split rule is the same as laptop/fetch_trashnet.py on the laptop,
# so the 240 test photos here are the same 240 used for every other number in the project.
import zipfile, io, random, json, time, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
PER_CLASS = 40
z = zipfile.ZipFile(hf_hub_download("garythung/trashnet", "dataset-resized.zip", repo_type="dataset"))
by = defaultdict(list)
for n in sorted(z.namelist()):
    p = Path(n)
    if p.suffix.lower() == ".jpg" and not p.name.startswith(".") and "__MACOSX" not in n:
        by[p.parent.name].append(n)
LABELS = sorted(by)
pool, test = [], []
for lab, names in by.items():
    step = max(1, len(names) // PER_CLASS); t = set(names[::step][:PER_CLASS])
    for n in names:
        (test if n in t else pool).append((Image.open(io.BytesIO(z.read(n))).convert("RGB"), LABELS.index(lab)))
random.Random(0).shuffle(pool)
val, train = pool[: len(pool) * 15 // 100], pool[len(pool) * 15 // 100 :]
print(LABELS, "train", len(train), "val", len(val), "test", len(test))
''')

code(r'''
import torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
dev = "cuda" if torch.cuda.is_available() else "cpu"

class DS(torch.utils.data.Dataset):
    def __init__(s, items, tf): s.items, s.tf = items, tf
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.tf(s.items[i][0]), s.items[i][1]

class Net(nn.Module):
    """backbone -> L2-normalised feature -> linear head. The Pi runs `backbone` as a .pte and the head in numpy."""
    def __init__(s, backbone, dim, n, scale=20.0):
        super().__init__(); s.backbone, s.fc, s.scale = backbone, nn.Linear(dim, n), scale
    def forward(s, x): return s.fc(F.normalize(s.backbone(x), dim=-1) * s.scale)

class V(nn.Module):
    def __init__(s, v): super().__init__(); s.v = v
    def forward(s, x): return s.v(x)

def make(arch):
    if arch == "mobilenet_v2":
        m = torchvision.models.mobilenet_v2(weights="DEFAULT")
        bb = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return bb, 1280, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], "torchvision mobilenet_v2 IMAGENET1K_V2"
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
    return V(clip.visual), 512, 256, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], "open_clip MobileCLIP2-S0 dfndr2b (image tower)"

def run(arch, epochs, lr_bb, lr_head, bs=48):
    bb, dim, res, mean, std, src = make(arch)
    net = Net(bb, dim, len(LABELS)).to(dev)
    tf_tr = T.Compose([T.RandomResizedCrop(res, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.RandomRotation(20), T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), T.Normalize(mean, std)])
    tf_te = T.Compose([T.Resize(res), T.CenterCrop(res), T.ToTensor(), T.Normalize(mean, std)])
    counts = np.bincount([y for _, y in train], minlength=len(LABELS))
    sampler = torch.utils.data.WeightedRandomSampler([1.0 / counts[y] for _, y in train], len(train))
    dl = torch.utils.data.DataLoader(DS(train, tf_tr), bs, sampler=sampler, num_workers=2, drop_last=True)
    ev = lambda items: torch.utils.data.DataLoader(DS(items, tf_te), 64, num_workers=2)
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": lr_bb},
                             {"params": net.fc.parameters(), "lr": lr_head}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr_head], total_steps=epochs * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    @torch.no_grad()
    def predict(items):
        net.eval(); P, Y = [], []
        for x, y in ev(items): P.append(net(x.to(dev)).argmax(1).cpu()); Y.append(y)
        return torch.cat(P), torch.cat(Y)
    best, state, t0 = 0, None, time.time()
    for ep in range(epochs):
        net.train()
        for x, y in dl:
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = F.cross_entropy(net(x.to(dev)), y.to(dev), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        p, y = predict(val); a = (p == y).float().mean().item()
        if a >= best: best, state = a, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"{arch} epoch {ep+1}/{epochs} val {a*100:.1f}% best {best*100:.1f}%  ({time.time()-t0:.0f}s)")
    net.load_state_dict(state)                      # model selected on validation only; test is touched once
    p, y = predict(test); cm = torch.zeros(len(LABELS), len(LABELS), dtype=int)
    for a_, b_ in zip(y, p): cm[a_, b_] += 1
    ta = (p == y).float().mean().item()
    print(f"\n{arch}: TEST {ta*100:.1f}%  ({int((p==y).sum())}/{len(y)})  rows=true cols=pred {LABELS}\n{cm.numpy()}")
    return net.cpu().eval(), dict(arch=arch, source=src, res=res, mean=mean, std=std, val_acc=best, test_acc=ta,
                                  test_correct=int((p == y).sum()), confusion=cm.tolist(), train_seconds=time.time() - t0)

results, nets = {}, {}
nets["mobilenet_v2"], results["mobilenet_v2"] = run("mobilenet_v2", epochs=25, lr_bb=1e-3, lr_head=5e-3)
nets["mobileclip2_s0"], results["mobileclip2_s0"] = run("mobileclip2_s0", epochs=20, lr_bb=5e-5, lr_head=5e-3)
''')

code(r'''
# Export each fine-tuned backbone to ExecuTorch (XNNPACK): FP32 always, INT8 (PT2E static, per-channel) for the CNN.
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e import move_exported_model_to_eval
OUT = Path("binsense_models"); OUT.mkdir(exist_ok=True)

def lower(ep, path):
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()],
                                       compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
    with open(path, "wb") as f: prog.write_to_file(f)
    return round(path.stat().st_size / 1e6, 2)

def count_correct(feat_fn, r, W, b):
    tf = T.Compose([T.Resize(r["res"]), T.CenterCrop(r["res"]), T.ToTensor(), T.Normalize(r["mean"], r["std"])])
    ok = 0
    for im, y in test:
        e = feat_fn(tf(im)[None]).flatten().numpy(); e = e / np.linalg.norm(e)
        ok += int((W @ e + b).argmax() == y)
    return ok

for name, net in nets.items():
    r = results[name]; bb = net.backbone.eval()
    if name == "mobileclip2_s0":
        from timm.utils import reparameterize_model
        bb = reparameterize_model(bb)
    x = torch.rand(1, 3, r["res"], r["res"])
    W = (net.fc.weight.detach() * net.scale).numpy().astype(np.float32); b = net.fc.bias.detach().numpy().astype(np.float32)
    fp = OUT / f"own_{name}_{r['res']}_fp32.pte"
    r["fp32_mb"] = lower(torch.export.export(bb, (x,)), fp)
    m = _load_for_executorch(str(fp))
    r["fp32_pte_test_correct"] = count_correct(lambda t: m.forward((t,))[0], r, W, b)
    if name == "mobilenet_v2":
        q = XNNPACKQuantizer(); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prep = prepare_pt2e(torch.export.export(bb, (x,)).module(), q); move_exported_model_to_eval(prep)
        tf = T.Compose([T.Resize(r["res"]), T.CenterCrop(r["res"]), T.ToTensor(), T.Normalize(r["mean"], r["std"])])
        with torch.no_grad():
            for im, _ in train[:150]: prep(tf(im)[None])
        conv = convert_pt2e(prep)
        with torch.no_grad(): r["int8_converted_test_correct"] = count_correct(lambda t: conv(t), r, W, b)
        ip = OUT / f"own_{name}_{r['res']}_int8.pte"
        r["int8_mb"] = lower(torch.export.export(conv, (x,)), ip)
        mi = _load_for_executorch(str(ip))
        r["int8_pte_test_correct_on_colab_x86"] = count_correct(lambda t: mi.forward((t,))[0], r, W, b)
    meta = {"model": r["source"] + ", fine-tuned on TrashNet", "pretrained": "fine-tuned", "logit_scale": 1.0, "labels": LABELS,
            "preprocess": {"size": r["res"], "mean": r["mean"], "std": r["std"]}}
    np.savez(OUT / f"own_{name}_{r['res']}_pack.npz", meta=np.array(json.dumps(meta)), head_w=W, head_b=b)
    print(name, {k: v for k, v in r.items() if k != "confusion"})
json.dump(results, open(OUT / "results.json", "w"), indent=1)
''')

code(r'''
import shutil; shutil.make_archive("binsense_models", "zip", "binsense_models")
from google.colab import files; files.download("binsense_models.zip")
''')

for c in cells:
    if c["cell_type"] == "code":
        ast.parse("\n".join(line for line in c["source"].split("\n") if not line.startswith("!")))
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).parent / "train_waste_models.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "- syntax ok")
