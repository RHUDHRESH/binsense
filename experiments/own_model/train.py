"""Fine-tune a torchvision MobileNet on the TrashNet pool (CPU), with a linear-probe baseline.

Split: data/trashnet_pool -> 85% train / 15% val (stratified, seed). Test = data/trashnet, touched
only once at the end for the selected checkpoint. Preprocessing at eval = Arm course contract
(Resize 256, CenterCrop 224, ImageNet mean/std).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])
HEAD = {"mobilenet_v2": "classifier.1", "mobilenet_v3_small": "classifier.3"}
HALF = {"mobilenet_v2": 9, "mobilenet_v3_small": 6}  # features[:HALF] frozen during warm-up


def list_dir(d: Path):
    items = [(p, CLASSES.index(p.parent.name)) for c in CLASSES for p in sorted((d / c).glob("*.jpg"))]
    return items


def split_pool(seed: int):
    from sklearn.model_selection import train_test_split

    items, seen = [], set()
    for p, y in list_dir(ROOT / "data/trashnet_pool"):  # drop byte-identical duplicates (3 found)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            items.append((p, y))
    ys = [y for _, y in items]
    tr, va = train_test_split(items, test_size=0.15, stratify=ys, random_state=seed)
    return tr, va


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def _winit(_):
    torch.set_num_threads(1)


def loader(items, tf, bs, shuffle, workers=0):  # 0: worker spawn starves under host contention
    return torch.utils.data.DataLoader(DS(items, tf), batch_size=bs, shuffle=shuffle, num_workers=workers,
                                       worker_init_fn=_winit, persistent_workers=workers > 0)


def build(arch: str, pretrained=True):
    import torchvision.models as M

    if arch == "mobilenet_v2":
        m = M.mobilenet_v2(weights=M.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    elif arch == "mobilenet_v3_small":
        m = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    else:
        raise ValueError(arch)
    return m


@torch.inference_mode()
def predict(model, dl):
    model.eval()
    logits, ys = [], []
    for x, y in dl:
        logits.append(model(x)), ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def metrics(pred, y):
    from sklearn.metrics import confusion_matrix

    pred, y = np.asarray(pred), np.asarray(y)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    rec = cm.diagonal() / cm.sum(1).clip(min=1)
    acc = float((pred == y).mean())
    n = len(y)
    z = 1.96  # Wilson 95% interval for accuracy
    c = (acc + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"acc": round(acc, 4), "acc_wilson95": [round(c - h, 4), round(c + h, 4)],
            "balanced_acc": round(float(rec.mean()), 4),
            "recall": {k: round(float(r), 4) for k, r in zip(CLASSES, rec)}, "confusion": cm.tolist(), "n": n}


@torch.inference_mode()
def feats(model, dl):
    model.eval()
    f, ys = [], []
    for x, y in dl:
        h = model.features(x)
        f.append(nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)), ys.append(y)
    return torch.cat(f).numpy(), torch.cat(ys).numpy()


def linear_probe(tr, va, te, seed):
    """Baseline: frozen ImageNet MobileNetV2 pooled features + logistic regression (C picked on val)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    m = build("mobilenet_v2")
    Xtr, ytr = feats(m, loader(tr, EVAL_TF, 64, False))
    Xva, yva = feats(m, loader(va, EVAL_TF, 64, False))
    Xte, yte = feats(m, loader(te, EVAL_TF, 64, False))
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        va_bal = metrics(clf.predict(sc.transform(Xva)), yva)["balanced_acc"]
        if best is None or va_bal > best[0]:
            best = (va_bal, C, clf)
    va_bal, C, clf = best
    # Near-duplicate audit in ImageNet feature space: cosine of each test image to its nearest pool image.
    pool = np.concatenate([Xtr, Xva])
    nrm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    cos = nrm(Xte) @ nrm(pool).T
    nn_cos = cos.max(1)
    pred = clf.predict(sc.transform(Xte))
    far = nn_cos < np.quantile(nn_cos, 0.5)
    return {"C": C, "val_balanced_acc": va_bal, "test": metrics(pred, yte),
            "nn_cos_quantiles": {q: round(float(np.quantile(nn_cos, q)), 4) for q in (0.1, 0.5, 0.9, 0.99, 1.0)},
            "n_test_nn_cos_ge_0.95": int((nn_cos >= 0.95).sum()), "n_test_nn_cos_ge_0.90": int((nn_cos >= 0.90).sum()),
            }, nn_cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v2", choices=list(HEAD))
    ap.add_argument("--budget-min", type=float, default=10.0)
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--warm-epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    t0 = time.time()
    tr, va = split_pool(args.seed)
    te = list_dir(ROOT / "data/trashnet")
    rec = {"arch": args.arch, "args": vars(args), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "train_class_counts": np.bincount([y for _, y in tr], minlength=6).tolist(),
           "torch": torch.__version__, "classes": CLASSES}
    # Guard: test files must not appear in train/val.
    assert not ({p.name for p, _ in te} & {p.name for p, _ in tr + va})

    model = build(args.arch)
    counts = np.bincount([y for _, y in tr], minlength=6)
    w = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    head_name = HEAD[args.arch].split(".")[0]
    head_p = [p for n, p in model.named_parameters() if n.startswith(head_name)]
    body_p = [p for n, p in model.named_parameters() if not n.startswith(head_name)]
    opt = torch.optim.AdamW([{"params": head_p, "lr": 1e-3}, {"params": body_p, "lr": 4e-4}], weight_decay=1e-2)
    dl_tr = loader(tr, TRAIN_TF, args.bs, True)
    t0 = time.time()  # training budget clock starts here (probe runs after training)
    dl_va = loader(va, EVAL_TF, 64, False)
    frozen = list(model.features[:HALF[args.arch]].parameters())

    def set_frozen(flag):
        for p in frozen:
            p.requires_grad_(not flag)

    # Time a few unfrozen steps to size the cosine schedule inside the budget.
    model.train()
    it = iter(dl_tr)
    xb, yb = next(it)
    ts = []
    for _ in range(3):
        s = time.time()
        crit(model(xb), yb).backward()
        opt.zero_grad(set_to_none=True)
        ts.append(time.time() - s)
    steps_per_epoch = len(dl_tr)
    est_epoch = min(ts) * steps_per_epoch * 1.15 + 15  # + val pass
    remaining = args.budget_min * 60 - (time.time() - t0)
    epochs = int(max(3, min(args.max_epochs, remaining // est_epoch)))
    rec["est_epoch_sec"], rec["planned_epochs"] = round(est_epoch, 1), epochs
    print(f"step {min(ts):.2f}s  est epoch {est_epoch:.0f}s  -> {epochs} epochs", flush=True)
    total = epochs * steps_per_epoch
    warm = steps_per_epoch // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best, hist, bad = None, [], 0
    for ep in range(epochs):
        es = time.time()
        set_frozen(ep < args.warm_epochs)
        model.train()
        if ep < args.warm_epochs:
            model.features[:HALF[args.arch]].eval()  # keep frozen BN statistics fixed
        tl, n = 0.0, 0
        for x, y in dl_tr:
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(), sched.step()
            tl += loss.item() * len(y); n += len(y)
        lv, yv = predict(model, dl_va)
        vm = metrics(lv.argmax(1).numpy(), yv.numpy())
        vloss = float(nn.functional.cross_entropy(lv, yv))
        h = {"epoch": ep + 1, "frozen_half": ep < args.warm_epochs, "train_loss": round(tl / n, 4),
             "val_loss": round(vloss, 4), "val_acc": vm["acc"], "val_bal": vm["balanced_acc"],
             "sec": round(time.time() - es, 1)}
        hist.append(h)
        print(json.dumps(h), flush=True)
        key = (vm["balanced_acc"], -vloss)
        if best is None or key > best[0]:
            best = (key, ep + 1, {k: v.clone() for k, v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            print("early stop", flush=True); break
        if time.time() - t0 > args.budget_min * 60 * 1.1:
            print("budget stop", flush=True); break
    rec["history"], rec["best_epoch"] = hist, best[1]
    model.load_state_dict(best[2])
    torch.save(model.state_dict(), OUT / f"{args.arch}_best.pt")
    rec["train_sec"] = round(time.time() - t0, 1)
    nn_cos = None
    if args.probe:
        s = time.time()
        rec["linear_probe"], nn_cos = linear_probe(tr, va, te, args.seed)
        rec["linear_probe"]["sec"] = round(time.time() - s, 1)
        print("probe test acc", rec["linear_probe"]["test"]["acc"], json.dumps({k: v for k, v in rec["linear_probe"].items() if k != "test"}), flush=True)
    # Single test evaluation of the selected checkpoint (eager FP32).
    lt, yt = predict(model, loader(te, EVAL_TF, 64, False))
    pred = lt.argmax(1).numpy()
    rec["test_eager"] = metrics(pred, yt.numpy())
    if nn_cos is not None:  # accuracy on the half of test images least similar to any pool image
        far = nn_cos < np.median(nn_cos)
        rec["test_eager_far_half_acc"] = round(float((pred[far] == yt.numpy()[far]).mean()), 4)
        rec["test_eager_near_half_acc"] = round(float((pred[~far] == yt.numpy()[~far]).mean()), 4)
        np.save(OUT / "test_nn_cos.npy", nn_cos)
    np.save(OUT / f"{args.arch}_test_logits_eager.npy", lt.numpy())
    (OUT / f"run_{args.arch}.json").write_text(json.dumps(rec, indent=1))
    print("TEST eager", json.dumps({k: rec["test_eager"][k] for k in ("acc", "acc_wilson95", "recall")}),
          "best epoch", best[1], "train_sec", rec["train_sec"], flush=True)


if __name__ == "__main__":
    main()
