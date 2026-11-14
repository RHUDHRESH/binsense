"""Accuracy experiments on frozen MobileCLIP2-S0 features. Selection uses pool CV only; test is reported, never tuned on."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "pi_bundle")
from binsense import BinSense, softmax
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

L = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
engine = BinSense("artifacts/mobileclip2-s0_256_fp32.pte", "artifacts/trashnet_pack.npz")
C = Path("experiments/accuracy/cache"); C.mkdir(exist_ok=True)

def object_crop(path, res, pad=0.12):
    """Zoom on the object: bounding box of pixels that differ from the border colour."""
    im = Image.open(path).convert("RGB"); a = np.asarray(im.resize((128, 96))).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]); bg = np.median(border, 0)
    m = np.abs(a - bg).sum(-1) > 60
    if m.sum() < 40: return None
    ys, xs = np.where(m); h, w = m.shape
    x0, x1, y0, y1 = xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h
    s = max(x1 - x0, y1 - y0) * (1 + 2 * pad); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    W, H = im.size; side = min(max(s * max(W, H), 0.35 * min(W, H)), min(W, H))
    l = min(max(cx * W - side / 2, 0), W - side); t = min(max(cy * H - side / 2, 0), H - side)
    return im.crop((int(l), int(t), int(l + side), int(t + side))).resize((res, res), Image.BILINEAR)

def emb(split, mode):
    f = C / f"{split}_{mode}.npz"
    if f.exists():
        z = np.load(f); return z["X"], z["y"], list(z["names"])
    X, y, names = [], [], []
    for i, lab in enumerate(L):
        for p in sorted((Path("data") / split / lab).glob("*.jpg")):
            if mode == "full": x = engine.preprocess(p)
            else:
                im = object_crop(p, engine.res)
                x = engine.preprocess(p) if im is None else ((np.asarray(im, np.float32).transpose(2, 0, 1) / 255 - engine.mean) / engine.std)[None]
            X.append(engine.embed(x)); y.append(i); names.append(p.name)
    X, y = np.stack(X), np.array(y); np.savez(f, X=X, y=y, names=np.array(names)); return X, y, names

Xp, yp, _ = emb("trashnet_pool", "full"); Xt, yt, names = emb("trashnet", "full")
Xpc, _, _ = emb("trashnet_pool", "crop"); Xtc, _, _ = emb("trashnet", "crop")
n = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
feats = {"full": (Xp, Xt), "crop": (Xpc, Xtc), "avg(full,crop)": (n(Xp + Xpc), n(Xt + Xtc)), "concat": (np.hstack([Xp, Xpc]), np.hstack([Xt, Xtc]))}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
out = []
for fn, (A, B) in feats.items():
    for cn, clf in {"LR C=10": LogisticRegression(C=10, max_iter=3000, class_weight="balanced"),
                    "LR C=100": LogisticRegression(C=100, max_iter=3000, class_weight="balanced"),
                    "SVM rbf C=10": SVC(C=10, kernel="rbf", class_weight="balanced", probability=True, random_state=0)}.items():
        c = cross_val_score(clf, A, yp, cv=cv).mean(); clf.fit(A, yp); pr = clf.predict_proba(B)
        out.append((c, fn, cn, (pr.argmax(1) == yt).mean(), pr))
        print(f"{fn:16s} {cn:13s} poolCV {c*100:5.1f}  test {(pr.argmax(1)==yt).mean()*100:5.1f}  ({int((pr.argmax(1)==yt).sum())}/240)", flush=True)
best = max(out, key=lambda r: r[0]); pr = best[4]; pred = pr.argmax(1); conf = pr.max(1)
print("\nCHOSEN BY POOL CV:", best[1], best[2], "-> test", f"{best[3]*100:.1f}%")
print("selective answering (abstain below threshold):")
for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
    k = conf >= th; print(f"  conf>={th:.1f}: answers {k.mean()*100:5.1f}% of items, accuracy on those {(pred[k]==yt[k]).mean()*100:5.1f}%")
rec = np.array([1, 1, 1, 1, 1, 0])  # recyclable vs non-recyclable view of the same predictions
print("as recyclable vs non-recyclable:", f"{(rec[pred]==rec[yt]).mean()*100:.1f}%")
print("still wrong:", [(names[i], L[yt[i]], L[pred[i]], round(float(conf[i]), 2)) for i in np.where(pred != yt)[0]])
np.save(C / "best_test_proba.npy", pr); json.dump({"feat": best[1], "clf": best[2]}, open(C / "best.json", "w"))
