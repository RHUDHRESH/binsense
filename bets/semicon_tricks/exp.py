"""Two ideas carried over from SEM-to-CAD image matching, tested on the waste models.

A. Confidence as a likelihood-ratio margin (peak vs runner-up), not the raw peak.
   In matching, the raw correlation peak is a poor score; peak-to-sidelobe margin separates right from wrong better.
   Here: does (p1 - p2) or log(p1/p2) flag wrong answers better than max probability?
B. Photometric robustness. Matching scores that survive nonlinear intensity changes beat plain correlation.
   Here: how much does accuracy fall under lighting changes, and does normalising the photo first recover it?
Indicative only: 240 test photos, and the cascade operating points are read off the test set.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "edge")
from binsense import BinSense

L = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
files = [(p, i) for i, l in enumerate(L) for p in sorted(Path("data/trashnet", l).glob("*.jpg"))]
y = np.array([i for _, i in files])
M = "colab/binsense_models"; S = "colab/students"
big = BinSense(f"{M}/own_mobileclip2_s0_256_fp32.pte", f"{M}/own_mobileclip2_s0_256_pack.npz", threads=4)
gate = BinSense(f"{S}/stu_mobilenetv2_100_96_fp32.pte", f"{S}/stu_mobilenetv2_100_96.npz", threads=4)
out = {}

def probs(model, tf=None):
    P = []
    for p, _ in files:
        if tf is None: x = model.preprocess(p)
        else:
            im = tf(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0)
            im = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)); w, h = im.size; s = model.res / min(w, h)
            im = im.resize((max(model.res, round(w * s)), max(model.res, round(h * s))), Image.BILINEAR); w, h = im.size
            l, t = (w - model.res) // 2, (h - model.res) // 2; im = im.crop((l, t, l + model.res, t + model.res))
            x = ((np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0 - model.mean) / model.std)[None]
        c = model.choice_from(model.embed(x)); P.append([c["probabilities"][k] for k in L])
    return np.array(P)

# ---------- A ----------
Pb, Pg = probs(big), probs(gate)
def scores(P):
    s = np.sort(P, 1)[:, ::-1]; p1, p2 = s[:, 0], s[:, 1]
    return {"max probability (what we used)": p1, "margin p1 - p2": p1 - p2, "log ratio log(p1/p2)": np.log(p1 / np.maximum(p2, 1e-9)),
            "negative entropy": (P * np.log(np.maximum(P, 1e-12))).sum(1)}
out["A_auroc_spotting_wrong_answers"] = {}
for name, P in (("careful model", Pb), ("fast gate", Pg)):
    ok = (P.argmax(1) == y).astype(int)
    out["A_auroc_spotting_wrong_answers"][name] = {"accuracy": round(float(ok.mean()) * 100, 1), "wrong": int((1 - ok).sum()),
                                                    **{k: round(float(roc_auc_score(ok, v)), 3) for k, v in scores(P).items()}}
# cascade: largest share the gate can answer while the two-stage total stays at the careful model's own accuracy
target = int((Pb.argmax(1) == y).sum()); out["A_cascade_max_fast_share_at_full_accuracy"] = {"target_correct": target}
gp, bp = Pg.argmax(1), Pb.argmax(1)
for k, v in scores(Pg).items():
    best = 0.0
    for th in np.unique(v):
        fast = v >= th; tot = int((gp[fast] == y[fast]).sum() + (bp[~fast] == y[~fast]).sum())
        if tot >= target: best = max(best, float(fast.mean()))
    out["A_cascade_max_fast_share_at_full_accuracy"][k] = round(best * 100, 1)

# ---------- B ----------
def normalise(a):  # grey-world white balance, then stretch the 1st-99th percentile to full range
    a = a * (a.mean() / np.maximum(a.reshape(-1, 3).mean(0), 1e-6))
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return (a - lo) / max(hi - lo, 1e-6)
light = {"original": lambda a: a, "darker (x0.5)": lambda a: a * 0.5, "brighter (x1.6)": lambda a: a * 1.6,
         "gamma 2.2": lambda a: a ** 2.2, "low contrast": lambda a: 0.5 + (a - 0.5) * 0.4,
         "warm colour cast": lambda a: a * np.array([1.3, 1.0, 0.7], np.float32)}
out["B_accuracy_under_lighting_changes"] = {}
for name, f in light.items():
    raw = float((probs(big, f).argmax(1) == y).mean() * 100)
    nrm = float((probs(big, lambda a, f=f: normalise(f(a))).argmax(1) == y).mean() * 100)
    out["B_accuracy_under_lighting_changes"][name] = {"as is": round(raw, 1), "normalised first": round(nrm, 1)}
    print(name, out["B_accuracy_under_lighting_changes"][name], flush=True)
Path("bets/semicon_tricks/results.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
