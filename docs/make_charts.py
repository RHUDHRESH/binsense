#!/usr/bin/env python3
"""Draw every chart in the README from the result files in this repo (no hand-typed numbers except where noted).

  .venv/Scripts/python.exe docs/make_charts.py     ->  docs/img/*.png  +  docs/img/data_points.json
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"; OUT.mkdir(parents=True, exist_ok=True)
R = lambda tag: json.loads((ROOT / "results" / f"results_laptop_{tag}.json").read_text())

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#9ec5f4", "#eb6834"
plt.rcParams.update({"font.family": ["Segoe UI", "Arial", "sans-serif"], "font.size": 11, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE})
DATA = {}


def finish(fig, ax, title, subtitle, name):
    h = fig.get_size_inches()[1]
    fig.text(0.02, 1 - 0.16 / h, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.50 / h, subtitle, fontsize=10.5, color=INK2, va="top")
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


# ---------- 1. the accuracy journey ----------
proba = np.load(ROOT / "experiments/accuracy/cache/best_test_proba.npy")
y = np.load(ROOT / "experiments/accuracy/cache/trashnet_full.npz")["y"]
svm_acc = float((proba.argmax(1) == y).mean())
steps = [("1  Labels as text only (zero-shot)", R("fp32")["accuracy"]),
         ("2  + 5 example photos per label", R("fp32_shot5")["accuracy"]),
         ("3  + 20 example photos per label", R("fp32_shot20")["accuracy"]),
         ("4  + small fitted layer on frozen model", R("fp32_head")["accuracy"]),
         ("5  + stronger fitted layer (RBF-SVM)", svm_acc),
         ("6  Fine-tune the whole model", R("own_mobileclip2_s0_256_fp32")["accuracy"])]
DATA["accuracy_journey"] = [{"step": s, "accuracy_pct": round(a * 100, 1), "correct_of_240": round(a * 240)} for s, a in steps]
fig, ax = plt.subplots(figsize=(9.2, 4.6)); fig.subplots_adjust(left=0.40, right=0.95, top=0.80, bottom=0.12)
names = [s for s, _ in steps][::-1]; vals = [a * 100 for _, a in steps][::-1]
ax.barh(names, vals, color=[BLUE] + [BLUE_LIGHT] * 5, height=0.58)
for i, v in enumerate(vals):
    ax.text(v - 0.7, i, f"{v:.1f}%", va="center", ha="right", color="#ffffff" if i == 0 else INK, fontsize=11, fontweight="bold" if i == 0 else "normal")
ax.axvline(95, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(95, len(vals) - 0.42, "95% target", color=INK2, fontsize=9.5, ha="center", va="bottom")
ax.set_xlim(60, 102); ax.set_xlabel("Correct on 240 held-out TrashNet photos (%)"); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
finish(fig, ax, "How accuracy climbed, one fix at a time", "Same 240 test photos at every step. Model: Apple MobileCLIP2-S0 image tower.", "01_accuracy_journey.png")

# ---------- 2. what the 92.9% model got wrong ----------
rows = list(csv.DictReader((ROOT / "results/results_laptop_fp32_head.csv").open()))
conf = Counter((r["label"], r["predicted"]) for r in rows if r["correct"] != "True")
pairs = conf.most_common()
DATA["errors_at_92_9"] = [{"true": a, "predicted": b, "count": n} for (a, b), n in pairs]
fig, ax = plt.subplots(figsize=(9.2, 5.0)); fig.subplots_adjust(left=0.30, right=0.95, top=0.82, bottom=0.11)
lab = [f"{a}  read as  {b}" for (a, b), _ in pairs][::-1]; cnt = [n for _, n in pairs][::-1]
ax.barh(lab, cnt, color=[ORANGE if "metal" in l.split("read as")[1] else BLUE_LIGHT for l in lab], height=0.6)
for i, v in enumerate(cnt): ax.text(v + 0.06, i, str(v), va="center", color=INK)
ax.set_xlim(0, max(cnt) + 0.8); ax.set_xticks(range(0, max(cnt) + 1)); ax.set_xlabel("Number of photos (17 errors in total)")
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
metal = sum(n for (a, b), n in pairs if b == "metal")
ax.text(0.99, 0.04, f"Orange: wrongly called metal ({metal} of 17)", transform=ax.transAxes, ha="right", color=INK2, fontsize=10)
finish(fig, ax, "Where the 92.9% model went wrong", "Every one of its 17 mistakes, grouped by what it was and what the model said.", "02_errors.png")

# ---------- 3. confusion matrix of the final model ----------
res = json.loads((ROOT / "colab/binsense_models/results.json").read_text())["mobileclip2_s0"]
cm = np.array(res["confusion"]); L = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DATA["confusion_final_model"] = {"labels": L, "rows_true_cols_predicted": cm.tolist()}
fig, ax = plt.subplots(figsize=(6.6, 5.9)); fig.subplots_adjust(left=0.19, right=0.97, top=0.80, bottom=0.13)
ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", [SURFACE, "#cde2fb", "#2a78d6", "#0d366b"]), vmin=0, vmax=40)
for i in range(6):
    for j in range(6):
        if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", color="#ffffff" if cm[i, j] > 20 else INK, fontsize=12)
ax.set_xticks(range(6)); ax.set_xticklabels(L); ax.set_yticks(range(6)); ax.set_yticklabels(L)
ax.set_xlabel("What the model said"); ax.set_ylabel("What it really was"); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, 6), minor=True); ax.set_yticks(np.arange(-.5, 6), minor=True); ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="minor", length=0)
finish(fig, ax, "Final model: 230 of 240 correct", "Fine-tuned MobileCLIP2-S0. Diagonal = right. 40 photos per row.", "03_confusion.png")

# ---------- 4. speed against accuracy ----------
pts = [("MobileNetV2 half-width, 96 px", R("stu_mobilenetv2_050_96")), ("MobileNetV2 half-width, 160 px", R("stu_mobilenetv2_050_160")),
       ("MobileNetV2, 96 px", R("stu_mobilenetv2_100_96")), ("MobileNetV2, 224 px", R("own_mobilenet_v2_224_fp32")),
       ("MobileCLIP2-S0, 256 px", R("own_mobileclip2_s0_256_fp32"))]
DATA["speed_vs_accuracy_laptop"] = [{"model": n, "inference_ms_p50": r["inference_ms"]["p50"], "accuracy_pct": round(r["accuracy"] * 100, 1),
                                     "pte_mb": r["pte_size_mb"]} for n, r in pts]
fig, ax = plt.subplots(figsize=(9.2, 5.0)); fig.subplots_adjust(left=0.09, right=0.96, top=0.80, bottom=0.13)
xs = [r["inference_ms"]["p50"] for _, r in pts]; ys = [r["accuracy"] * 100 for _, r in pts]
ax.scatter(xs, ys, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
offs = {0: (10, -4), 1: (10, -14), 2: (10, 6), 3: (10, -4), 4: (-12, -30)}
for i, (n, r) in enumerate(pts):
    ax.annotate(f"{n}\n{ys[i]:.1f}% at {xs[i]:g} ms", (xs[i], ys[i]), xytext=offs[i], textcoords="offset points", fontsize=9.5, color=INK,
                ha="right" if offs[i][0] < 0 else "left", va="center")
ax.axhline(95, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(8, 95.3, "95% target", color=INK2, fontsize=9.5)
ax.set_xscale("log"); ax.set_xlim(0.6, 130); ax.set_ylim(80, 98); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_xlabel("Time per image on the laptop, milliseconds (log scale; left is faster)"); ax.set_ylabel("Accuracy (%)")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
finish(fig, ax, "Faster models are less accurate", "Each dot is one model we trained. Laptop CPU, 4 threads. Not yet measured on the Raspberry Pi.", "04_speed_vs_accuracy.png")

# ---------- 5. two-stage: share on the fast path ----------
cas = [("Half-width 96 px gate", "stu_mobilenetv2_050_96"), ("Half-width 160 px gate", "stu_mobilenetv2_050_160"), ("Full 96 px gate", "stu_mobilenetv2_100_96")]
DATA["two_stage"] = []
fig, ax = plt.subplots(figsize=(9.2, 4.2)); fig.subplots_adjust(left=0.32, right=0.97, top=0.78, bottom=0.14)
ylab, fast, acc = [], [], []
for n, t in cas:
    for th in ("0.8", "0.9"):
        r = R(f"cascade_{t}_{th}"); ylab.append(f"{n}, sure ≥ {int(float(th) * 100)}%"); fast.append(r["two_stage"]["fast_path_share"] * 100); acc.append(r["accuracy"] * 100)
        DATA["two_stage"].append({"gate": n, "threshold": float(th), "fast_path_share_pct": round(fast[-1], 1), "overall_accuracy_pct": round(acc[-1], 1),
                                  "fast_path_accuracy_pct": round(r["two_stage"]["fast_path_accuracy"] * 100, 1)})
ylab, fast, acc = ylab[::-1], fast[::-1], acc[::-1]
ax.barh(ylab, fast, color=BLUE, height=0.58)
for i, v in enumerate(fast): ax.text(v + 1, i, f"{v:.0f}% fast   →   {acc[i]:.1f}% overall accuracy", va="center", color=INK, fontsize=10)
ax.set_xlim(0, 165); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Share of photos answered by the fast model alone (%)")
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
finish(fig, ax, "Two-stage: how often the fast model is enough", "Fast MobileNetV2 answers when sure; the careful MobileCLIP takes the rest.", "05_two_stage.png")

# ---------- 6. abstaining ----------
confd = proba.max(1); pred = proba.argmax(1); ths = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
cov = [float((confd >= t).mean() * 100) for t in ths]; ac = [float((pred[confd >= t] == y[confd >= t]).mean() * 100) for t in ths]
DATA["abstain"] = [{"min_confidence": t, "answered_pct": round(c, 1), "accuracy_on_answered_pct": round(a, 1)} for t, c, a in zip(ths, cov, ac)]
fig, ax = plt.subplots(figsize=(9.2, 4.4)); fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.15)
ax.plot(cov, ac, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
for t, c, a in zip(ths, cov, ac):
    ax.annotate(("answers all" if t == 0 else f"needs {int(t * 100)}%") + f"\n{a:.1f}%", (c, a), xytext=(8, -20), textcoords="offset points", ha="left", fontsize=9.5, color=INK)
ax.set_xlim(102, 71); ax.set_ylim(93, 101); ax.set_xlabel("Share of photos it still answers (%)  →  more cautious"); ax.set_ylabel("Accuracy on answered (%)")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
finish(fig, ax, "Saying \"not sure\" buys accuracy", "Label = how sure the model must be before it answers. Skipped photos go to a person.", "06_abstain.png")

# ---------- 7. why the delegate matters ----------
# 8862 ms: portable-kernel export of the same 192 px tower, timed once on this laptop by laptop/export_portable.py (printed, not stored in a json).
port, xnn = 8862.0, R("fp32_192_head")["inference_ms"]["p50"]
DATA["delegate"] = {"portable_ms": port, "xnnpack_ms": xnn, "speedup_x": round(port / xnn)}
fig, ax = plt.subplots(figsize=(9.2, 3.1)); fig.subplots_adjust(left=0.30, right=0.93, top=0.70, bottom=0.21)
ax.barh(["With XNNPACK delegate", "Without (portable kernels)"], [xnn, port], color=[BLUE, BLUE_LIGHT], height=0.55)
ax.text(xnn + 120, 0, f"{xnn:.0f} ms", va="center", color=INK, fontweight="bold"); ax.text(port - 120, 1, f"{port:,.0f} ms", va="center", ha="right", color=INK)
ax.set_xlabel("Time per image on the laptop (ms)"); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
finish(fig, ax, f"Same model, about {round(port / xnn)}x faster with the delegate", "MobileCLIP2-S0 image tower at 192 px, exported two ways.", "07_delegate.png")

# ---------- 8. INT8 ----------
f32, i8 = R("own_mobilenet_v2_224_fp32"), R("own_mobilenet_v2_224_int8")
DATA["int8_mobilenetv2"] = {"fp32": {"mb": f32["pte_size_mb"], "accuracy_pct": round(f32["accuracy"] * 100, 1)}, "int8": {"mb": i8["pte_size_mb"], "accuracy_pct": round(i8["accuracy"] * 100, 1)}}
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4)); fig.subplots_adjust(left=0.13, right=0.96, top=0.70, bottom=0.2, wspace=0.55)
for ax, key, unit, lim, lbl in ((axs[0], "pte_size_mb", " MB", 11, "File size (MB)"), (axs[1], "accuracy", "%", 100, "Accuracy (%)")):
    v = [f32[key], i8[key]] if key != "accuracy" else [f32[key] * 100, i8[key] * 100]
    ax.barh(["32-bit (FP32)", "8-bit (INT8)"], v, color=[BLUE_LIGHT, BLUE], height=0.55)
    for i, x in enumerate(v): ax.text(x + lim * 0.015, i, f"{x:.1f}{unit}", va="center", color=INK)
    ax.set_xlim(0 if key != "accuracy" else 80, lim * 1.12 if key != "accuracy" else 100); ax.set_xlabel(lbl); ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
finish(fig, axs[0], "8-bit numbers: much smaller, a little less accurate", "Our fine-tuned MobileNetV2. (8-bit broke MobileCLIP, so that one stays 32-bit.)", "08_int8.png")

(OUT / "data_points.json").write_text(json.dumps(DATA, indent=1))
print(json.dumps({k: v for k, v in DATA.items() if k in ("accuracy_journey", "delegate", "int8_mobilenetv2", "abstain")}, indent=0)[:1500])
