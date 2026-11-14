# Edge-Jev: offline waste decisions on a Raspberry Pi with ExecuTorch

Namma AI Hackathon 2026, Problem Statement 3 (AI for Sustainability: edge-based waste sorting and food categorisation).

Edge-Jev is a small decision engine that looks at a photo of a discarded item and answers typed questions about it,
fully offline, through ExecuTorch with the XNNPACK delegate:

- **which label?** one category with a probability per label
- **multi-select:** every label and tag that applies, for example `paper + recyclable + dry_waste`
- **yes / no:** for example "is it recyclable?" (text-label model)
- **score:** a 1-5 contamination level (text-label model)

The idea is borrowed from cloud "System One" decision models (typed answers with probabilities instead of prose),
rebuilt so it runs with no network on a Pi.

## Status, stated plainly

| Item | State |
|---|---|
| Export to ExecuTorch (.pte), XNNPACK delegation | Done, verified against the original model (cosine 1.0) |
| Accuracy on 240 held-out TrashNet photos | Measured on a laptop through the same engine the Pi uses |
| Multi-select labels, abstain, audit log, boot service | Done, smoke-tested on a laptop |
| **Run on the Raspberry Pi 4** | **Not done yet.** All timings below are laptop timings. |
| Organiser images and labels | Not received at the time of writing |

## Results (laptop, x86, 4 threads, 240 held-out TrashNet photos, 6 classes)

| Model | Accuracy | Inference / image | .pte size |
|---|---|---|---|
| MobileCLIP2-S0 image tower, frozen, text labels only (zero-shot) | 72.1% | ~60 ms | 45.5 MB |
| + 20 example photos per label (prototypes) | 84.6% | ~60 ms | 45.5 MB |
| + small fitted layer on frozen features | 92.9% | ~60 ms | 45.5 MB |
| Fine-tuned MobileNetV2, FP32 | 93.8% | 3.9 ms | 8.9 MB |
| Fine-tuned MobileNetV2, INT8 (PT2E static, per-channel) | 88.8% | 4.7 ms | 2.5 MB |
| Fine-tuned MobileCLIP2-S0 image tower, FP32 | **95.8%** | 59 ms | 45.5 MB |
| Two-stage: MobileNetV2 gate, MobileCLIP only when unsure (threshold 0.9) | 95.8% | 64% of items take the ~4 ms path | both |
| Same model without XNNPACK (portable kernels) | same | ~8.9 s | 45.6 MB |

Other measurements:

- Abstaining below 0.7 confidence (frozen model, RBF-SVM head): answers 92% of items at 98.6% accuracy.
- Scored as recyclable vs non-recyclable: 97.5%.
- XNNPACK delegation of the MobileCLIP tower: 325 nodes delegated, 51 left on portable kernels.

### Caveats

- **The test split is optimistic.** TrashNet contains near-identical neighbouring shots, and the split puts some on
  both sides. Removing those neighbours from training cost one model about 6 points. Expect lower accuracy on new photos.
- 240 images is a small test: one image is 0.4 points.
- The two-stage threshold was chosen by looking at the test set.
- Full INT8 breaks the MobileCLIP tower (output no longer matches the float model), so it ships as FP32.
- A Pi 4 (Cortex-A72) has no dot-product or SME instructions, so KleidiAI kernels do not engage; INT8 gains are smaller than on a Pi 5.

## What failed (kept on purpose)

- Full and mixed-precision INT8 of MobileCLIP2-S0: broken output.
- Six frozen ImageNet CNNs with a linear head: best 89.2%, none beat the MobileCLIP features.
- Bigger backbones (MobileCLIP2-S2, DINOv2 ViT-S): no accuracy gain, 2-4x slower.
- Training on the laptop CPU while other jobs ran: one step took 59 s; moved to a cloud GPU.

## Run it on a Raspberry Pi 4 (64-bit Pi OS)

Copy `pi_bundle/` to the Pi, then:

```bash
bash setup_pi.sh                 # one-time, needs internet
source ~/jev-venv/bin/activate
bash tune_pi.sh                  # performance governor, quiet services
python run_eval.py --pte models/own_mobileclip2_s0_256_fp32.pte --pack packs/own_mobileclip2_s0_256.npz --images images
bash bench_pi.sh                 # every model x 1-4 threads + thermal log -> results/SUMMARY.md
```

`run_eval.py` prints the mandatory deliverables: a prediction per image, the results table against the supplied labels,
total correct, accuracy, model, inference time, model size and peak memory. See `pi_bundle/README.txt` for the two-stage
and baseline commands.

### New labels from the organisers

1. Put the label names and a few descriptive phrases in a JSON file like `labels/trashnet.json`.
2. `python src/build_labels.py --labels <file> --spec artifacts/mobileclip2-s0_256_spec.json --out pi_bundle/packs/<name>_256.npz`
3. Optional: `src/fit_head.py` fits a small layer from public photos mapped to those labels (`bets/data/category_map.json`).
4. Optional: `src/add_groups.py` adds multi-select tags such as recyclable / wet / dry / hazardous.

No model retraining and no re-export is needed for steps 1-2.

## Layout

| Path | What |
|---|---|
| `edge/` | Everything that runs on the Pi: engine, evaluation, service, setup, tuning, benchmark |
| `src/` | Development-machine scripts: export, label packs, head fitting, data fetch |
| `colab/` | GPU notebooks (fine-tuning, distillation) and the trained models |
| `pi_bundle/` | Ready-to-copy folder for the Pi (code + models + packs + sample images) |
| `bets/` | Parallel experiments with their own ledgers: CNN backbones, foundation backbones, datasets, own model, accuracy |
| `results/` | Result tables (Markdown, CSV, JSON) and the failure contact sheet |
| `docs/` | Two-minute pitch and a plain-language explainer |

## Models and licences

- MobileCLIP2-S0 (`dfndr2b`, via open_clip): Apple's model terms apply; check them before any use beyond this demo.
- MobileNetV2: torchvision ImageNet weights (BSD-3).
- TrashNet: MIT. Other datasets and their licences are listed in `bets/data/DATASETS.md`.
- ExecuTorch 1.3.1, PyTorch 2.12.1. Export flow adapted from Arm's course code (`ArmDeveloperEcosystem/courses-code-executorch`).
