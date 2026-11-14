# What we built and why it works: a plain-language walkthrough

Read this top to bottom once; each section is one idea you can explain to a judge.

## 1. The pipeline in one picture

```
laptop / cloud GPU (ahead of time)                     Raspberry Pi (at run time)
---------------------------------                      --------------------------
PyTorch model                                          photo
  -> torch.export        (freeze the graph)              -> resize + scale        (preprocess)
  -> lower to XNNPACK    (hand layers to fast kernels)   -> .pte forward pass     (ExecuTorch)
  -> .pte file           (one portable file)             -> embedding (512 numbers)
labels.json                                              -> compare with label pack
  -> text model -> label pack (.npz, a few KB)           -> typed answer + probabilities
```

The Pi never runs PyTorch training code or a text model. It runs one `.pte` file and does a few dot products.

## 2. Embeddings: the whole trick

The image model turns a photo into 512 numbers (an embedding). Similar-looking things get similar numbers.
Classification is then just "which label's numbers is this photo closest to?".

- **Zero-shot:** the label's numbers come from text ("a photo of a plastic bottle"). No photos needed. We measured 72%.
- **Prototypes:** average the embeddings of 20 example photos per label. 85%.
- **Fitted layer:** learn one small matrix on top of frozen embeddings. 93%.
- **Fine-tuning:** let the whole model adjust to waste photos. 96%.

Each step trades flexibility for accuracy. Zero-shot handles any label instantly; fine-tuning is best on known labels.

## 3. Why ExecuTorch needs a delegate

ExecuTorch on its own uses "portable kernels": correct on any chip, but slow. We measured about 8.9 seconds per image.
Lowering with the XNNPACK partitioner hands supported layers to hand-tuned Arm Neon code: about 60 ms on the laptop.
The export report says how many nodes were delegated (325) and how many fell back (51). A judge from Arm will ask this.

## 4. Quantization (INT8) and why it bit us

INT8 stores weights as 8-bit integers instead of 32-bit floats: about 4x smaller and often faster.
It needs "calibration": run ~100 real photos through so it learns each layer's value range.

- MobileNetV2 (plain convolutions): fine. 8.9 MB to 2.5 MB, about 5 points of accuracy lost.
- MobileCLIP (convolutions plus attention): broken. The output no longer resembled the float model.

Lesson: always compare the quantized model's output to the float model's on a real image before trusting it.

## 5. The two-stage path (fast then careful)

A tiny model answers first. If it is at least 90% sure, we stop (about 4 ms). If not, the big model decides (about 60 ms).
On our test, 64% of items took the fast path and accuracy matched the big model alone. Average time drops a lot
because most items are easy.

## 6. Distillation

A big accurate "teacher" labels photos with its full probability spread ("80% plastic, 15% metal"), and a small
"student" learns to copy that spread, not just the single right answer. The student ends up better than if it had
trained on plain labels. Our first student (MobileNetV2 at 160 pixels) matched the 224-pixel model at 93.8%.

## 7. Multi-select and abstaining

- **Multi-select:** report every label above a probability floor, plus group tags (recyclable = sum of the
  probabilities of its member labels). A carton can be "paper + cardboard".
- **Abstain:** wrong answers had average confidence 0.56; right answers 0.85. So low confidence is a usable warning.
  Skipping items below 0.7 answered 92% of items at 98.6% accuracy.

## 8. What "fast on a Pi 4" depends on

- Threads: the Pi has 4 cores; small models can be faster on fewer threads. Sweep 1-4.
- Image size: cost grows with the square of the side length (160 px is about half the work of 224 px).
- CPU governor: `performance` holds the clock high.
- Heat: the Pi slows itself above 80 C. Log temperature during every benchmark.
- Preprocessing: decoding and resizing the JPEG can cost as much as a small model. Time it separately.

## 9. Mistakes worth remembering

1. **Leaky test split.** Near-duplicate photos on both sides of the split made every number look better than it is.
   Split by group (or by shooting session), not by every N-th file.
2. **Too many jobs on one laptop.** Four agents shared one CPU and a training step took 59 s. Run CPU jobs one at a time.
3. **A wrong conclusion, corrected.** One test suggested INT8 files misbehave on Windows; a second test did not
   reproduce it. One measurement is not a finding.
4. **Disk nearly full.** Large model files from rejected experiments almost blocked a download. Clean up losers early.
5. **The Pi came last.** Everything is measured on a laptop. The one number that counts is still missing.

## 10. Words to use correctly

export, lowering, delegate, partitioner, portable kernels, fallback operators, `.pte` artefact, ahead-of-time (AOT),
PT2E quantization, calibration, embedding, zero-shot, distillation, p50 / p90 latency, peak RSS.
