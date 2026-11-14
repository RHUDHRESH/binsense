# EXPERIMENTS — foundation-backbone bet (append-only)

## F1 — bigger foundation backbones vs MobileCLIP2-S0 (registered 2026-11-14, before any run)

- **Hypothesis:** a larger SSL / image-text backbone (DINOv2 ViT-S/14, MobileCLIP-S1, MobileCLIP2-S2, ViT-B/32 laion2b)
  gives a decisive (> ~3 pp, paired-significant) held-out accuracy win over MobileCLIP2-S0 for the 6-class TrashNet task,
  at an FP32 XNNPACK latency still usable on a Pi 4.
- **Decision informed:** which image tower ships to the Pi. Wrong-way cost: a slower tower that is not really better
  (latency, battery, memory) vs leaving accuracy on the table.
- **Metrics:** held-out top-1 accuracy on balanced 240-image test set (40/class) -> aligns with per-item bin decision
  (classes balanced in test, so accuracy == macro recall). Zero-shot, 20-shot nearest prototype (mean +/- sd over 50
  support draws), LogisticRegression(C=10, max_iter=2000) on all L2-normalised pool features. Wilson 95% CI;
  paired exact McNemar vs S0 on the same 240 images.
- **Baseline:** MobileCLIP2-S0 256 px (brief: ZS 72.1 / 20-shot 84.6 / linear 92.9). Re-run through this exact
  protocol as `s0_ref` so the comparison is like-for-like. Dumb baseline: majority class = 16.7% on balanced test.
- **Data / split:** fit on data/trashnet_pool (2287), test data/trashnet (240), fixed every-Nth split by
  laptop/fetch_trashnet.py. Not group-aware: TrashNet has repeated objects -> near-duplicate check (max cosine test->pool)
  and exact md5 check are part of the eval.
- **Preprocess:** identical to pi_bundle/binsense.py (shortest-side bilinear resize + centre crop, per-model mean/std/res).
- **Seed:** 0 (prototype draws seeds 0..49). **Config/script:** experiments/foundation/bet.py.
  **Interpreter:** <project>/.venv/Scripts/python.exe (torch 2.12.1+cpu, executorch 1.3.1).
- **Caveat registered up-front:** laptop latency is measured while 3 other agents share the CPU; ratios vs s0_ref
  measured in the same window are more trustworthy than absolute ms. No Pi run in this bet.
- **Result:** (appended below after the run)

### F1 result (appended 2026-11-14 ~13:50)

Protocol deviations, decided before seeing the result: pool capped at 100/class (seed 0, 597 images) for every model,
because the CPU was pegged at 100% by other agents (S0 ran at about 140 ms/img instead of about 60). S0 on the full
2287-image pool is also reported. Bug found and fixed during the run: the preprocess output was a non-C-contiguous numpy
array, and the ExecuTorch pybinding ignores strides, so pixels came out scrambled (pte-vs-eager cosine was 0.12-0.36 on
real images and 1.0 on torch.rand). Fixed with np.ascontiguousarray. Every number below is from after the fix (cosine 1.0).

| model | params M | pte MB | deleg/non-deleg | ms pte (same window) | ZS | 20-shot (50 draws) | linear 597 [95% CI] |
|---|---|---|---|---|---|---|---|
| S0 ref 256 | 11.4 | 45.5 | (spec 325/51) | 118.7 | 61.3 | 80.5 +/- 1.4 | 93.3 [89.4, 95.9]; full pool 92.9 |
| DINOv2 S/14 224 | 21.6 | 86.7 | 391/342, 62 subgraphs | 209.2 | n/a | 78.4 +/- 1.7 | 90.0 [85.6, 93.2]; McNemar vs S0 7:15, p=0.13 |
| MobileCLIP2-S2 256 | 35.7 | 143.0 | 647/93 | 435.3 | 64.2 | 83.6 +/- 1.7 | 93.3 [89.4, 95.9]; McNemar 6:6, p=1.0 |
| ViT-B/32 224 | 87.9 | 351.6 | 451/489, 63 subgraphs | 2660 (eager 108) | not run | not run | not run (36 addmm + 26 LayerNorm not delegated) |
| MobileCLIP-S1, DINOv2-reg4 | - | - | - | - | not run (time) | | |

Leakage: no exact md5 duplicates between test and pool. Near-duplicates (max cosine >= 0.95): 3 (S0), 13 (S2), 0 (DINOv2).
Removing them changes linear accuracy by 0.3 pp or less. Verdict: **no decisive win; hypothesis rejected for linear-head
accuracy.** S2 matches S0 on the linear head, is +3 pp on prototypes and zero-shot (within test-set noise), and is 3.7x
slower and 3x larger. Verifier: none (self-run).
