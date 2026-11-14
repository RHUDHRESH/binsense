# experiments/own_model — experiment ledger (append-only)

Registered 2026-11-14 BEFORE any run. Results appended below after running.

| id | hypothesis | metric (why) | baseline | data / split | seed | config | interpreter | result | verdict |
|---|---|---|---|---|---|---|---|---|---|
| OM-0 | Leakage audit: no test file shares a name or byte-hash with the pool; near-duplicates (dHash<=6) are rare | count of exact/near dups | n/a | data/trashnet (240) vs data/trashnet_pool (2287) | n/a | experiments/own_model/leak_check.py | .venv py -X utf8 | (pending) | |
| OM-B | Frozen ImageNet MobileNetV2 features + logistic regression (linear probe) is the honest baseline; majority-class = 16.7% on the balanced test | test top-1 accuracy (test set is class-balanced 40/class, so accuracy == balanced accuracy; the deployed decision is a single argmax label) | majority 16.7% | train = pool minus 15% stratified val; test = data/trashnet | 0 | experiments/own_model/train.py (probe stage) | .venv py -X utf8 | (pending) | |
| OM-1 | Fine-tuned MobileNetV2 (ImageNet init, half-frozen warmup, AdamW+cosine, LS 0.1, class-weighted CE) beats the linear probe by >=5 pts on test; INT8 PTQ (per-channel symmetric, FP32 classifier) loses <=2 pts | same | OM-B | same; val (15% of pool) for early stopping only | 0 | experiments/own_model/train.py --arch mobilenet_v2 | .venv py -X utf8 | (pending) | |
| OM-2 | Same recipe on mobilenet_v3_small: smaller/faster, within 3 pts of v2 (only if time allows) | same | OM-B | same | 0 | experiments/own_model/train.py --arch mobilenet_v3_small | .venv py -X utf8 | (pending) | |

## Run notes (appended)
- OM-0 result: 0 shared file names, 0 shared SHA-256 between test (240) and pool (2287); 3 byte-identical duplicates inside the pool (dropped before split). 9x8 dHash is NOT discriminative on TrashNet (white backgrounds: cross-class pairs at Hamming 1), so near-dup audit moved to ImageNet-feature cosine (computed in the probe stage).
- OM-1 attempt 1 FAILED/KILLED (~10 min wasted): DataLoader workers (spawn on Windows) starved with host CPU at 100% (shared with 3 agents); main proc idle at 187 CPU-s, probe never finished, and probe time was counted against the training budget. Fix: workers=0, probe moved after training, budget clock starts at training. Log: train_mobilenet_v2_attempt1_killed.log
- OM-1 attempt 2 KILLED at the 35-min time box: host CPU 99.9%, processor queue length 91 (other agents' bench_cnn.py x2, leak_probe.py, export_tower.py, exp.py all Normal priority). Measured in-run: MobileNetV2 train step (bs 32, 224px, 4 threads) = 59.2 s -> planned 3 epochs = ~3.5 h. Standalone probe right after: 1 thread 12.6 s/step, 2 threads 6.1 s/step (4-thread OpenMP collapses under oversubscription). No checkpoint, no test number. Log: train_mobilenet_v2.log
- OM-B, OM-1, OM-2: NOT RUN TO COMPLETION -> no results. export_eval.py (FP32/INT8 .pte + PT2E-converted INT8 eval, per coordinator correction) written but never executed.
