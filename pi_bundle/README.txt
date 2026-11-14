Copy this whole folder to the Pi (e.g. scp -r pi_bundle pi@<ip>:~/jev), then on the Pi:
  cd ~/jev
  bash setup_pi.sh                      # needs internet once (about 10 min)
  source ~/jev-venv/bin/activate
  bash tune_pi.sh                       # performance governor, quiet services

Single model (zero-shot + head, any label set):
  python run_eval.py --pte models/mobileclip2-s0_256_fp32.pte --pack packs/trashnet_256.npz --images images

Our fine-tuned model (best accuracy on the laptop: 230/240):
  python run_eval.py --pte models/own_mobileclip2_s0_256_fp32.pte --pack packs/own_mobileclip2_s0_256.npz --images images

Two-stage (fast MobileNetV2 gate, careful MobileCLIP only when unsure):
  python run_eval.py --pte models/own_mobileclip2_s0_256_fp32.pte --pack packs/own_mobileclip2_s0_256.npz \
      --gate-pte models/own_mobilenet_v2_224_fp32.pte --gate-pack packs/own_mobilenet_v2_224.npz --threshold 0.9 --images images

Full sweep (every model x 1-4 threads + thermal log -> results/SUMMARY.md):
  bash bench_pi.sh

Baselines for the comparison table:
  python eager_baseline.py MobileCLIP2-S0 dfndr2b 192      # needs: pip install open_clip_torch timm
  models/mobileclip2-s0_192_portable.pte is the no-XNNPACK row (very slow; run with --repeat 1 on 3 images)
