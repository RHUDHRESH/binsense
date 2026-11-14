Copy this whole folder to the Pi (e.g. scp -r pi_bundle pi@<ip>:~/binsense), then on the Pi:
  cd ~/binsense
  bash setup_pi.sh                      # needs internet once (about 10 min)
  source ~/binsense-venv/bin/activate
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

Fastest two-stage pairing found on the laptop (96 px MobileNetV2 student as the gate):
  python run_eval.py --pte models/own_mobileclip2_s0_256_fp32.pte --pack packs/own_mobileclip2_s0_256.npz \
      --gate-pte models/stu_mobilenetv2_100_96_fp32.pte --gate-pack packs/stu_mobilenetv2_100_96.npz --threshold 0.9 --images images

Terminal proof of concept (offline check, model list, classification loop, hardware line, summary):
  python demo.py                      # all sample images once
  python demo.py --loop 5             # five passes
  python demo.py --watch inbox        # classify photos as they are dropped into inbox/

Trace ONE photo step by step (input, preprocessing, each model, raw scores, probabilities, decision):
  python trace.py images/plastic/plastic11.jpg
  python trace.py images/trash/trash1.jpg --both      # also run the careful model when the fast one is sure

Live webcam (USB camera plugged into the Pi; prints in the terminal, no window):
  python live.py                 # classify continuously, announces a decision after 3 matching frames
  python live.py --snap          # press Enter to capture one frame
  python live.py --camera 1      # if camera 0 is not the webcam

Web UI (no terminal needed):
  Windows: double-click start_ui.bat          Pi: bash start_ui.sh, then open http://<pi-address>:8000
  Step 1 choose or drop photos, Step 2 pick or type labels, Step 3 pick a model and move the sliders, Step 4 click a photo or Run all.
