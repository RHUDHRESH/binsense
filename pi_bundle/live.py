#!/usr/bin/env python3
"""Live webcam proof of concept: hold an item in front of the camera, BinSense names it in the terminal.

  python live.py                 # camera 0, classify continuously
  python live.py --camera 1      # pick another camera
  python live.py --snap          # press Enter to capture one frame and classify it
Needs opencv (pip install opencv-python-headless). No window is opened; everything prints in the terminal.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from binsense import BinSense, Cascade  # noqa: E402

TTY = sys.stdout.isatty()
G, Y, B, X = [(c if TTY else "") for c in ("\033[32m", "\033[33m", "\033[1m", "\033[0m")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pte", default="models/own_mobileclip2_s0_256_fp32.pte")
    ap.add_argument("--pack", default="packs/own_mobileclip2_s0_256.npz")
    ap.add_argument("--gate-pte", default="models/stu_mobilenetv2_100_96_fp32.pte")
    ap.add_argument("--gate-pack", default="packs/stu_mobilenetv2_100_96.npz")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--review", type=float, default=0.7)
    ap.add_argument("--every", type=float, default=0.3, help="seconds between frames")
    ap.add_argument("--stable", type=int, default=3, help="same answer this many frames in a row = a decision")
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = until Ctrl+C)")
    ap.add_argument("--save", default="live_last.jpg", help="keep the latest frame here so it can be inspected")
    args = ap.parse_args()
    base = Path(__file__).parent

    import cv2
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        sys.exit(f"camera {args.camera} did not open. Try --camera 1, or check `ls /dev/video*` on the Pi.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        cam.read()                                   # let exposure settle
    ok, f = cam.read()
    print(f"{B}camera {args.camera}{X} open: {f.shape[1]}x{f.shape[0]} frames" if ok else "camera opened but gave no frame")

    model = BinSense(base / args.pte, base / args.pack)
    if (base / args.gate_pte).exists():
        model = Cascade(BinSense(base / args.gate_pte, base / args.gate_pack), model, args.threshold)
    tmp = Path(tempfile.gettempdir()) / "binsense_frame.jpg"
    if Path("/dev/shm").exists():
        tmp = Path("/dev/shm/binsense_frame.jpg")      # RAM disk on the Pi: no SD card wear
    recent, n, times, last_decision = deque(maxlen=args.stable), 0, [], None
    print("Hold an item in front of the camera. Ctrl+C to stop.\n")
    try:
        while not args.frames or n < args.frames:
            if args.snap:
                input("press Enter to capture... ")
                for _ in range(3):
                    cam.read()                       # drop buffered frames
            ok, frame = cam.read()
            if not ok:
                print("no frame from camera"); time.sleep(0.5); continue
            cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if args.save:
                cv2.imwrite(str(base / args.save), frame)
            try:
                d = model.decide(tmp)
            except Exception as e:  # noqa: BLE001  a bad frame must not stop the demo
                print(f"       frame skipped ({type(e).__name__})"); time.sleep(0.3); n += 1; continue
            c = d["choice"]; n += 1
            ms = d["ms"]["inference"]; times.append(ms)
            top3 = sorted(c["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            shown = "REVIEW" if c["confidence"] < args.review else c["answer"]
            colour = Y if shown == "REVIEW" else G
            line = (f"{n:5d}  {colour}{shown:<10}{X} {c['confidence'] * 100:5.1f}%  {d.get('path', 'single'):<7} {ms:6.1f} ms   "
                    + "  ".join(f"{k} {v * 100:.0f}%" for k, v in top3))
            print(line)
            recent.append(shown)
            if len(recent) == args.stable and len(set(recent)) == 1 and shown != last_decision:
                last_decision = shown
                tags = " + ".join(d["multi"]["labels"] + d["multi"]["tags"])
                print(f"       {B}>>> DECISION: {shown}{X}   ({tags})" if shown != "REVIEW" else f"       {B}>>> NOT SURE: ask a person{X}")
            if not args.snap:
                time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
    if times:
        s = sorted(times)
        print(f"\n{n} frames   inference median {s[len(s) // 2]:.1f} ms, max {s[-1]:.1f} ms")
        if isinstance(model, Cascade):
            print(f"fast model answered {model.fast} of {model.fast + model.slow}")


if __name__ == "__main__":
    main()
