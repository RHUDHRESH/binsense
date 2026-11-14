#!/usr/bin/env python3
"""One program for the judges: runs the whole BinSense proof in order. Press Enter to move to the next part.

  python show.py            # all four parts
  python show.py 3          # jump straight to one part (1-4)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PARTS = [
    ("1/4  OFFLINE CHECK + 30 TEST PHOTOS", "Checks the internet is unreachable, loads both models, classifies 30 photos, prints accuracy and speed.",
     ["demo.py"]),
    ("2/4  ONE PHOTO, STEP BY STEP", "Shows the input, the tensor, each model's raw scores and probabilities, and the decision.",
     ["trace.py", "images/plastic/plastic11.jpg"]),
    ("3/4  A HARD PHOTO: FAST MODEL HANDS OVER", "The fast model is not sure enough, so the careful model decides.",
     ["trace.py", "images/trash/trash1.jpg"]),
    ("4/4  LIVE WEBCAM", "Hold an item in front of the camera. Ctrl+C to stop.",
     ["live.py"] + sys.argv[2:]),
]


def main():
    only = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    for i, (title, what, cmd) in enumerate(PARTS, 1):
        if only and i != only:
            continue
        print("\n" + "=" * 74 + f"\n  {title}\n  {what}\n" + "=" * 74)
        if not only:
            try:
                input("  press Enter to run (Ctrl+C to quit) ")
            except (KeyboardInterrupt, EOFError):
                print(); return
        try:
            subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], cwd=HERE)
        except KeyboardInterrupt:
            pass
    print("\nDone.")


if __name__ == "__main__":
    main()
