"""Generates colab/students_a100.ipynb: teacher + ten distilled students in one run on a high-RAM A100 runtime."""
import ast, json, re
from pathlib import Path
here = Path(__file__).parent
nb1 = json.loads((here / "train_waste_models.ipynb").read_text())
src = [c["source"] for c in nb1["cells"] if c["cell_type"] == "code"]
install, data, train, export = src[0], src[1], src[2], src[3]
# teacher only (MobileNetV2 is already trained and downloaded); more loader workers for the bigger machine
train = train.replace('nets["mobilenet_v2"], results["mobilenet_v2"] = run("mobilenet_v2", epochs=25, lr_bb=1e-3, lr_head=5e-3)\n', "")
train = train.replace("num_workers=2", "num_workers=10").replace("bs=48", "bs=96")
train += '\nfor n in nets: nets[n].backbone = V(nets[n].backbone)   # attribute-named wrapper keeps exported parameter names valid'
helpers = export.split("for name, net in nets.items():")[0]
students = (here / "students.py").read_text().replace("num_workers=2", "num_workers=10").replace("bs=64", "bs=128")
cells = [{"cell_type": "markdown", "metadata": {}, "source": "# BinSense: teacher + distilled students (A100)"}]
for s in (install, data, train, helpers, students):
    ast.parse("\n".join(l for l in s.split("\n") if not l.startswith("!")))
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
(here / "students_a100.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote students_a100.ipynb", [len(c["source"]) for c in cells])
