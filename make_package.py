# ------------------ Inlämningspaket -----------------
# Bygger GroupXX_ECG_Project.zip bredvid repot med den struktur som uppgiften kräver.
# python make_package.py Group07   (gruppnamnet blir början på mappen och zip-filen)

import glob
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
GROUP = sys.argv[1] if len(sys.argv) > 1 else "GroupXX"
NAME = f"{GROUP}_ECG_Project"
ZIP_FILE = os.path.abspath(os.path.join(ROOT, "..", f"{NAME}.zip"))

FILES = ["report.pdf", "README.md", "requirements.txt", "notebooks/EDA.ipynb"]
FOLDERS = ["src", "configs", "results", "max78002"]

with open(os.path.join(ROOT, "configs", "final_model.json"), encoding="utf-8") as f:
    config = json.load(f)

# Bara de slutliga modellerna: tre seeds i FP32 och INT8
filters = "-".join(map(str, config["filters"]))
model_files = []
for seed in config["seeds"]:
    model_files += glob.glob(os.path.join(ROOT, "models", f"SmallCNN-valF1-*-seed{seed}-lr{config['lr']}-cw0-sampler1-rr1-filters{filters}.pth"))
    model_files += glob.glob(os.path.join(ROOT, "models", f"SmallCNN-INT8-valF1-*-seed{seed}-filters{filters}.pth"))
assert len(model_files) == 2 * len(config["seeds"]), f"hittade {len(model_files)} modeller"

paths = [os.path.join(ROOT, f) for f in FILES if os.path.exists(os.path.join(ROOT, f))]
for folder in FOLDERS:
    for current, dirs, files in os.walk(os.path.join(ROOT, folder)):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        paths += [os.path.join(current, f) for f in files if not f.endswith(".pyc")]
paths += model_files

missing = [f for f in FILES if not os.path.exists(os.path.join(ROOT, f))]
with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as package:
    for path in sorted(paths):
        package.write(path, f"{NAME}/" + os.path.relpath(path, ROOT).replace(os.sep, "/"))

print(f"Sparade {ZIP_FILE}: {len(paths)} filer, {os.path.getsize(ZIP_FILE) / 1e6:.1f} MB")
if missing:
    print("SAKNAS:", ", ".join(missing))
