"""
Settings for the chosen model, read from configs/final_model.json.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "configs", "final_model.json"), encoding="utf-8") as f:
    CONFIG = json.load(f)
