# ------------------ Deployment -----------------
# Förbereder den valda INT8-modellen för MAX78002: lagertabell, minneskontroll,
# läsbar export av vikterna och en testkörning i Python

import glob
import json
import os

import numpy as np
import torch
import wfdb
from sklearn.metrics import f1_score

import lib

SEED = 42
FILTERS = [16, 32, 64]
WEIGHT_LIMIT_BYTES = 2_000_000    # MAX78002-gränser från Task 1
MEMORY_LIMIT_BYTES = 1_300_000
window_size = 100
offset = 0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "max78002")
QUANTIZATION_DIR = os.path.join(OUT_DIR, "quantization_output")
SYNTHESIS_DIR = os.path.join(OUT_DIR, "synthesis_output")
os.makedirs(QUANTIZATION_DIR, exist_ok=True)
os.makedirs(SYNTHESIS_DIR, exist_ok=True)

filters_name = "-".join(map(str, FILTERS))
fp32_file, = glob.glob(os.path.join(HERE, "models", f"SmallCNN-valF1-*-seed{SEED}-lr0.02-cw0-sampler1-rr1-filters{filters_name}.pth"))
int8_file, = glob.glob(os.path.join(HERE, "models", f"SmallCNN-INT8-valF1-*-seed{SEED}-filters{filters_name}.pth"))
fp32_model = lib.load_cnn_fp32(fp32_file, FILTERS, window_size)
int8_model = lib.load_cnn_int8(int8_file, FILTERS, window_size)

# ---- 1. Lagertabell: typ, in- och utdimensioner och parametrar per lager ----
layers = []
hooks = [layer.register_forward_hook(lambda layer, inputs, output: layers.append(
    (layer, tuple(inputs[0].shape[1:]), tuple(output.shape[1:])))) for layer in fp32_model]
with torch.no_grad():
    fp32_model(torch.zeros(1, 3, window_size))
for hook in hooks:
    hook.remove()


def layer_settings(layer):
    if isinstance(layer, torch.nn.Conv1d):
        return f"kernel {layer.kernel_size[0]}, stride {layer.stride[0]}, padding {layer.padding[0]}"
    if isinstance(layer, torch.nn.MaxPool1d):
        return f"kernel {layer.kernel_size}, stride {layer.stride}"
    return ""


def shape_text(shape):
    return " × ".join(map(str, shape))


# ---- 2. Minne: vikter i INT8 och största lagrets in- och utdata ----
n_parameters = sum(p.numel() for p in fp32_model.parameters())
n_weights = sum(layer.weight.numel() for layer in fp32_model if hasattr(layer, "weight"))
n_bias = n_parameters - n_weights
weight_bytes = n_weights + 4 * n_bias                  # INT8-vikter, bias sparas med 32 bitar i PyTorch
activation_bytes = max(int(np.prod(i) + np.prod(o)) for _, i, o in layers)   # 1 byte per värde i INT8


# ---- 3. Läsbar export av INT8-vikterna till JSON ----
def scales(weight):
    if weight.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
        return weight.q_per_channel_scales().tolist(), weight.q_per_channel_zero_points().tolist()
    return [weight.q_scale()], [weight.q_zero_point()]


export = {
    "model": os.path.basename(int8_file),
    "note": "weight = (weight_int8 - weight_zero_point) * weight_scale, one scale per output channel",
    "input": {"shape": [3, window_size], "channels": ["ECG window (z-normalized)", "pre-RR", "post-RR"],
              "scale": float(int8_model.quant.scale), "zero_point": int(int8_model.quant.zero_point)},
    "layers": [],
}
largest_difference = 0.0
for name, layer in int8_model.module.named_children():
    if not hasattr(layer, "_weight_bias"):
        continue   # ReLU, pooling och flatten har inga vikter
    weight = layer.weight()
    weight_scale, weight_zero_point = scales(weight)
    export["layers"].append({
        "name": name,
        "type": type(layer).__name__,
        "weight_shape": list(weight.shape),
        "weight_int8": weight.int_repr().tolist(),
        "weight_scale": weight_scale,
        "weight_zero_point": weight_zero_point,
        "bias": layer.bias().tolist(),
        "output_scale": float(layer.scale),
        "output_zero_point": int(layer.zero_point),
    })

    # Kontroll: vikterna byggda från JSON-talen ska bli samma som modellens
    shape = [-1] + [1] * (weight.dim() - 1)
    rebuilt = ((np.array(export["layers"][-1]["weight_int8"]) - np.reshape(weight_zero_point, shape))
               * np.reshape(weight_scale, shape))
    largest_difference = max(largest_difference, float(np.abs(rebuilt - weight.dequantize().numpy()).max()))

JSON_FILE = os.path.join(QUANTIZATION_DIR, "model_int8.json")
with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(export, f, indent=2)

# ---- 4. Testkörning i Python på valideringsdatan, med annoteringarnas RR som i quantize.py ----
rr_mean, rr_std = lib.train_rr_stats(window_size, offset)
X, y = [], []
for record_name in lib.val_records:
    record = wfdb.rdrecord(lib.directory + record_name)
    annotation = wfdb.rdann(lib.directory + record_name, 'atr')
    signal = record.p_signal[:, 0]
    beat_samples, labels = lib.kept_beats(annotation, len(signal), window_size, offset)
    rr = (lib.rr_intervals(beat_samples, lib.annotated_peaks(annotation), record.fs) - rr_mean) / rr_std
    windows = lib.beat_windows(lib.filter_signal(signal, record.fs), beat_samples, window_size, offset)
    X.append(np.concatenate([windows[:, None, :], np.repeat(rr[:, :, None], window_size, axis=2)], axis=1))
    y.append(labels)
X = torch.tensor(np.concatenate(X), dtype=torch.float32)
y = np.concatenate(y)

with torch.no_grad():
    fp32_pred = fp32_model(X).argmax(dim=1).numpy()
    int8_pred = torch.cat([int8_model(batch).argmax(dim=1) for batch in X.split(256)]).numpy()
fp32_f1 = f1_score(y, fp32_pred, average='macro')
int8_f1 = f1_score(y, int8_pred, average='macro')
same_prediction = (fp32_pred == int8_pred).mean()

# ---- Sammanfattning ----
rows = [f"| {i + 1} | {type(layer).__name__} | {shape_text(i_shape)} | {shape_text(o_shape)} | {layer_settings(layer)} "
        f"| {sum(p.numel() for p in layer.parameters()):,} |" for i, (layer, i_shape, o_shape) in enumerate(layers)]

lines = [
    "# MAX78002 model configuration",
    "",
    f"Model: CNN with RR input, filters {filters_name}, INT8 (PyTorch post-training quantization), seed {SEED}.  ",
    f"INT8 file: `models/{os.path.basename(int8_file)}`  ",
    f"FP32 file: `models/{os.path.basename(fp32_file)}`",
    "",
    "## Input and output",
    "",
    f"- **Input:** 3 × {window_size}. Channel 1 is the ECG window (lead MLII, {window_size} samples = "
    f"{window_size / 360 * 1000:.0f} ms at 360 Hz, band-pass and notch filtered, z-normalized per window). "
    "Channels 2 and 3 hold pre-RR and post-RR as constant values, normalized with the training mean and std.",
    "- **Output:** 4 values, one per class (N, S, V, F). The predicted class is the largest value.",
    "",
    "## Layers",
    "",
    "| # | Type | Input | Output | Settings | Parameters |",
    "|---|---|---|---|---|---|",
    *rows,
    f"| | **Total** | | | | **{n_parameters:,}** |",
    "",
    "In the INT8 model each Conv1d and ReLU pair is fused into one ConvReLU1d layer.",
    "",
    "## Size and memory",
    "",
    "| | Value | Limit |",
    "|---|---|---|",
    f"| Parameters | {n_parameters:,} ({n_weights:,} weights, {n_bias} biases) | |",
    f"| Weights in INT8 (biases stored as 32-bit) | {weight_bytes / 1000:.1f} kB | {WEIGHT_LIMIT_BYTES / 1e6:.1f} MB |",
    f"| Largest layer input + output in INT8 (estimate) | {activation_bytes / 1000:.1f} kB | {MEMORY_LIMIT_BYTES / 1e6:.1f} MB |",
    "",
    "## Software verification",
    "",
    f"- Validation data ({len(y):,} beats, annotation RR): F1 macro {fp32_f1 * 100:.2f} % for FP32 and "
    f"{int8_f1 * 100:.2f} % for INT8. Both give the same prediction for {same_prediction * 100:.2f} % of the beats.",
    f"- The INT8 weights are exported to `quantization_output/model_int8.json`. Rebuilding the weights from that file "
    f"gives the same values as the model (largest difference {largest_difference:.1e}).",
    "",
    "## Notes",
    "",
    "- Verified in software only (Python, PyTorch). The model has not been run on MAX78002 hardware.",
    "- The model is quantized with PyTorch, not with Analog Devices ai8x-training, which uses its own quantization.",
    "- The RR values are not computed by the CNN accelerator. They come from an R-peak detector running on the "
    "microcontroller core and are passed to the CNN as input channels.",
    "- Not yet checked against the ai8x documentation: Conv1d with kernel 5 and padding 2, how pooling after a "
    "convolution maps to the accelerator, and a Linear layer with 768 inputs.",
]
CONFIG_FILE = os.path.join(OUT_DIR, "model_configuration.md")
with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

with open(os.path.join(SYNTHESIS_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write("# Synthesis output\n\n"
            "Synthesis to C code with Analog Devices ai8x-synthesis was not performed. "
            "The course instructor confirmed that deployment may stay in Python, so the model is verified in software only. "
            "See `../model_configuration.md` for the layer table, memory use and software verification.\n")

print("\n".join(lines))
print(f"\nSparade {CONFIG_FILE} och {JSON_FILE}")
