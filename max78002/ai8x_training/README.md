# MAX78002 toolchain (ai8x-training and ai8x-synthesis)

The chosen CNN with RR input (filters 16-32-64) rebuilt with Analog Devices layers, trained with quantization-aware training, quantized and synthesized to C code for MAX78002.

## Result per step

| Step | Tool | Result |
|---|---|---|
| 1. Prepare the model for the MAX78002 training framework | ai8x-training | PASS |
| 2. Quantize the model | ai8x-training (QAT) and ai8x-synthesis `quantize.py` | PASS, checkpoint in `../quantization_output/ai87-ecg-rr-qat8-q.pth.tar` |
| 3. Synthesize the model for MAX78002 | ai8x-synthesis `ai8xize.py` | PASS, no errors or warnings in `log.txt` |
| 4. Generate device-specific C code | ai8x-synthesis `ai8xize.py` | PASS, C project in `../synthesis_output/ecg_rr/` |
| Compile the C code | MAX78002 SDK | Not done, SDK not installed |
| Run on hardware | MAX78002 development kit | Not done, no kit |

## Model on the accelerator

| Layer | Operation | Output |
|---|---|---|
| 0 | Conv1d 3 → 16, kernel 5, padding 2, ReLU | 16 × 100 |
| 1 | Max pool 2, Conv1d 16 → 32, kernel 5, padding 2, ReLU | 32 × 50 |
| 2 | Max pool 2, Conv1d 32 → 64, kernel 5, padding 2, ReLU | 64 × 25 |
| 3 | Max pool 2 (passthrough layer) | 64 × 12 |
| 4 | Flatten, Linear 768 → 4, 32-bit output | 4 |

16,228 parameters, 16,228 bytes of 8-bit weights and biases. On the accelerator pooling runs before the operation in the same layer, so the pooling after the first two convolutions moves into the next layer. Pooling cannot be combined with flattening, so the last pooling is a layer of its own.

The input is the same 3 × 100 input as the PyTorch model (ECG window, pre-RR, post-RR), with the z-normalized values mapped from −4…+4 to 0…1 by `export_data.py` and then to the 8-bit range by ai8x. About 0.03 % of the training values and 0.35 % of the validation values fall outside −4…+4 and are clipped.

## Accuracy of the ai8x model

Quantized model in 8-bit mode, validation data (9,763 beats, annotation RR):

| | N | S | V | F |
|---|---|---|---|---|
| Recall (%) | 96.6 | 7.3 | 41.4 | 0.0 |
| F1 (%) | 96.4 | 6.5 | 51.1 | 0.0 |

Accuracy 90.22 %, F1 macro 38.49 %. This is clearly lower than the PyTorch model (F1 macro 51.70 % for INT8, see `../model_configuration.md`), so the PyTorch model remains the main model. Likely reasons, not tested one by one:

- ai8x-training picks the best epoch by accuracy on a validation split taken from the training data. Since the training data was resampled with replacement to balance the classes, that split contains copies of training beats (98.8 % accuracy) and is a poor guide for selection. The PyTorch model was selected on F1 macro on separate validation records.
- Different optimizer and schedule (Adam with lr 0.001 instead of SGD with lr 0.02).
- Clipped input and 8-bit activations during quantization-aware training.

## How to run

In WSL/Ubuntu, with the tools in `~/ai8x`:

```
git clone --recurse-submodules https://github.com/analogdevicesinc/ai8x-training.git
git clone --recurse-submodules https://github.com/analogdevicesinc/ai8x-synthesis.git
```

Setup used here (ai8x-training commit 2dfe594, ai8x-synthesis commit 7379b29), with these differences from the official instructions:

- Python 3.11.8 (training) and 3.11.9 (synthesis) from `uv` instead of `pyenv`, since `pyenv` needs system packages that require administrator rights.
- PyTorch 2.3.1 CPU build from `https://download.pytorch.org/whl/cpu`, since there is no NVIDIA GPU.
- `pyffmpeg==2.4.2.18.1` is no longer available on PyPI, so `pyffmpeg==2.5.2.3.2` is installed instead. It is only used by a video dataset.
- `visdom` (needed by distiller) is built with `setuptools<70` and `--no-build-isolation`, since newer setuptools removed `pkg_resources`.

Then:

1. Export the data with the project's normal Python environment: `python max78002/ai8x_training/export_data.py`. This writes `../ai8x_data/ECG_RR/ecg_rr.npz` next to the repo.
2. In WSL: `bash max78002/ai8x_training/run_ai8x.sh all`, or one step at a time with `train`, `quantize`, `evaluate` or `synthesize`. Training takes about 15 minutes on a laptop CPU.

The script copies `datasets/ecg_rr.py`, `models/ai87net-ecg.py`, `schedule_ecg.yaml` and `qat_policy_ecg.yaml` into ai8x-training, and `ecg_rr.yaml` describes the network for ai8x-synthesis. DS2 is not used; ai8x calls the validation data its "test" set.
