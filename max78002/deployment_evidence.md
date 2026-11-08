# Deployment evidence (MAX78002)

Evidence of the synthesis process for the CNN with RR input (filters 16-32-64), produced with Analog Devices ai8x-training (commit 2dfe594) and ai8x-synthesis (commit 7379b29) by `ai8x_training/run_ai8x.sh`, target `--device MAX78002`.

| Requirement | Result | Evidence |
|---|---|---|
| Model accepted by MAX78002 toolchain | PASS | The model is built from ai8x layers (`ai8x_training/models/ai87net-ecg.py`) and trained for MAX78002 (`synthesis_output/train.log`). ai8x-synthesis reads the checkpoint and the network description `ai8x_training/ecg_rr.yaml` without errors (`synthesis_output/synthesize.log`). |
| Quantization completed | PASS | Quantization-aware training from epoch 25, then `quantize.py` converts all 4 parameter layers to 8-bit weights and biases (`synthesis_output/quantize.log`). The quantized model is evaluated in 8-bit mode in `synthesis_output/evaluate.log`. Checkpoint: `quantization_output/ai87-ecg-rr-qat8-q.pth.tar`. |
| Synthesis completed | PASS | `ai8xize.py` maps the network to 5 accelerator layers using all 4 quadrants and 64 processors. No errors or warnings in `synthesis_output/ecg_rr/log.txt`. |
| C/device files generated | PASS | `synthesis_output/ecg_rr/`: `cnn.c`, `cnn.h` (accelerator configuration), `weights.h` (kernels and biases), `main.c` (loads weights, runs a sample, checks the output), `sampledata.h`, `sampleoutput.h`, `softmax.c`, `Makefile`. |
| Weight-memory requirement | 16,228 bytes (PASS) | 16,112 weights and 116 biases, all 8-bit, as reported by ai8x-synthesis (`synthesize.log`: "16,228 parameters, 16,228 bytes"). Limit 2 MB. |
| Data-memory requirement | about 3.2 kB (PASS) | Largest layer input plus output at 8 bits: 16 × 100 in and 32 × 50 out in layer 1. Calculated from the layer dimensions in `synthesis_output/ecg_rr/log.txt`; the toolchain places the layer data in two alternating buffers (offsets 0x0000 and 0x2000). Limit 1.3 MB. |

## Accelerator layers

From `synthesis_output/ecg_rr/log.txt`:

| Layer | Operator | Input | Pooling | Output | Activation | Output bits |
|---|---|---|---|---|---|---|
| 0 | Conv1d, kernel 5, padding 2 | 3 × 100 | none | 16 × 100 | ReLU | 8 |
| 1 | Conv1d, kernel 5, padding 2 | 16 × 100 | max 2 | 32 × 50 | ReLU | 8 |
| 2 | Conv1d, kernel 5, padding 2 | 32 × 50 | max 2 | 64 × 25 | ReLU | 8 |
| 3 | Passthrough | 64 × 25 | max 2 | 64 × 12 | none | 8 |
| 4 | Linear (flatten 768) | 64 × 12 | none | 4 | none | 32 |

## Architecture changes needed for synthesis

The first attempt was accepted after two adjustments, both without changing what the model computes:

- On the accelerator, pooling is done before the operation in the same layer, so the pooling after the first two convolutions is moved into the following layer.
- Pooling cannot be combined with flattening, so the last pooling became a separate passthrough layer.

The input is scaled to the 8-bit range the accelerator uses (−4…+4 in z-values mapped to −128…+127), and the last layer outputs 32 bits (`wide=True`) so the class scores are not clipped.

## Verification level

| | Status |
|---|---|
| Trained, quantized and evaluated in Python (ai8x-training, 8-bit mode) | Verified in software |
| Synthesized and C code generated (ai8x-synthesis) | Verified in software |
| C code compiled with the MAX78002 SDK | Not done |
| Run on MAX78002 hardware | Not done, no development kit. No latency or power figures are claimed. |

## Accuracy of the synthesized model

The synthesized model was retrained with ai8x-training and reaches F1 macro 38.49 % and accuracy 90.22 % on the validation data in 8-bit mode, against 51.70 % for the PyTorch INT8 model. See `ai8x_training/README.md` for the likely reasons.
