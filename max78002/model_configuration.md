# MAX78002 model configuration

Model: CNN with RR input, filters 16-32-64, INT8 (PyTorch post-training quantization), seed 42.  
INT8 file: `models/SmallCNN-INT8-valF1-0.5274-seed42-filters16-32-64.pth`  
FP32 file: `models/SmallCNN-valF1-0.5240-seed42-lr0.02-cw0-sampler1-rr1-filters16-32-64.pth`

## Input and output

- **Input:** 3 × 100. Channel 1 is the ECG window (lead MLII, 100 samples = 278 ms at 360 Hz, band-pass and notch filtered, z-normalized per window). Channels 2 and 3 hold pre-RR and post-RR as constant values, normalized with the training mean and std.
- **Output:** 4 values, one per class (N, S, V, F). The predicted class is the largest value.

## Layers

| # | Type | Input | Output | Settings | Parameters |
|---|---|---|---|---|---|
| 1 | Conv1d | 3 × 100 | 16 × 100 | kernel 5, stride 1, padding 2 | 256 |
| 2 | ReLU | 16 × 100 | 16 × 100 |  | 0 |
| 3 | MaxPool1d | 16 × 100 | 16 × 50 | kernel 2, stride 2 | 0 |
| 4 | Conv1d | 16 × 50 | 32 × 50 | kernel 5, stride 1, padding 2 | 2,592 |
| 5 | ReLU | 32 × 50 | 32 × 50 |  | 0 |
| 6 | MaxPool1d | 32 × 50 | 32 × 25 | kernel 2, stride 2 | 0 |
| 7 | Conv1d | 32 × 25 | 64 × 25 | kernel 5, stride 1, padding 2 | 10,304 |
| 8 | ReLU | 64 × 25 | 64 × 25 |  | 0 |
| 9 | MaxPool1d | 64 × 25 | 64 × 12 | kernel 2, stride 2 | 0 |
| 10 | Flatten | 64 × 12 | 768 |  | 0 |
| 11 | Linear | 768 | 4 |  | 3,076 |
| | **Total** | | | | **16,228** |

In the INT8 model each Conv1d and ReLU pair is fused into one ConvReLU1d layer.

## Size and memory

| | Value | Limit |
|---|---|---|
| Parameters | 16,228 (16,112 weights, 116 biases) | |
| Weights in INT8 (biases stored as 32-bit) | 16.6 kB | 2.0 MB |
| Largest layer input + output in INT8 (estimate) | 3.2 kB | 1.3 MB |

## Software verification

- Validation data (9,763 beats, annotation RR): F1 macro 52.40 % for FP32 and 52.74 % for INT8. Both give the same prediction for 99.46 % of the beats.
- The INT8 weights are exported to `quantization_output/model_int8.json`. Rebuilding the weights from that file gives the same values as the model (largest difference 4.5e-08).

## Notes

- Verified in software only (Python, PyTorch). The model has not been run on MAX78002 hardware.
- The model is quantized with PyTorch, not with Analog Devices ai8x-training, which uses its own quantization.
- The RR values are not computed by the CNN accelerator. They come from an R-peak detector running on the microcontroller core and are passed to the CNN as input channels.
- Not yet checked against the ai8x documentation: Conv1d with kernel 5 and padding 2, how pooling after a convolution maps to the accelerator, and a Linear layer with 768 inputs.
