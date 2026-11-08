# Synthesis output

Output of the Analog Devices toolchain for MAX78002, produced by `../ai8x_training/run_ai8x.sh`.

| File | Content |
|---|---|
| `train.log` | Training with quantization-aware training in ai8x-training |
| `quantize.log` | Conversion of the trained checkpoint to 8-bit with ai8x-synthesis `quantize.py` |
| `evaluate.log` | Evaluation of the quantized model in 8-bit mode on the validation data, with confusion matrix |
| `synthesize.log` | Output of `ai8xize.py` |
| `ecg_rr/` | Generated C project for MAX78002: `cnn.c`, `cnn.h`, `main.c`, `weights.h`, `sampledata.h`, `sampleoutput.h`, `softmax.c`, `Makefile` and `log.txt` (layer and register configuration) |

`main.c` loads the weights, runs one sample beat through the accelerator and compares the result with the expected output in `sampleoutput.h`, printing PASS or FAIL. The C code has not been compiled or run, since neither the MAX78002 SDK nor a development kit was used. See `../ai8x_training/README.md` for the steps and their results.
