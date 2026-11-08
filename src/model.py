"""
The chosen CNN with RR input, and loading of its FP32 and INT8 versions.
"""
import torch
import torch.nn as nn


def build_cnn(filters, in_channels=3, window_size=100):
    layers = []
    length = window_size
    for out_channels in filters:
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)]
        in_channels = out_channels
        length //= 2
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(in_channels * length, 4))


def load_cnn_fp32(path, filters, window_size=100):
    model = build_cnn(filters, window_size=window_size)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def load_cnn_int8(path, filters, window_size=100):
    # Samma steg som i quantize.py, men skalorna läses från filen i stället för att kalibreras
    import torch.ao.quantization as quantization
    torch.backends.quantized.engine = "onednn"
    conv_relu_pairs = [[str(i), str(i + 1)] for i in range(0, 3 * len(filters), 3)]
    model = quantization.QuantWrapper(quantization.fuse_modules(build_cnn(filters, window_size=window_size), conv_relu_pairs)).eval()
    model.qconfig = quantization.get_default_qconfig("onednn")
    quantization.prepare(model, inplace=True)
    with torch.no_grad():
        model(torch.zeros(1, 3, window_size))
    model = quantization.convert(model)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return model
