"""
CNN with RR input for MAX78002, same layers as the chosen SmallCNN (filters 16-32-64).
Copied into ai8x-training/models/ by run_ai8x.sh.
"""
from torch import nn

import ai8x


class AI87ECGNet(nn.Module):
    """
    Conv1d + ReLU blocks with max pooling, then one Linear layer.
    On the accelerator pooling runs before the convolution in the same layer, so the pooling
    after block 1 and 2 moves into the next layer. Pooling and flattening cannot share a layer,
    so the last pooling is a layer of its own.
    """
    def __init__(self, num_classes=4, num_channels=3, dimensions=(100, 1), bias=False, **kwargs):
        super().__init__()
        length = dimensions[0]
        self.conv1 = ai8x.FusedConv1dReLU(num_channels, 16, 5, padding=2, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv1dReLU(16, 32, 5, pool_size=2, pool_stride=2,
                                                 padding=2, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv1dReLU(32, 64, 5, pool_size=2, pool_stride=2,
                                                 padding=2, bias=bias, **kwargs)
        self.pool = ai8x.MaxPool1d(2, stride=2, **kwargs)
        self.fc = ai8x.Linear(64 * (length // 8), num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ai87ecgnet(pretrained=False, **kwargs):
    """
    Constructs an AI87ECGNet model.
    """
    assert not pretrained
    return AI87ECGNet(**kwargs)


models = [
    {
        'name': 'ai87ecgnet',
        'min_input': 1,
        'dim': 1,
    },
]
