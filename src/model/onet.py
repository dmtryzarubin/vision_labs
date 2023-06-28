from torch import nn

__all__ = ["ONet"]


def conv3x3mp3x3(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(3, 2, ceil_mode=True),
    )


def conv3x3mp2x2(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
    )


class ONet(nn.Sequential):
    def __init__(self, dropout: float = 0.3, *args, **kwargs):
        super().__init__()
        self.add_module("block1", conv3x3mp3x3(3, 32))
        self.add_module("block2", conv3x3mp3x3(32, 64))
        self.add_module("block3", conv3x3mp2x2(64, 64))
        self.add_module("block4", nn.Conv2d(64, 128, 2))
        self.add_module("flatten", nn.Flatten())
        self.add_module("drop1", nn.Dropout(p=dropout))
        self.add_module("fc1", nn.Linear(128 * 3 * 3, 256))
        self.add_module("drop1", nn.Dropout(p=dropout))
        self.add_module("fc2", nn.Linear(256, 136))
