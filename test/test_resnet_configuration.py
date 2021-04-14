import torch
from pytorch.models import *


def test_default_resnet():
    configs = [
        [{"conv": Conv, "stride": 1}, {"conv": Conv, "stride": 1}],
        [{"conv": Conv, "stride": 2}, {"conv": Conv, "stride": 1}],
        [{"conv": Conv, "stride": 2}, {"conv": Conv, "stride": 1}],
        [{"conv": Conv, "stride": 2}, {"conv": Conv, "stride": 1}],
    ]

    net = ResNet18(configs)
    y = net(torch.randn(1, 3, 32, 32))
    assert y is not None
