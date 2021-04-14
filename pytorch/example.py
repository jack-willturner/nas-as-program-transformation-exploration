import torch
import numpy as np
from models import *

seed = 0

np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gen_random_net_config():
    res34, strides = [3, 4, 6, 3], [1, 2, 2, 2]
    configs = []

    for block, stride in zip(res34, strides):
        configs_ = []
        for i, layer in enumerate(range(block)):
            subconfig = {}
            conv = np.random.choice([Conv, SplitConv, UnrollConv, SpatialGroupConv])
            subconfig["conv"] = conv
            subconfig["stride"] = stride if i == 0 else 1
            if conv == SplitConv:
                sf = np.random.choice([1, 2, 4, 8])
                subconfig["split_factor"] = sf
                subconfig["groups"] = np.random.choice([1, 2, 4, 8], sf)
            elif conv == UnrollConv:
                subconfig["unroll_factor"] = np.random.choice([1, 2, 4, 8, 16])
                subconfig["unrollconv_groups"] = np.random.choice([1, 2, 3, 4])
            elif conv == SpatialGroupConv:
                sf = np.random.choice([1, 2, 4, 8])
                subconfig["split_factor"] = sf

            configs_.append(subconfig)
        configs.append(configs_)
    return configs


invalid = 0
for i in range(100):
    try:
        model = ResNet34(gen_random_net_config())

        test = torch.randn((1, 3, 32, 32))
        model(test)
    except:
        invalid += 1

print(f"{100-invalid}/100 configs were valid")
