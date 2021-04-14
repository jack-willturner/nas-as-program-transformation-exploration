import torch
import torch.nn as nn

__all__ = ["Conv", "Seq1", "Seq2", "Seq3"]

'''
Each of the convolutions in this file correspond to sequences in 
section 7.3 of the paper. 
'''

class ConvModule(nn.Module):
    def _cache_sizes(self, x, convs):
        self._sizecache = []

        N, CI, H, W = x.size()
        CO, KH, KW, stride, pad, G = (
            convs[0].out_channels,
            convs[0].kernel_size[0],
            conv[0].kernel_size[1],
            convs[0].stride[0],
            convs[0].padding[0],
            convs[0].groups,
        )

        for conv in convs:
            N, CI, H, W = x.size()
            CO, KH, KW, stride, pad, G = (
                conv.out_channels,
                conv.kernel_size[0],
                conv.kernel_size[1],
                conv.stride[0],
                conv.padding[0],
                conv.groups,
            )
            self._sizecache.append([N, H, W, CO, CI, KH, KW, stride, pad, G])
            x = conv(x)


class Conv(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1, args=None
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)


class Seq1(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1, args=None
    ):
        super(Seq1, self).__init__()
        convs = []
        sf = args["split_factor"]
        for i, layer in enumerate(range(sf)):
            g = args["groups"][i]
            convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels // sf,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=padding,
                    groups=g,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)


class Seq2(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1, args=None
    ):
        super(Seq2, self).__init__()
        self.unroll_factor = args["unroll_factor"]
        g = args["unrollconv_groups"]
        self.conv1 = nn.Conv2d(
            in_channels,
            self.unroll_factor,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.convg1 = nn.Conv2d(
            (in_channels - self.unroll_factor),
            (out_channels - self.unroll_factor),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=g,
        )

    def forward(self, x):
        l_slice = x
        r_slice = x[:, self.unroll_factor :, :, :]

        l_out = self.conv1(l_slice)
        r_out = self.convg1(r_slice)

        return torch.cat((l_out, r_out), 1)


class Seq3(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1, args=None
    ):
        super(Seq3, self).__init__()
        self.split_factor = args["split_factor"]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=padding,
                )
                for i in range(args["split_factor"])
            ]
        )

    def forward(self, x):
        H = x.shape[2]
        Hg = H // self.split_factor

        outs = []
        for i, conv in enumerate(self.convs):
            x_ = x[:, :, i * Hg : (i + 1) * Hg, :]
            outs.append(conv(x_))

        return torch.cat(outs, 2)
