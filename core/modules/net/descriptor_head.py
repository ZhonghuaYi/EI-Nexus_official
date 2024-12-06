import torch
import torch.nn as nn

from .vgg import vgg_block


class VGGDescriptorHead(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        use_batchnorm: bool = True,
        padding: int = 1,
    ) -> None:
        torch.nn.Module.__init__(self)

        assert padding in {0, 1}

        # descriptor head (decoder)
        self._desH1 = vgg_block(
            in_channels,
            out_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm - note that normailzation is calculated later
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
            )

    def forward(self, x: torch.Tensor):
        x = self._desH1(x)
        x = self._desH2(x)
        return x
    