import torch
import torch.nn as nn

from .vgg import vgg_block


class VGGBackBone(torch.nn.Module):
    """
    The VGG backbone.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 128,
        use_batchnorm: bool = False,
        use_max_pooling: bool = True,
        padding: int = 1,
    ):
        """
        Initialize the VGG backbone model.
        Can take an input image of any number of channels (e.g. grayscale, RGB).
        """
        torch.nn.Module.__init__(self)

        assert padding in {0, 1}

        self.padding = padding
        self.use_max_pooling = use_max_pooling

        if use_max_pooling:
            self.mp = torch.nn.MaxPool2d(2, stride=2)
        else:
            self.mp = torch.nn.Identity()

        # convolution layers (encoder)
        self.l1 = torch.nn.Sequential(
            vgg_block(
                in_channels,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l2 = torch.nn.Sequential(
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l3 = torch.nn.Sequential(
            vgg_block(
                64,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l4 = torch.nn.Sequential(
            vgg_block(
                128,
                feat_channels,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                feat_channels,
                feat_channels,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Goes through the layers of the VGG model as the forward pass.
        Computes the output.
        Args:
            images (tensor): image pytorch tensor with
                shape N x num_channels x H x W
        Returns:
            output (tensor): the output point pytorch tensor with
            shape N x cell_size^2+1 x H/8 x W/8.
        """
        o1 = self.l1(images)
        o1 = self.mp(o1)

        o2 = self.l2(o1)
        o2 = self.mp(o2)

        o3 = self.l3(o2)
        o3 = self.mp(o3)

        # features
        o4 = self.l4(o3)

        return o4

