import torch
import torch.nn as nn


def vgg_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_batchnorm: bool = True,
    non_linearity: str = "relu",
    padding: int = 1,
) -> torch.nn.Module:
    """
    The VGG block for the model.
    This block contains a 2D convolution, a ReLU activation, and a
    2D batch normalization layer.
    Args:
        in_channels (int): the number of input channels to the Conv2d layer
        out_channels (int): the number of output channels
        kernel_size (int): the size of the kernel for the Conv2d layer
        use_batchnorm (bool): whether or not to include a batchnorm layer.
            Default is true (batchnorm will be used).
    Returns:
        vgg_blk (nn.Sequential): the vgg block layer of the model
    """

    if non_linearity == "relu":
        non_linearity = torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    # the paper states that batchnorm is used after each convolution layer
    if use_batchnorm:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
            torch.nn.BatchNorm2d(out_channels),
        )
    # however, the official implementation does not include batchnorm
    else:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
        )

    return vgg_blk
