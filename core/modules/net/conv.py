import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_convs=3,
        kernel_size=3,
        stride=1,
        padding=1,
        downsample=True,
        dilation=1,
    ):
        super(ConvBlock, self).__init__()
        self.modules = []

        c_in = in_channels
        c_out = out_channels
        for i in range(n_convs):
            self.modules.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    dilation=dilation,
                )
            )
            self.modules.append(nn.BatchNorm2d(num_features=out_channels))
            self.modules.append(nn.LeakyReLU(0.1))
            c_in = c_out

        if downsample:
            self.modules.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.modules.append(nn.ReLU())
            # self.modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #     self.res_link = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=2,
        #     stride=2,
        #     padding=0,
        #     )
        # else:
        #     self.res_link = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     )

        self.model = nn.Sequential(*self.modules)
        self.init_weights()

    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # return self.model(x) + self.res_link(x)
        return self.model(x)
