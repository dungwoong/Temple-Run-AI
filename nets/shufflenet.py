# CODE SOURCE:
# https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/shufflenetv2.py
# THIS IS NOT MY CODE: it is copied and annotated for my use.

import torch
import torch.nn as nn


# can alternatively use nn.ChannelShuffle() but I will make the function myself
# so I can annotate it.
def channel_shuffle(x: torch.Tensor, g: int) -> torch.Tensor:
    """
    Shuffles channels. Non-random shuffle, done by splitting the channels into groups
    then transposing the resulting tensor, allowing groups to communicate with each other.

    Precondition: x's number of channels is divisible by g(number of groups).
    """
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // g

    # reshape, create groups
    x = x.view(batch_size, g, channels_per_group, height, width)

    # shuffle by transposing groups and channels
    x = torch.transpose(x, 2, 1).contiguous()

    # flatten back into original shape
    x = x.view(batch_size, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual block.

    Block c) or d) in Ma et al. 2018.

    If stride > 1, is block d).
    Performs spatial downsampling and has 2 branches.

    If stride == 1, is block c).
    Basic unit. Doesn't perform spatial downsampling.

    Channel shuffle at the end.
    """

    def __init__(self, inp, oup, stride):
        """
        inp: in channels
        oup: out channels
        stride: stride of the intermediate conv layers.
        """
        # I think inp and oup is channels in and out.
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('Illegal Stride Value')

        self.stride = stride

        branch_features = oup // 2  # feature maps in each branch

        # asserts that the spatial dimensions will actually go down(?)
        assert (self.stride != 1) or (inp == branch_features << 1)  # remember << is bitwise right shift(eq. x2?)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        # each input channel will be convolved with its own set of filters, of size o/i
        # thus, if o == i, we get depthwise separable convolutions!
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # break channels in half
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)  # don't break channels in half.

        out = channel_shuffle(out, 2)  # channel shuffle.

        return out


class ShuffleNetV2(nn.module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual):
        """
        Initializes a ShuffleNetV2 Model.

        Architecture: Conv+BN+ReLU > Max Pool > 3 Blocks of shit > Conv+BN+ReLU > FC

        Each block of shit consists of one d) block, then a few c) blocks from Ma et al. 2018

        @param stages_repeats: number of times to repeat c) blocks in the 3 middle blocks
        @param stages_out_channels: output channels of each stage
        """
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('Expected stages repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('Expected stages out channels as list of 5 positive ints')

        self._stage_out_channels = stages_out_channels

        ###################################################
        # first conv block
        ###################################################

        # input and output channels of the first conv block
        input_channels = 3  # RGB input
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        input_channels = output_channels

        # MAX POOL
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ###################################################
        # 3 middle blocks of d + repeat * c
        ###################################################
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]

        # put together stage name, repeats, output_channels for each stage beyond the first.
        # note that we still have one int in _stage_out_channels.
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]
        ):
            seq = [inverted_residual(input_channels, output_channels, stride=2)]  # block d
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, stride=1))  # block c

                # set stage_i to a sequential block.
                setattr(self, name, nn.Sequential(*seq))  # passes list in seq as arguments to init
                input_channels = output_channels

        ###################################################
        # Final conv block
        ###################################################
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # I have 0 idea why they do this
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # N C W H so this performs avg pooling
        return x

    def forward(self, x):
        return self._forward_impl(x)

# original repo has functions for generating a pretrained shufflenet, x0.5, etc. but yeah.
# I'm not planning on using a pretrained shufflenet, but we'll see.


def get_args(x: str):
    """
    Get the arguments necessary for a specific sized shufflenet model (x1.0, x0.5, etc.)
    """
    stages_repeats = {"1_0": [4, 8, 4],
                      "0_5": [4, 8, 4]}
    stages_out_channels = {"1_0": [24, 116, 232, 464, 1024],
                           "0_5": [24, 48, 96, 192, 1024]}
    return {'stages_repeats': stages_repeats[x],
            'stages_out_channels': stages_out_channels[x]}