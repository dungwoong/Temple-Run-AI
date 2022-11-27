from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax

"""LeNet is a convolutional architecture made by the legend himself Yann Lecunn.
It consists of convolutional/pooling layers with relu activation functions, which connect to a
fully connected network. Pretty shallow by today's standards, and quite inefficient compared to shufflenet
considering its size.

CODE FROM https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

I added a third convolutional + pooling layer, then averaged along channels so that I have a guaranteed
number of outputs to be fed into the FC portion of the network, rather than the architecture given in the URL above.

I hope this is not a naive decision. I dare not tread on the sacred architecture of LeNet, but I will anyways.
"""

"""
The original model's conv sizes, built for MNIST.
orig 28x28
conv1 24x24

maxpool1 12x12
conv2 8x8
maxpool2 4x4

Flatten: 4x4 * 50 = 800
this would be fed into a FC
"""


class LeNet(Module):
    def __init__(self, num_channels, num_classes):
        super(LeNet, self).__init__()

        # relu, between all layers
        # element-wise operations take non-negligible time.
        self.relu = ReLU()

        # first set of conv pooling
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=128, kernel_size=(5, 5))
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # shrinks image down by 2x width/height i think

        # second set of conv pooling
        self.conv2 = Conv2d(in_channels=64, out_channels=256,
                            kernel_size=(5, 5))  # the conv layers shrink the image as well
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # third set of conv pooling
        self.conv3 = Conv2d(in_channels=128, out_channels=512,
                            kernel_size=(5, 5))  # the conv layers shrink the image as well
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # fully connected layer
        self.fc1 = Linear(in_features=512, out_features=512)

        # softmax classifier
        self.fc2 = Linear(in_features=512, out_features=num_classes)

        self.logsoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # first set of layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # second set
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # third set
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        # flatten, send to FC
        x = x.mean([2, 3])  # N C W H so average pooling
        x = self.fc1(x)
        x = self.relu(x)

        # go to outputs
        x = self.fc2(x)

        return x
