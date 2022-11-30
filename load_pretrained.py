from nets.shufflenet import ShuffleNetV2
import torch
from torch import nn


def load_1_0(new_num_classes=7, weights_file='../resources/shufflenetv2_x1-5666bf0f80.pth'):
    """
    Loads a pretrained shufflenet x1.0
    """
    net = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=1000)
    net.load_state_dict(torch.load(weights_file))
    final_in = net.fc.in_features
    net.fc = nn.Linear(final_in, new_num_classes)
    return net


def freeze(net: nn.Module, unfreeze=[]):
    """
    Freeze all parameters of net, except those in modules listed in <unfreeze>
    """
    for param in net.parameters():
        param.requires_grad = False
    for mod in unfreeze:
        for param in mod.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    net = load_1_0()
    freeze(net, unfreeze=[net.fc])
    # print(net)
    for p in net.parameters():
        if p.requires_grad:
            print(p.name, p.data.size())
