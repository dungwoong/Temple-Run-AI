import time
import torch
import matplotlib.pyplot as plt

from nets.shufflenet import ShuffleNetV2
from nets.lenet import LeNet
from torchvision import transforms
from PIL import Image


def load_image():
    img = Image.open('res/templerun.jpg')
    trans = transforms.ToTensor()
    return trans(img)


def test_net(mod, n=10):
    im = load_image()
    im = torch.unsqueeze(im, 0)  # B C H W but that's whatever

    avg = []
    for i in range(n):
        start = time.time()
        mod(im)
        duration = time.time() - start
        print(f"{i} - {duration}s")
        avg.append(duration)
    print(f'avg: {sum(avg) / len(avg)}s')
    return avg


if __name__ == '__main__':
    num_runs = 50
    shufflenet_1_0 = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=6)
    lenet = LeNet(num_channels=3, num_classes=6)
    mods = {'shufflenetx1.0': shufflenet_1_0,
            'lenet(kinda)': lenet}
    for mod in mods:
        print(f'############ TESTING {mod.upper()} ################')
        mod_times = test_net(mods[mod], num_runs)
        plt.style.use('ggplot')
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        plt.suptitle(f'{mod} runtimes(n={num_runs}), avg={sum(mod_times) / len(mod_times)}')
        ax1.hist(mod_times)
        ax1.set_xlabel('Runtime(seconds)')
        ax2.boxplot(mod_times)
        plt.savefig(f'misc_img/model_runtimes/{mod}.png')

