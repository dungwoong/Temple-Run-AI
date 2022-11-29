import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch

import matplotlib.pyplot as plt


def make_spreadsheet(root='data', out_path='imgs.csv'):
    df_dict = {'path': [], 'label': []}
    labels = {f'{root}/A': 'A',
              f'{root}/D': 'D',
              f'{root}/l': 'l',
              f'{root}/w': 'w',
              f'{root}/u': 'u',
              f'{root}/r': 'r',
              f'{root}/na': 'n'}
    dir_path = "data"
    for root, subfolder, path in os.walk(dir_path):
        root = root.replace('\\', '/')
        if root not in labels:
            continue
        label = labels[root]
        if not subfolder:
            subfolder = ['']
        for folder in subfolder:
            if folder != '':
                folder += '/'
            for f in path:
                full_path = root + '/' + folder + f
                df_dict['path'].append(full_path)
                df_dict['label'].append(label)

    df = pd.DataFrame(df_dict)
    df.to_csv(out_path, index=False)


class TempleRunImageDataset(Dataset):
    def __init__(self, df_file):
        self.df = pd.read_csv(df_file)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = read_image(img_path)  # C H W
        img = torch.transpose(img, 1, 2)  # C W H
        label = self.df.iloc[idx, 1]
        return img, label


if __name__ == '__main__':
    img_dataset = TempleRunImageDataset('imgs.csv')
    img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)
    imgs, labels = next(iter(img_dataloader))
    print(imgs.size())  # C W H
    example = imgs[0].transpose(0, 2) # H W C
    plt.imshow(example) # imshow is H W C
    plt.title(labels[0])
    plt.show()
