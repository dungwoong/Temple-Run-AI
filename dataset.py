import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch
from PIL import Image

import matplotlib.pyplot as plt


def rename_by_indices(base_folder='data', graph=False):
    """
    Goes through all individual folders in <base_folder> reorganizes them starting from 000001

    Assumes files have 6 digits, with leading zeroes.
    Returns the total number of observations collected.
    """
    total = 0
    folders = []
    numbers = []
    for root, subfolder, path in os.walk(base_folder):
        root = root.replace('\\', '/')
        if not subfolder:
            subfolder = ['']
        for folder in subfolder:
            idx = 0
            if folder != '':
                folder += '/'
            for f in sorted(path):  # sorted, so starts from lowest.
                old_path = root + '/' + folder + f
                new_path = root + '/' + folder + '{:06d}'.format(idx) + '.jpg'
                os.rename(old_path, new_path)
                idx += 1
            if idx > 0:
                print(root + '/' + folder)
                print(idx)
                folders.append(root + '/' + folder)
                numbers.append(idx)
                total += idx
    print('total:', total)
    if graph:
        numbers, folders = zip(*sorted(zip(numbers, folders), key=lambda x: x[0]))
        plt.figure(figsize=(16, 8))
        plt.bar(folders, numbers)
        plt.title('count of obs in folders')
        plt.show()
    return total


def make_spreadsheet(out_path='imgs.csv', base_folder='data'):
    """
    Goes through <base_folder> making a spreadsheet of image labels and paths.
    """

    df_dict = {'path': [], 'label': []}
    labels = {f'{base_folder}/A': 'A',
              f'{base_folder}/D': 'D',
              f'{base_folder}/l': 'l',
              f'{base_folder}/w': 'w',
              f'{base_folder}/u': 'u',
              f'{base_folder}/r': 'r',
              f'{base_folder}/na': 'n'}
    dir_path = base_folder
    for root, subfolder, path in os.walk(dir_path):  # ok I understand this line is bad
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


def make_weights_for_balanced_classes(spreadsheet_path, param=0, label_col='label'):
    """
    Make weights for weighted random sampling, to balance classes.
    """
    df = pd.read_csv(spreadsheet_path)
    label_dict = {'n': 0,
                  'A': 1,
                  'D': 2,
                  'l': 3,
                  'r': 4,
                  'u': 5,
                  'w': 6}
    df[label_col] = df[label_col].map(label_dict)
    class_weights = len(df) / (param + df[label_col].value_counts().sort_index())
    img_weights = df[label_col].map(dict(class_weights))
    print(class_weights)
    return img_weights


class TempleRunImageDataset(Dataset):
    """
    Temple run image dataset.
    Labels are 0-7, observations are images.
    """

    def __init__(self, df_file):
        self.df = pd.read_csv(df_file)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        label_dict = {'n': 0,
                      'A': 1,
                      'D': 2,
                      'l': 3,
                      'r': 4,
                      'u': 5,
                      'w': 6}
        img_path = self.df.iloc[idx, 0]
        # img = read_image(img_path)  # C H W
        img = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(img)  # img.float() if img is tensor
        img = torch.transpose(img, 1, 2)  # C W H
        label = self.df.iloc[idx, 1]

        return img, label_dict[label]


if __name__ == '__main__':
    # rename_by_indices(graph=True)
    # make_spreadsheet('imgs.csv')
    w = make_weights_for_balanced_classes('imgs.csv')
    print(w)

    # img_dataset = TempleRunImageDataset('imgs.csv')
    # img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)
    # imgs, labels = next(iter(img_dataloader))
    # print(imgs.size())  # C W H
    # example = imgs[0].transpose(0, 2)  # H W C
    # plt.imshow(example)  # imshow is H W C
    # plt.title(labels[0])
    # plt.show()
