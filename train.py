"""
Doesn't even include a validation set, simple PoC training script.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

from dataset import TempleRunImageDataset, make_spreadsheet, make_weights_for_balanced_classes
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch
from load_pretrained import load_1_0, load_0_5, freeze

from nets.shufflenet import ShuffleNetV2

MODEL = 0.5
TRAIN_VAL_SPLIT = 0.3
INITIAL_LR = 3e-4
BATCH_SIZE = 64
BALANCE_CLASSES = False
BALANCE_CLASSES_PARAM = 75
NUM_EPOCHS = 20
WEIGHT_DECAY = 3e-5
RANDOMEAFFINE = True
UNFREEZE_MORE = True
AUTO_EPOCH = True
REDUCE_AT_EPOCH = 100

# changed ADAM LR
# added dropout, but removed it
# tried balancing classes, didn't seem to help much as dataset is so small
# TRY using x0.5 --> seems better
# try using randomaffine --> much better. Less overfitting, we break 75%, but overall still not too accurate
# it's about 75-80 train/val acc.
# maybe collect more data from early stages of the game. I think too much river/wood/minecart is messing it up.
# I think we need to restrict gameplay to slower levels, else the model will predict jump too soon.
# will try unfreezing block 3, reduce LR, 20 epochs
# max score 104,494. NICE! I think i care more about distance than score, though.
# I think i need to collect data from earlier parts of the game cuz it speeds up.
# it always suffers early, once it makes it past early it's super ez.
# start collecting data from before 100k
# look at gradients

data_csv = 'imgs.csv'
make_spreadsheet(data_csv)
df = pd.read_csv(data_csv)
# split into train val
val_msk = np.random.rand(len(df)) < TRAIN_VAL_SPLIT
df_val = df[val_msk]
df_val.reset_index(drop=True, inplace=True)
df_train = df[~val_msk]
df_train.reset_index(drop=True, inplace=True)
# print(df_val)

if RANDOMEAFFINE:
    ra = transforms.RandomAffine(degrees=5, translate=(0.2, 0.1), scale=(1, 1.2))
    ra = [ra]
else:
    ra = []

train_data = TempleRunImageDataset(df_train, randomaffine=ra)
val_data = TempleRunImageDataset(df_val)

# balance classes, make dataloader
if BALANCE_CLASSES:
    weights = make_weights_for_balanced_classes(df_train.copy(), param=BALANCE_CLASSES_PARAM)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)  # add shuffle=True if no sampler
else:
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# validation set
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

# other stuff
train_steps = len(train_dataloader.dataset) // BATCH_SIZE
val_steps = len(val_dataloader.dataset) // BATCH_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load and unfreeze some layers of the model.
if MODEL == 1:
    model = load_1_0()
else:
    model = load_0_5()
freeze(model, unfreeze=[model.fc, model.conv5, model.stage4])
model.to(device)
opt = Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

lossfn = nn.CrossEntropyLoss()
print(device)

metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print('training the network')
for e in range(0, NUM_EPOCHS):
    if e == REDUCE_AT_EPOCH:
        opt = Adam(model.parameters(), lr=5e-5, weight_decay=5e-6)

    if UNFREEZE_MORE and e == 14:
        print('[UNFREEZING STAGE 3]')
        freeze(model, unfreeze=[model.fc, model.conv5, model.stage4, model.stage3])
        # opt = Adam(model.parameters(), lr=5e-5, weight_decay=5e-6)
    print('Epoch', e + 1)
    model.train()

    total_train_loss = 0
    train_correct = 0
    total_val_loss = 0
    val_correct = 0
    spec = True
    for (x, y) in train_dataloader:
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        loss = lossfn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sm = nn.LogSoftmax(dim=1)
        pred = sm(pred)
        # if spec:
        #     print(pred.size())
        #     spec = False
        total_train_loss += loss
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        for (x, y) in val_dataloader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            total_val_loss += lossfn(pred, y)
            val_correct += (pred.argmax(1) == y).type(
                torch.float).sum().item()

    avg_train_loss = total_train_loss/train_steps
    train_acc = train_correct / len(train_dataloader.dataset)
    avg_val_loss = total_val_loss/val_steps
    val_acc = val_correct / len(val_dataloader.dataset)
    print('avg train loss:', avg_train_loss)
    print('train accuracy:', train_acc)
    print('avg val loss:', avg_val_loss)
    print('val accuracy:', val_acc)

    metrics['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    metrics['train_acc'].append(train_acc)
    metrics['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    metrics['val_acc'].append(val_acc)

    torch.save(model.state_dict(), 'garbage.pth')
    if not AUTO_EPOCH and e > 13:
        cont = input('Continue[y/n] ')
        if cont == 'n':
            break

plt.style.use('ggplot')
plt.figure(figsize=(16, 8))
plt.plot(metrics['train_loss'], label='train_loss')
plt.plot(metrics['val_loss'], label='val_loss')
plt.plot(metrics['val_acc'], label='val_acc')
plt.plot(metrics['train_acc'], label='train_acc')
plt.title('Training loss and accuracy on dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.show()
