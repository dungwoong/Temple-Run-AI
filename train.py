"""
Doesn't even include a validation set, simple PoC training script.
"""

import torch.cuda

from dataset import TempleRunImageDataset, make_spreadsheet, make_weights_for_balanced_classes
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch
from load_pretrained import load_1_0, freeze

from nets.shufflenet import ShuffleNetV2

data_csv = 'imgs.csv'
make_spreadsheet(data_csv)
data = TempleRunImageDataset(data_csv)
batch_size = 64

# balance classes, make dataloader
weights = make_weights_for_balanced_classes(data_csv, param=50)
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)  # add shuffle=True if no sampler

# other stuff
train_steps = len(dataloader.dataset) // batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_1_0()
freeze(model, unfreeze=[model.fc, model.conv5, model.stage4])
model.to(device)
opt = Adam(model.parameters(), lr=1e-3)
lossfn = nn.CrossEntropyLoss()
print(device)

print('training the network')
for e in range(0, 5):
    print('Epoch', e)
    model.train()

    total_train_loss = 0
    train_correct = 0
    spec = True
    for (x, y) in dataloader:
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        loss = lossfn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sm = nn.LogSoftmax(dim=1)
        pred = sm(pred)
        if spec:
            print(pred.size())
            spec = False
        total_train_loss += loss
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print('avg train loss:', total_train_loss/train_steps)
    print('train accuracy:', train_correct/len(dataloader.dataset))

    torch.save(model.state_dict(), 'garbage.pth')

