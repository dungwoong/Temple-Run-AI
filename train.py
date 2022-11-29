import torch.cuda

from dataset import TempleRunImageDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch

from nets.shufflenet import ShuffleNetV2

data = TempleRunImageDataset('imgs.csv')
batch_size = 64
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
train_steps = len(dataloader.dataset) // batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=7).to(device)
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

    torch.save(model.state_dict, 'garbage.pth')

