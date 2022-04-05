import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from matplotlib import pyplot as plt

input_size = 1*28*28
num_classes = 10
batch_size = 100
lr = 0.01

train_set = torchvision.datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root="./data/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)

class LogisticModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

model = LogisticModel(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_set = []

for epoch in range(100):
    for step, (x, y) in enumerate(train_loader):
        x = torch.autograd.Variable(x.view(-1, input_size))
        y = torch.autograd.Variable(y)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            loss_set.append(loss.data)
            print(f"epoch = {epoch} current loss = {loss.data}")

rand_data = torch.rand(input_size)
store = torch.jit.trace(model, rand_data)
torch.jit.save(store, "logistic.pt")
model = torch.jit.load("logistic.pt")

correct = 0
total = 0

for (x, y) in test_loader:
    x = torch.autograd.Variable(x.view(-1, input_size))
    y_pred = model(x)
    print(x.shape)
    _, pred = torch.max(y_pred.data, 1)
    total += y.size(0)
    correct += (pred == y).sum()

print('accuracy of the model %.2f' % (100 * correct / total))
plt.plot(loss_set)
plt.show()
