import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from helpers import eval_model

train_dataset = MNIST("datasets", download=True, transform=ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    for x, y in train_dataloader:
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 2 == 0:
        train_loss, train_acc = eval_model(model, train_dataloader, loss_fn)

        test_loss, test_acc = eval_model(model, test_dataloader, loss_fn)

        print(
            f"Epoch {epoch} | Train Loss {train_loss:.3f} | Train Acc {train_acc:.3f} | Test Loss {test_loss:.3f} | Test Acc {test_acc:.3f}"
        )