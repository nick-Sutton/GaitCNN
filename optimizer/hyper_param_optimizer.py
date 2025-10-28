import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def define_model():
    pass

def get_dataset():
    pass

def objective(trail):
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST(
        './data', train=True, download=True, transform=transform), batch_size=32, shuffle=True)

    model = Net(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return loss.item()
