"""Basic logistic regression using PyTorch.

Basically just an exercise to see if PyTorch is at least as comfortable."""

import argparse

import torch
from torch import nn

import data.epsilon_torch as epsilon
import results


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=20,
    help="Number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=64,
    help="Batch size")
args = parser.parse_args()

nepochs = args.epochs
batch_size = args.batch_size

results_dir = results.create_results_directory()
results.log_arguments(args, results_dir)

train_dataset = epsilon.EpsilonDataset(train=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = epsilon.EpsilonDataset(train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


class Logistic(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2000, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.stack(x)


def train(dataloader, model, loss_fn, optimizer):
    model.train()  # doesn't do anything here, but as a habit

    nbatches = len(train_dataloader)

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"[{batch}/{nbatches}] loss: {loss.item()}...", end='\r')

    return loss.item()


def test(dataloader, model, loss_fn):
    model.eval()  # doesn't do anything here, but as a habit
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            correct += ((pred > 0.5) == y).type(torch.float).sum().item()

    test_loss = total_loss / len(test_dataloader)
    accuracy = correct / len(test_dataset)

    return test_loss, accuracy


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Logistic().to(device)
print("model:", model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

csv_logfile = results_dir / 'training.csv'
csv_logger = results.CsvLogger(csv_logfile)

for i in range(nepochs):
    print(f"Epoch {i}...", end='\r')
    train_loss = train(train_dataloader, model, loss_fn, optimizer)

    test_loss, accuracy = test(test_dataloader, model, loss_fn)
    print(f"Epoch {i}: training loss {train_loss}, test loss {test_loss}, accuracy {accuracy}")
    csv_logger.log(i, {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'accuracy': accuracy,
    })


# Evaluation

test_loss, accuracy = test(test_dataloader, model, loss_fn)
results.log_evaluation({
    'test_loss': test_loss,
    'accuracy': accuracy,
}, results_dir)
