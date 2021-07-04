"""Basic logistic regression using federated averaging using PyTorch.

Client data is randomly divided equally among nodes.

This is basically just an exercise, like a translation of fedavg.py but not
really.  Rather than use a federated learning library, it's based loosely on
this page:
https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029
"""
# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse

import torch
from torch import nn

import data.epsilon as epsilon
import data.utils
import results


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", "--rounds", type=int, default=20,
    help="Number of rounds")
parser.add_argument("-b", "--batch-size", type=int, default=64,
    help="Batch size")
parser.add_argument("-c", "--clients", type=int, default=10,
    help="Number of clients")
parser.add_argument("-l", "--lr-client", type=float, default=1e-2,
    help="Learning rate at client")
args = parser.parse_args()

nrounds = args.rounds
nclients = args.clients
batch_size = args.batch_size

results_dir = results.create_results_directory()
results.log_arguments(args, results_dir)

# Load data (only training data is split)

train_dataset = epsilon.EpsilonDataset(train=True)
client_lengths = data.utils.divide_integer_evenly(len(train_dataset), nclients)
client_datasets = torch.utils.data.random_split(train_dataset, client_lengths)
client_dataloaders = [
    torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for dataset in client_datasets
]

test_dataset = epsilon.EpsilonDataset(train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


# Model

class Logistic(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2000, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.stack(x)


# Training

# copied from logistic-torch.py's train()
def client_train(dataloader, model, loss_fn, optimizer):
    model.train()  # doesn't do anything here, but as a habit

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def server_aggregate(global_model, client_models):
    """Aggregates client models by just taking the mean, a.k.a.
    federated averaging."""

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        client_states = [model.state_dict()[k].float() for model in client_models]
        global_dict[k] = torch.stack(client_states, 0).mean(0)

    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


# copied from logistic-torch.py
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

global_model = Logistic().to(device)
client_models = [Logistic().to(device) for i in range(nclients)]
for model in client_models:  # initial sync with global model
    model.load_state_dict(global_model.state_dict())
print("model:", global_model)

loss_fn = nn.BCELoss()
client_optimizers = [torch.optim.SGD(model.parameters(), lr=args.lr_client) for model in client_models]

csv_logfile = results_dir / 'training.csv'
csv_logger = results.CsvLogger(csv_logfile, index_field='round')

for r in range(nrounds):

    records = {}

    # train clients, just one epoch
    clients = zip(client_dataloaders, client_models, client_optimizers)
    for i, (dataloader, model, optimizer) in enumerate(clients):
        loss = client_train(dataloader, model, loss_fn, optimizer)
        print(f"Client {i} of {nclients}: loss {loss}")
        records[f"train_loss_client{i}"] = loss

    server_aggregate(global_model, client_models)

    test_loss, accuracy = test(test_dataloader, global_model, loss_fn)
    print(f"Epoch {r}: test loss {test_loss}, accuracy {accuracy}")
    records['test_loss'] = test_loss
    records['accuracy'] = accuracy
    csv_logger.log(r, records)

# Evaluation

test_loss, accuracy = test(test_dataloader, global_model, loss_fn)
results.log_evaluation({
    'test_loss': test_loss,
    'accuracy': accuracy,
}, results_dir)
