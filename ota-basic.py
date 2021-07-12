"""Basic logistic regression using an over-the-air model.

This is identical to federated averaging, except that an extra step is added in
the aggregation step. Rather than having the server aggregate the client models
directly, clients only send their model parameters. A separate function then
simulates the channel, and the server's training function is only allowed access
to the channel's output.

Client data is randomly divided equally among nodes.
"""
# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse
import json
from math import sqrt
from typing import List

import numpy as np
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
parser.add_argument("-n", "--clients", type=int, default=10,
    help="Number of clients, n")
parser.add_argument("-l", "--lr-client", type=float, default=1e-2,
    help="Learning rate at client")
parser.add_argument("-N", "--noise", type=float, default=1.0,
    help="Noise level (variance), σₙ²")
parser.add_argument("-P", "--power", type=float, default=1.0,
    help="Power level, P")
parser.add_argument("-B", "--parameter-radius", type=float, default=1.0,
    help="Parameter radius, B")
parser.add_argument("--data-per-client", type=int, default=None,
    help="Override the number of data points each client has (default: "
         "divide all data equally among clients)")
parser.add_argument("--small", action="store_true", default=False,
    help="Use a small dataset for testing")
parser.add_argument("--cpu", action="store_true", default=False,
    help="Force use of CPU (i.e. don't use CUDA)")
args = parser.parse_args()

nrounds = args.rounds
nclients = args.clients
batch_size = args.batch_size
noise = args.noise
power = args.power
parameter_radius = args.parameter_radius

results_dir = results.create_results_directory()
results.log_arguments(args, results_dir)

# Load data (only training data is split)

train_dataset = epsilon.EpsilonDataset(train=True, small=args.small)

if args.data_per_client is None:
    client_lengths = data.utils.divide_integer_evenly(len(train_dataset), nclients)
else:
    if args.data_per_client * nclients > len(train_dataset):
        print(f"There isn't enough data ({len(train_dataset)}) to get {args.data_per_client} "
              f"examples for each of {nclients} clients.")
        exit(2)
    client_lengths = [args.data_per_client] * nclients
    train_dataset = torch.utils.data.Subset(train_dataset, range(args.data_per_client * nclients))

client_datasets = torch.utils.data.random_split(train_dataset, client_lengths)
client_dataloaders = [
    torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for dataset in client_datasets
]

test_dataset = epsilon.EpsilonDataset(train=False, small=args.small)
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
    """Trains the client `model` through one pass of the given (client)
    `dataloader`. Returns the loss from the last batch. (I'm not sure how
    meaningful that is.) The `optimizer` should be set up with the model
    parameters from the given (client) `model`.
    """
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


def client_transmit(model) -> torch.Tensor:
    """Returns the symbols that should be transmitted from the client that is
    working with the given (client) `model`, as a row tensor. The symbols
    """
    state = model.state_dict()  # this is an OrderedDict
    values = torch.column_stack(tuple(state.values()))
    symbols = values * sqrt(power) / parameter_radius
    assert symbols.dim() == 2 and symbols.size()[0] == 1
    return symbols


def channel(client_symbols: List[torch.Tensor]) -> torch.Tensor:
    """Returns the channel output when the channel inputs are as provided in
    `client_symbols`, which should be a list of tensors.
    """
    all_symbols = torch.vstack(client_symbols)
    sum_symbols = torch.sum(all_symbols, dim=0, keepdim=True)
    noise_sample = torch.normal(0.0, sqrt(noise), size=sum_symbols.size()).to(device)
    output = sum_symbols + noise_sample
    assert output.dim() == 2 and output.size()[0] == 1
    return output


def disaggregate(model, tensor) -> dict:
    """Disaggregates the single row tensor into tensors for each state in the
    model."""
    flattened = tensor.flatten()
    new_state_dict = {}
    cursor = 0
    for key, value in model.state_dict().items():
        numel = value.numel()
        part = flattened[cursor:cursor + numel]
        new_state_dict[key] = part.reshape(value.size())
        cursor += numel
    assert cursor == flattened.numel()
    return new_state_dict


def server_receive(model, symbols):
    """Updates the global `model` given the `symbols` received from the channel.
    """
    scaled_symbols = symbols / nclients * parameter_radius / sqrt(power)
    new_state_dict = disaggregate(model, scaled_symbols)
    model.load_state_dict(new_state_dict)


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


device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
print(f"Using device: {device}")

global_model = Logistic().to(device)
client_models = [Logistic().to(device) for i in range(nclients)]
for model in client_models:  # initial sync with global model
    model.load_state_dict(global_model.state_dict())
print("model:", global_model)
print("state dict:", global_model.state_dict())

loss_fn = nn.functional.binary_cross_entropy
client_optimizers = [torch.optim.SGD(model.parameters(), lr=args.lr_client) for model in client_models]

csv_logfile = results_dir / 'training.csv'
csv_logger = results.CsvLogger(csv_logfile, index_field='round')

tx_powers = np.zeros((nrounds, nclients))

for r in range(nrounds):

    records = {}

    # train clients, just one epoch
    clients = zip(client_dataloaders, client_models, client_optimizers)
    tx_symbols = []
    for i, (dataloader, model, optimizer) in enumerate(clients):
        loss = client_train(dataloader, model, loss_fn, optimizer)
        print(f"Client {i} of {nclients}: loss {loss}")
        records[f"train_loss_client{i}"] = loss

        symbols = client_transmit(model)
        tx_symbols.append(symbols)
        tx_power = (symbols.square().sum().cpu() / symbols.numel()).numpy()
        records[f"tx_power_client{i}"] = tx_power
        tx_powers[r, i] = tx_power

    rx_symbols = channel(tx_symbols)
    server_receive(global_model, rx_symbols)

    test_loss, accuracy = test(test_dataloader, global_model, loss_fn)
    print(f"Epoch {r}: test loss {test_loss}, accuracy {accuracy}")
    records['test_loss'] = test_loss
    records['accuracy'] = accuracy
    csv_logger.log(r, records)

    model_file = results_dir / f"model_at_{r}.json"
    with open(model_file, 'w') as f:
        states = {k: v.tolist() for k, v in global_model.state_dict().items()}
        json.dump(states, f, indent=2)

# Evaluation

test_loss, accuracy = test(test_dataloader, global_model, loss_fn)
avg_power = tx_powers.mean()
eval_dict = {
    'test_loss': test_loss,
    'accuracy': accuracy,
    'avg_power': avg_power,
    'snr': avg_power / noise,
}
results.log_evaluation(eval_dict, results_dir)
print(json.dumps(eval_dict, indent=2))
