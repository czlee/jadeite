"""Basic logistic regression, but with federated averaging, using the vanilla
TensorFlow library, not tensorflow-federated.

Client data is divided equally among nodes without shuffling.

This is basically just an exercise, like a translation of fedavg-torch.py back
into TensorFlow.
"""

import argparse

import tensorflow as tf

import data.epsilon_tf as epsilon
import results


parser = argparse.ArgumentParser()
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

train_dataset = epsilon.train_dataset().batch(batch_size)
client_shards = [train_dataset.shard(nclients, i) for i in range(nclients)]
test_dataset = epsilon.test_dataset().batch(batch_size)


# Model

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2000,)),
    ])


# Training

def client_train(dataset, model, loss_fn, optimizer):
    for x, y in dataset:
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = loss_fn(y, pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return loss


def server_aggregate(global_model, client_models):
    """Aggregates client models by just taking the mean, a.k.a.
    federated averaging."""
    client_weights = [model.get_weights() for model in client_models]
    new_weights = [
        tf.math.reduce_mean(tf.stack(weights, axis=0), axis=0)
        for weights in zip(*client_weights)
    ]
    for model in client_models:
        model.set_weights(new_weights)


def test(dataset, model, loss_fn, accuracy_fn):
    test_losses = []
    accuracy_fn.reset_state()
    for x, y in dataset:
        pred = model(x)
        accuracy_fn.update_state(y, pred)
        test_losses.append(loss_fn(y, pred))

    test_loss = tf.math.reduce_mean(test_losses)
    accuracy = accuracy_fn.result().numpy()
    return test_loss, accuracy


global_model = create_model()
client_models = [create_model() for i in range(nclients)]
for model in client_models:
    model.set_weights(global_model.get_weights())
model.summary()

loss_fn = tf.keras.losses.BinaryCrossentropy()
accuracy_fn = tf.keras.metrics.BinaryAccuracy()
client_optimizers = [tf.keras.optimizers.SGD(learning_rate=args.lr_client) for i in range(nclients)]

csv_logfile = results_dir / 'training.csv'
csv_logger = results.CsvLogger(csv_logfile, index_field='round')

for r in range(nrounds):

    records = {}

    # train clients, just one epoch
    clients = zip(client_shards, client_models, client_optimizers)
    for i, (dataset, model, optimizer) in enumerate(clients):
        loss = client_train(dataset, model, loss_fn, optimizer)
        print(f"Client {i} of {nclients}: loss {loss}")
        records[f"train_loss_client{i}"] = loss

    server_aggregate(global_model, client_models)

    test_loss, accuracy = test(test_dataset, global_model, loss_fn, accuracy_fn)
    print(f"Epoch {r}: test loss {test_loss}, accuracy {accuracy}")
    records['test_loss'] = test_loss
    records['accuracy'] = accuracy
    csv_logger.log(r, records)


# Evaluation

global_model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
test_dataset = epsilon.test_dataset().batch(batch_size)
evaluation = global_model.evaluate(test_dataset, return_dict=True)
results.log_evaluation(evaluation, results_dir)
