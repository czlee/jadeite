"""Basic logistic regression, but with federated averaging, using
tensorflow-federated.

Client data is divided equally among nodes without shuffling.

This is basically just an exercise, there's nothing fancy in here.
"""
# Chuan-Zheng Lee <czlee@stanford.edu>
# July 2021

import argparse

import tensorflow as tf
import tensorflow_federated as tff

import data.epsilon_tf as epsilon
import results


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-r", "--rounds", type=int, default=20,
    help="Number of rounds")
parser.add_argument("-b", "--batch-size", type=int, default=64,
    help="Batch size")
parser.add_argument("-c", "--clients", type=int, default=10,
    help="Number of clients")
args = parser.parse_args()

nrounds = args.rounds
nclients = args.clients
batch_size = args.batch_size

results_dir = results.create_results_directory()
results.log_arguments(args, results_dir)

# Training

train_dataset = epsilon.train_dataset().batch(batch_size)
client_shards = [train_dataset.shard(nclients, i) for i in range(nclients)]


def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2000,)),
    ])


def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_dataset.element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(),
)

state = iterative_process.initialize()

csv_logfile = results_dir / 'rounds.csv'
csv_logger = results.CsvLogger(csv_logfile)

for r in range(nrounds):
    print(f"Round {r} of {nrounds}...", end='\r')
    state, metrics = iterative_process.next(state, client_shards)
    csv_logger.log(r, metrics['train'])

csv_logger.close()

# Evaluation

test_model = create_keras_model()
test_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
state.model.assign_weights_to(test_model)
test_dataset = epsilon.test_dataset().batch(batch_size)
evaluation = test_model.evaluate(test_dataset, return_dict=True)
results.log_evaluation(evaluation, results_dir)
