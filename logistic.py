"""Basic logistic regression.

This is basically just an exercise, there's nothing even remotely fancy in here,
unless you count arguments and logging callbacks.
"""

import argparse

import tensorflow as tf

import data.epsilon as epsilon
import results


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=20,
    help="Number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=64,
    help="Batch size")
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

results_dir = results.create_results_directory()
results.log_arguments(args, results_dir)

train_logger = tf.keras.callbacks.CSVLogger(results_dir / "training.csv")
train_dataset = epsilon.train_dataset().repeat(epochs).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2000,)),
])
model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])

model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=epsilon.ntrain // batch_size,
    callbacks=[train_logger],
)

test_dataset = epsilon.test_dataset().batch(batch_size)
evaluation = model.evaluate(test_dataset, return_dict=True)
results.log_evaluation(evaluation, results_dir)
