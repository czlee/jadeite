"""Logistic regression experiments."""

import data.epsilon as epsilon

import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2000,)),
])

model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])

nepochs = 20
batch_size = 64
train_dataset = epsilon.train_dataset().repeat(nepochs).batch(batch_size)
model.fit(train_dataset, epochs=nepochs, steps_per_epoch=epsilon.ntrain / batch_size)

test_dataset = epsilon.test_dataset().batch(batch_size)
results = model.evaluate(test_dataset, return_dict=True)
for key, value in results.items():
    print(f"{key}: {value}")
