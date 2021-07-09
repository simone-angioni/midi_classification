from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Classifier:

  def __init__(self, input_shape, vocabulary_size):
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    latent_dim = 64
    num_classes = 5

    input = layers.Input(shape=input_shape, name='input')
    x = layers.Embedding(vocabulary_size + 1, latent_dim, name='embed')(input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # Creating and compiling the model
    self.loss = "sparse_categorical_crossentropy"
    self.model = keras.Model(input, output)
    self.model.compile(opt, loss=self.loss, metrics=["accuracy"])

    # Setting checkpoint
    self.checkpoint_path = "clf_training_1/cp.ckpt"
    self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

  def load_model(self):
    self.model.load_weights(self.checkpoint_path)

  def fit(self, x_train, x_test, y_train, y_test):
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    history = self.model.fit(x_train, y_train, batch_size=256,
                             epochs=250,
                             validation_data=(x_test, y_test),
                             callbacks=[cp_callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

