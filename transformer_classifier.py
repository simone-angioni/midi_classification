from transformer_block import TransformerBlock
from token_and_position_embedding import TokenAndPositionEmbedding
from sklearn.metrics import f1_score, accuracy_score

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import backend as K

physical_devices = tf.config.experimental.list_physical_devices('CPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class TransformerClassifier:

  def __init__(self, input_shape, vocabulary_size, maxlen=12050):
    # opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    vocabulary_size = vocabulary_size
    latent_dim = 128
    num_classes = 3

    input = layers.Input(shape=input_shape, name='input')
    x = layers.Embedding(vocabulary_size + 1, latent_dim, name='embed')(input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # input = layers.Input(shape=input_shape, name='input')
    # embedding_layer = TokenAndPositionEmbedding(maxlen, vocabulary_size, latent_dim)
    # x = embedding_layer(input)
    # transformer_block = TransformerBlock(latent_dim, num_heads=3, ff_dim=64)
    # x = transformer_block(x)
    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(0.2)(x)
    # output = layers.Dense(num_classes, activation="softmax", name='output')(x)

    # Creating and compiling the model
    self.loss = "sparse_categorical_crossentropy"
    self.model = keras.Model(input, output)
    self.model.compile(opt, loss=self.loss, metrics=["accuracy"])

    # Setting checkpoint
    self.checkpoint_path = "transformer_training_1/cp.ckpt"
    self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

  def load_model(self):
    self.model.load_weights(self.checkpoint_path)

  def fit(self, x_train, x_test, y_train, y_test):

    history = self.model.fit(x_train, y_train, batch_size=256,
                             epochs=100,
                             validation_data=(x_test, y_test))

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.show()
    #
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.show()

