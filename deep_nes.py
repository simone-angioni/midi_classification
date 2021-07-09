from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# DeepNES: Class used for training the music generator
#     - input_shape: used for the Input layer
#     - vocabulary_sizes: size of each (musical) word vocabulary; it is important for the Embedding layer
class DeepNES:

    def __init__(self, input_shapes, vocabulary_sizes):

        # Compiling the model
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

        latent_dim = 128

        # Intput layers
        input1 = layers.Input(shape=input_shapes[0], name='input1')
        input2 = layers.Input(shape=input_shapes[1], name='input2')
        input3 = layers.Input(shape=input_shapes[2], name='input3')
        inputs = [input1, input2, input3]

        # Embedding layers
        embedding_layer1 = layers.Embedding(vocabulary_sizes[0] + 1, latent_dim, name='embed1')(input1)
        embedding_layer2 = layers.Embedding(vocabulary_sizes[1] + 1, latent_dim, name='embed2')(input2)
        embedding_layer3 = layers.Embedding(vocabulary_sizes[2] + 1, latent_dim, name='embed3')(input3)

        # Instrument 1 column
        x1 = layers.LSTM(latent_dim, return_sequences=True)(embedding_layer1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(latent_dim, return_sequences=True)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.LSTM(int((latent_dim + vocabulary_sizes[0]) / 2), return_sequences=False)(x1)
        x1 = layers.Dropout(0.2)(x1)
        x1 = layers.Dense(vocabulary_sizes[0], activation='relu')(x1)

        # Instrument 2 column
        x2 = layers.LSTM(latent_dim, return_sequences=True)(embedding_layer2)
        x2 = layers.Dropout(0.2)(x2)
        x2 = layers.LSTM(latent_dim, return_sequences=True)(x2)
        x2 = layers.Dropout(0.2)(x2)
        x2 = layers.LSTM(int((latent_dim + vocabulary_sizes[0]) / 2), return_sequences=False)(x2)
        x2 = layers.Dropout(0.2)(x2)
        x2 = layers.Dense(vocabulary_sizes[0], activation='relu')(x2)

        # Instrument 3 column
        x3 = layers.LSTM(latent_dim, return_sequences=True)(embedding_layer3)
        x3 = layers.Dropout(0.2)(x3)
        x3 = layers.LSTM(latent_dim, return_sequences=True)(x3)
        x3 = layers.Dropout(0.2)(x3)
        x3 = layers.LSTM(int((latent_dim + vocabulary_sizes[0]) / 2), return_sequences=False)(x3)
        x3 = layers.Dropout(0.2)(x3)
        x3 = layers.Dense(vocabulary_sizes[0], activation='relu')(x3)

        # Concatenation layers
        concat1 = layers.concatenate([x1, x2, x3])
        concat2 = layers.concatenate([x1, x2, x3])
        concat3 = layers.concatenate([x1, x2, x3])

        # Output layers
        output1 = layers.Dense(vocabulary_sizes[0], activation='softmax', name='output1')(concat1)
        output2 = layers.Dense(vocabulary_sizes[1], activation='softmax', name='output2')(concat2)
        output3 = layers.Dense(vocabulary_sizes[2], activation='softmax', name='output3')(concat3)
        outputs = [output1, output2, output3]

        # Creating and compiling the model
        self.loss = "sparse_categorical_crossentropy"
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(opt, loss=[self.loss, self.loss, self.loss], metrics=["accuracy"])

        # Setting checkpoint
        self.checkpoint_path = os.path.join('models', 'deep_nes', 'cp.ckpt')
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

    def fit(self, x_train, x_test, y_train, y_test):
        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

        history = self.model.fit(x_train, y_train,
                                 batch_size=128,
                                 epochs=180,
                                 validation_data=(x_test, y_test),
                                 callbacks=[cp_callback])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()
