from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

from token_and_position_embedding import TokenAndPositionEmbedding
from transformer_block import TransformerBlock

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Attention layer implemented for using it inside the AttentionNES
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W)+self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(Attention, self).get_config()


# AttentionNES: Class used for training the music generator
#     - input_shape: used for the Input layer
#     - vocabulary_sizes: size of each (musical) word vocabulary; it is important for the Embedding layer
class AttentionNES:

    def __init__(self, input_shapes, vocabulary_size):

        # Compiling the model
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        use_transformer = True
        latent_dim = 128

        if use_transformer:
            input = layers.Input(shape=input_shapes, name='input')
            embedding_layer = TokenAndPositionEmbedding(20000, 20000, latent_dim)
            x = embedding_layer(input)
            transformer_block = TransformerBlock(latent_dim, num_heads=3, ff_dim=64)
            x = transformer_block(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(32, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            output = layers.Dense(vocabulary_size + 1, activation="softmax", name='output')(x)
        else:
            # Intput layers
            input = layers.Input(shape=input_shapes, name='input')
            embedding_layer = layers.Embedding(vocabulary_size + 2, latent_dim, name='embed', mask_zero=True)(input)
            x = layers.LSTM(256, return_sequences=True)(embedding_layer)
            x = layers.Dropout(0.2)(x)
            x = layers.LSTM(256, return_sequences=True)(x)
            att_out = Attention()(x)
            output = layers.Dense(vocabulary_size + 1, activation='softmax', name='output')(att_out)

        # Creating and compiling the model
        self.loss = "sparse_categorical_crossentropy"
        self.model = keras.Model(inputs=input, outputs=output)
        self.model.compile(opt, loss=self.loss, metrics=["accuracy"])

        # Setting checkpoint
        self.checkpoint_path = os.path.join('models', 'attention_nes', 'cp.ckpt')
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
                                 epochs=250,
                                 validation_data=(x_test, y_test),
                                 callbacks=[cp_callback])

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.show()
