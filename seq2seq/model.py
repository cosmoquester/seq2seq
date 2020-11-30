from typing import Optional

import tensorflow as tf


# Remove SampleModel and replace
class SampleModel(tf.keras.Model):
    def __init__(self, hidden_dim: int):
        super(SampleModel, self).__init__()

        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        output = self.dense2(self.dense1(inputs))
        return output
