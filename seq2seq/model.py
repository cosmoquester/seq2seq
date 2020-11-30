from typing import Optional

import tensorflow as tf


class BiLSTMSeq2Seq(tf.keras.Model):
    """
    Seq2seq model using BiLSTM.

    Arguments:
        hidden_dim: Integer, the hidden dimension size of SampleModel.

    Call arguments:
        encoder_tokens: A 3D tensor, with shape of `[BatchSize, EncoderSequenceLength]`.
                            all values are in [0, VocabSize).
        encoder_tokens: A 3D tensor, with shape of `[BatchSize, DecoderSequenceLength]`.
                            all values are in [0, VocabSize).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, VocabSize]
    """

    def __init__(self, vocab_size, hidden_dim):
        super(BiLSTMSeq2Seq, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_dim)
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_state=True))
        self.decoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim))
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, encoder_tokens: tf.Tensor, decoder_tokens: tf.Tensor, training: Optional[bool] = None):
        # [BatchSize, SequenceLength, VocabSize]
        encoder_embedding = self.embedding(encoder_tokens)
        decoder_embedding = self.embedding(decoder_tokens)

        # [BatchSize, SequenceLength, HiddenDim]
        encoded, *encoder_states = self.encoder(encoder_embedding)
        decoded = self.decoder(decoder_embedding, initial_state=encoder_states)

        # [BatchSize, VocabSize]
        output = self.dense(decoded)
        return output
