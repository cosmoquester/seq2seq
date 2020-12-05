import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    """
    BahdanauAttention for seq2seq

    Arguments:
        hidden_dim: Integer, the hidden dimension size of seq2seq rnn.

    Call arguments:
        decoder_hidden: [BatchSize, HiddenDim]
        encoder_hiddens: [BatchSize, SequenceLength, HiddenDim]


    Output Shape:
        2D tensor with shape:
            `[BatchSize, HiddenDim]`
    """

    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()

        self.Wh = tf.keras.layers.Dense(hidden_dim, name="hidden_converter")
        self.Ws = tf.keras.layers.Dense(hidden_dim, name="value_converter")
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_hiddens):
        # [BatchSize, HiddenDim]
        query = self.Wh(decoder_hidden)
        # [BatchSize, SequenceLength, HiddenDim]
        values = self.Ws(encoder_hiddens)

        # [BatchSize, SequenceLength, 1]
        score = self.V(tf.nn.tanh(tf.expand_dims(query, axis=1) + values))
        attention = tf.nn.softmax(score, axis=1)

        # [BatchSize, HiddenDim]
        context = tf.reduce_sum(attention * encoder_hiddens, axis=1)
        return context
