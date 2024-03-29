from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Layer, LayerNormalization, SimpleRNN


class BiRNN(Layer):
    """
    Custom Bi-directional RNN Wrapper because of issue.
    https://github.com/tensorflow/tensorflow/issues/48880

    Arguments:
        rnn_class: RNN class, the class of SimpleRNN, LSTM, GRU.
        units: Integer, the hidden dimension size of seq2seq rnn.
        dropout: Float, dropout rate.
        recurrent_dropout: Float, reccurent dropout rate.

    Call arguments:
        inputs: [BatchSize, SequenceLength, HiddenDim]
        mask: [BatchSize, SequenceLength]
        initial_state: Tuple of [BatchSize, HiddenDim]

    Output Shape:
        output: `[BatchSize, SequenceLength, HiddenDim]`
        state: `[BatchSize, HiddenDim]`
    """

    def __init__(
        self,
        rnn_class: Union[SimpleRNN, LSTM, GRU],
        units: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs,
    ):
        super(BiRNN, self).__init__(**kwargs)

        self.forward_rnn = rnn_class(
            units=units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name="forward_rnn",
        )
        self.backward_rnn = rnn_class(
            units=units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            go_backwards=True,
            name="backward_rnn",
        )

    def call(
        self, inputs: tf.Tensor, mask: tf.Tensor, initial_state: Optional[tf.Tensor] = None, training: bool = False
    ) -> List:
        if initial_state is None:
            forward_states = None
            backward_states = None
        else:
            num_states = len(initial_state)
            forward_states = initial_state[: num_states // 2]
            backward_states = initial_state[num_states // 2 :]

        forward_output, *forward_states = self.forward_rnn(
            inputs, mask=mask, initial_state=forward_states, training=training
        )
        backward_output, *backward_states = self.backward_rnn(
            inputs, mask=mask, initial_state=backward_states, training=training
        )
        output = tf.concat([forward_output, tf.reverse(backward_output, axis=[1])], axis=-1)
        return [output] + forward_states + backward_states


class BahdanauAttention(Layer):
    """
    BahdanauAttention for seq2seq

    Arguments:
        hidden_dim: Integer, the hidden dimension size of seq2seq rnn.

    Call arguments:
        decoder_hidden: [BatchSize, HiddenDim]
        encoder_hiddens: [BatchSize, SequenceLength, HiddenDim]
        mask: [BatchSize, SequenceLength]
            Bool type tensor. The timesteps which has zero value will be ignored.


    Output Shape:
        2D tensor with shape:
            `[BatchSize, HiddenDim]`
    """

    def __init__(self, hidden_dim, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)

        self.Wh = Dense(hidden_dim, name="hidden_converter")
        self.Ws = Dense(hidden_dim, name="value_converter")
        self.V = Dense(1, name="score")

    def call(self, decoder_hidden: tf.Tensor, encoder_hiddens: tf.Tensor, mask: tf.Tensor):
        # [BatchSize, HiddenDim]
        query = self.Wh(decoder_hidden)
        # [BatchSize, SequenceLength, HiddenDim]
        values = self.Ws(encoder_hiddens)

        # [BatchSize, SequenceLength, 1]
        score = self.V(tf.nn.tanh(tf.expand_dims(query, axis=1) + values))
        score -= 1e9 * (1.0 - tf.cast(tf.expand_dims(mask, axis=2), score.dtype))
        attention = tf.nn.softmax(score, axis=1)

        # [BatchSize, HiddenDim]
        context = tf.reduce_sum(attention * encoder_hiddens, axis=1)
        return context


class PositionalEncoding(Layer):
    """
    Positional encoding for transformer

    Arguments:
        dim_embedding: Integer, the embedding dimension of transformer.
        positional_max_sequence: Integer, positional encoding max sequence.

    Call arguments:
        embedding: [BatchSize, SequenceLength, DimEmbedding]

    Output Shape:
        same as call argument `embedding`
    """

    def __init__(self, dim_embedding: int, positional_max_sequence: int = 1024, **kwargs):
        super(PositionalEncoding, self).__init__(trainable=False, **kwargs)

        angles = 1 / np.power(10000, (2 * (np.arange(dim_embedding) // 2)) / dim_embedding)
        angles = angles * np.arange(positional_max_sequence)[:, np.newaxis]

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self.pos_encode = tf.cast(angles[np.newaxis, ...], tf.float32)

    def call(self, embedding: tf.Tensor):
        sequence_length = tf.shape(embedding)[1]
        return embedding + tf.cast(self.pos_encode[:, :sequence_length, :], embedding.dtype)


class ScaledDotProductAttention(Layer):
    """
    Dot product attention for multihead attention.

    Arguments:
        dim_head: embedding dimension for a head, calculated as dim_embedding / num_heads.

    Call Arguemnts:
        query: [BatchSize, QuerySequenceLength, DimQuery-Key]
        key: [BatchSize, KeySequenceLength, DimQuery-Key]
        value: [BatchSize, KeySequenceLength, DimValue]
        mask: [BatchSize, QuerySequenceLength, KeySequenceLength]

    Output Shape:
        3D tensor with shape:
            `[BatchSize, SequenceLength, DimValue]
    """

    def __init__(self, dim_head, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

        self.Wq = Dense(dim_head, name="query")
        self.Wk = Dense(dim_head, name="key")
        self.Wv = Dense(dim_head, name="value")
        self.divider = tf.math.sqrt(tf.cast(dim_head, tf.float32))

    def call(self, query, key, value, mask=None):
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        # [BatchSize, QuerySequenceLength, KeySequenceLength]
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True) / tf.cast(self.divider, key.dtype)

        if mask is not None:
            scaled_attention_logits += tf.cast(mask * -1e9, scaled_attention_logits.dtype)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output


class MultiHeadAttention(Layer):
    """
    Multihead attention for transformer.

    Arguments:
        dim_embedding: Integer, model internal dimension.
        num_heads: Integer, the number of heads.

    Call Arguments:
        query: [BatchSize, SequenceLength, DimEmbedding]
        key: [BatchSize, SequenceLength, DimEmbedding]
        value: [BatchSize, SequenceLength, DimEmbedding]
        mask: [BatchSize, SequenceLength, SequenceLength]

    Output Shape:
        3D tensor with shape:
            `[BatchSize, SequenceLength, DimEmbedding]
    """

    def __init__(self, dim_embedding, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert dim_embedding % num_heads == 0, "`dim_embedding` should be divided by `num_heads`"

        self.attentions = [
            ScaledDotProductAttention(dim_embedding // num_heads, name=f"scaled_dot_product_attention{i}")
            for i in range(1, num_heads + 1)
        ]
        self.dense = Dense(dim_embedding, name="linear_transform")

    def call(self, query, key, value, mask=None):
        batch_size, sequence_length, _ = tf.unstack(tf.shape(query), 3)

        outputs = tf.reshape(tf.constant([]), [batch_size, sequence_length, 0])
        for attention in self.attentions:
            output = attention(query, key, value, mask)
            outputs = tf.concat([tf.cast(outputs, output.dtype), output], axis=-1)

        # [BatchSize, SequenceLength, DimEmbedding]
        outputs = self.dense(outputs)
        return outputs


class TransformerEncoderLayer(Layer):
    """
    Transformer encoder layer.

    Arguments:
        dim_embedding: Integer, model internal dimension.
        num_heads: Integer, the number of heads.
        dim_feedfoward: Integer, feedforward dimension.
        dropout: Float, dropout rate.
        activation: Integer, feedforward activation function.

    Call Arguments:
        input_embedding: [BatchSize, SequenceLength, DimEmbedding]
        mask: [BatchSize, SequenceLength]

    Output Shape:
        3D tensor with shape:
            `[BatchSize, SequenceLength, DimEmbedding]
    """

    def __init__(self, dim_embedding, num_heads, dim_feedfoward, dropout, activation="relu", **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)

        self.multihead_attention = MultiHeadAttention(dim_embedding, num_heads, name="self_attention")
        self.attention_layernorm = LayerNormalization(name="attention_layernorm")
        self.feedfoward_in = Dense(dim_feedfoward, activation=activation, name="feedforward_in")
        self.feedfoward_out = Dense(dim_embedding, name="feedforward_out")
        self.feedforward_layernorm = LayerNormalization(name="feedforward_layernorm")
        self.dropout = Dropout(dropout, name="dropout")

    def call(self, input_embedding, mask=None, training: bool = False):
        # [BatchSize, SequenceLength, DimEmbedding]
        attention_output = self.multihead_attention(input_embedding, input_embedding, input_embedding, mask)
        normalized_output = self.attention_layernorm(
            input_embedding + self.dropout(attention_output, training=training)
        )

        # [BatchSize, SequenceLength, DimEmbedding]
        output = self.feedfoward_out(self.feedfoward_in(normalized_output))
        output = self.feedforward_layernorm(normalized_output + self.dropout(output, training=training))

        return output


class TransformerDecoderLayer(Layer):
    """
    Transformer decoder layer.

    Arguments:
        dim_embedding: Integer, model internal dimension.
        num_heads: Integer, the number of heads.
        dim_feedfoward: Integer, feedforward dimension.
        dropout: Float, dropout rate.
        activation: Integer, feedforward activation function.

    Call Arguments:
        input_embedding: [BatchSize, SequenceLength, DimEmbedding]
        encoder_output: [BatchSize, SequenceLength, DimEmbedding]
        mask: [BatchSize, SequenceLength]

    Output Shape:
        3D tensor with shape:
            `[BatchSize, SequenceLength, DimEmbedding]
    """

    def __init__(self, dim_embedding, num_heads, dim_feedfoward, dropout, activation="relu", **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)

        self.self_attention = MultiHeadAttention(dim_embedding, num_heads, name="self_attention")
        self.attention_layernorm = LayerNormalization(name="attention_layernorm")
        self.encoder_decoder_attention = MultiHeadAttention(dim_embedding, num_heads, name="cross_attention")
        self.encoder_decoder_layernorm = LayerNormalization(name="attention_layernorm")
        self.feedfoward_in = Dense(dim_feedfoward, activation=activation, name="feedforward_in")
        self.feedfoward_out = Dense(dim_embedding, name="feedforward_out")
        self.feedforward_layernorm = LayerNormalization(name="feedforward_layernorm")
        self.dropout = Dropout(dropout, name="dropout")

    def call(self, input_embedding, encoder_output, encoder_mask=None, decoder_mask=None, training: bool = False):
        # [BatchSize, SequenceLength, DimEmbedding]
        attention_output = self.self_attention(input_embedding, input_embedding, input_embedding, decoder_mask)
        normalized_output = self.attention_layernorm(
            input_embedding + self.dropout(attention_output, training=training)
        )

        # [BatchSize, SequenceLength, DimEmbedding]
        attention_output = self.encoder_decoder_attention(
            normalized_output, encoder_output, encoder_output, encoder_mask
        )
        normalized_output = self.encoder_decoder_layernorm(
            normalized_output + self.dropout(attention_output, training=training)
        )

        # [BatchSize, SequenceLength, DimEmbedding]
        output = self.feedfoward_out(self.feedfoward_in(normalized_output))
        output = self.feedforward_layernorm(normalized_output + self.dropout(output, training=training))

        return output
