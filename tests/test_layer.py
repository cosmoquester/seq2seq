import pytest
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN

from seq2seq.layer import (
    BahdanauAttention,
    BiRNN,
    MultiHeadAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


@pytest.mark.parametrize(
    "rnn_class,units,dropout,recurrent_dropout,batch_size,sequence_length",
    [
        (LSTM, 128, 0.1, 0.12, 32, 110),
        (GRU, 256, 0.8, 0.0, 63, 121),
        (SimpleRNN, 100, 0.1234, 0.4321, 11, 1),
    ],
)
def test_bi_rnn_shape(rnn_class, units, dropout, recurrent_dropout, batch_size, sequence_length):
    birnn = BiRNN(rnn_class, units, dropout, recurrent_dropout)

    inputs = tf.random.uniform([batch_size, sequence_length, units])
    output, *states = birnn(inputs)

    assert len(states) % 2 == 0
    tf.debugging.assert_equal(tf.shape(output), [batch_size, sequence_length, units * 2])
    tf.debugging.assert_equal(tf.shape(states[0]), [batch_size, units])


def test_bahdanau_attention_shape():
    batch_size = 4
    hidden_dim = 123
    sequence_length = 32

    inputs = (tf.random.normal((batch_size, hidden_dim)), tf.random.normal((batch_size, sequence_length, hidden_dim)))
    attention = BahdanauAttention(hidden_dim)

    output = attention(*inputs)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, hidden_dim])


def test_scaled_dotproduct_attention_shape():
    batch_size = 4
    dim_head = 44
    sequence_length = 32

    inputs = (
        tf.random.normal((batch_size, sequence_length, dim_head)),
        tf.random.normal((batch_size, sequence_length, dim_head)),
        tf.random.normal((batch_size, sequence_length, dim_head)),
    )
    attention = ScaledDotProductAttention(dim_head)

    output = attention(*inputs)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, sequence_length, dim_head])


def test_multihead_attention_shape():
    batch_size = 4
    dim_embedding = 44
    num_heads = 2
    sequence_length = 32

    inputs = (
        tf.random.normal((batch_size, sequence_length, dim_embedding)),
        tf.random.normal((batch_size, sequence_length, dim_embedding)),
        tf.random.normal((batch_size, sequence_length, dim_embedding)),
    )
    attention = MultiHeadAttention(dim_embedding, num_heads)

    output = attention(*inputs)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, sequence_length, dim_embedding])


def test_positional_encoding_shape():
    batch_size = 3
    dim_embedding = 111
    sequence_length = 22

    pos_encode = PositionalEncoding(dim_embedding, sequence_length)

    input_shape = [batch_size, sequence_length, dim_embedding]
    output = pos_encode(tf.random.normal(input_shape))
    tf.debugging.assert_equal(tf.shape(output), input_shape)


def test_transformer_encoder_layer_shape():
    batch_size = 4
    sequence_length = 33
    dim_embedding = 48
    num_heads = 2
    dim_feedforward = 128
    dropout = 0.1

    input_embedding = tf.random.normal((batch_size, sequence_length, dim_embedding))
    attention_mask = tf.random.uniform((batch_size, 1, sequence_length), 0, 2, tf.float32)
    encoder_layer = TransformerEncoderLayer(dim_embedding, num_heads, dim_feedforward, dropout)

    output = encoder_layer(input_embedding, attention_mask)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, sequence_length, dim_embedding])


def test_transformer_decoder_layer_shape():
    batch_size = 4
    source_sequence_length = 33
    target_sequence_length = 20
    encoder_dim_embedding = 44
    dim_embedding = 48
    num_heads = 2
    dim_feedforward = 128
    dropout = 0.1

    input_embedding = tf.random.normal((batch_size, target_sequence_length, dim_embedding))
    encoder_output = tf.random.normal((batch_size, source_sequence_length, encoder_dim_embedding))
    encoder_mask = tf.random.uniform((batch_size, 1, source_sequence_length), 0, 2, tf.float32)
    decoder_mask = tf.random.uniform((batch_size, 1, target_sequence_length), 0, 2, tf.float32)
    decoder_layer = TransformerDecoderLayer(dim_embedding, num_heads, dim_feedforward, dropout)

    output = decoder_layer(input_embedding, encoder_output, encoder_mask, decoder_mask)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, target_sequence_length, dim_embedding])
