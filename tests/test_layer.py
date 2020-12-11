import tensorflow as tf

from seq2seq.layer import (
    BahdanauAttention,
    MultiHeadAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


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

    input_embedding = tf.random.normal((batch_size, sequence_length, dim_embedding))
    attention_mask = tf.random.uniform((batch_size, sequence_length), 0, 2, tf.float32)
    encoder_layer = TransformerEncoderLayer(dim_embedding, num_heads, dim_feedforward)

    output = encoder_layer(input_embedding, attention_mask)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, sequence_length, dim_embedding])


def test_transformer_decoder_layer_shape():
    batch_size = 4
    sequence_length = 33
    encoder_dim_embedding = 44
    dim_embedding = 48
    num_heads = 2
    dim_feedforward = 128

    inputs = (
        tf.random.normal((batch_size, sequence_length, dim_embedding)),
        tf.random.normal((batch_size, sequence_length, encoder_dim_embedding)),
        tf.random.uniform((batch_size, sequence_length), 0, 2, tf.float32),
    )
    decoder_layer = TransformerDecoderLayer(dim_embedding, num_heads, dim_feedforward)

    output = decoder_layer(*inputs)
    tf.debugging.assert_equal(tf.shape(output), [batch_size, sequence_length, dim_embedding])
