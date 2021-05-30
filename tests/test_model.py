import pytest
import tensorflow as tf

from seq2seq.model import RNNSeq2Seq, RNNSeq2SeqWithAttention, TransformerSeq2Seq


def test_rnn_seq2seq():
    batch_size = 4
    encoder_sequence = 12
    decoder_sequence = 16
    vocab_size = 128
    hidden_dim = 64

    model = RNNSeq2Seq(
        cell_type="SimpleRNN",
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.15,
    )
    inputs = (
        tf.random.uniform((batch_size, encoder_sequence), maxval=vocab_size),
        tf.random.uniform((batch_size, decoder_sequence), maxval=vocab_size),
    )
    output = model(inputs)
    output_with_dropout = model(inputs, training=True)

    tf.debugging.assert_equal(tf.shape(output), (batch_size, vocab_size))
    tf.debugging.assert_none_equal(output, output_with_dropout)

    with pytest.raises(Exception):
        model(tf.constant([[30]]))


def test_rnn_seq2seq_with_attention():
    batch_size = 4
    encoder_sequence = 12
    decoder_sequence = 16
    vocab_size = 128
    hidden_dim = 64

    model = RNNSeq2SeqWithAttention(
        cell_type="SimpleRNN",
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.15,
    )
    inputs = (
        tf.random.uniform((batch_size, encoder_sequence), maxval=vocab_size),
        tf.random.uniform((batch_size, decoder_sequence), maxval=vocab_size),
    )
    output = model(inputs)
    output_with_dropout = model(inputs, training=True)

    tf.debugging.assert_equal(tf.shape(output), (batch_size, vocab_size))
    tf.debugging.assert_none_equal(output, output_with_dropout)

    with pytest.raises(Exception):
        model(tf.constant([[30]]))


def test_transformer_seq2seq():
    batch_size = 4
    encoder_sequence = 12
    decoder_sequence = 16

    vocab_size = 128
    dim_embedding = 64
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedfoward = 256
    activation = "relu"
    dropout = 0.15

    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        dim_embedding=dim_embedding,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedfoward=dim_feedfoward,
        activation="relu",
        dropout=dropout,
    )

    inputs = (
        tf.random.uniform((batch_size, encoder_sequence), maxval=vocab_size),
        tf.random.uniform((batch_size, decoder_sequence), maxval=vocab_size),
    )
    output = model(inputs)
    output_with_dropout = model(inputs, training=True)

    tf.debugging.assert_equal(tf.shape(output), (batch_size, vocab_size))
    tf.debugging.assert_none_equal(output, output_with_dropout)

    with pytest.raises(Exception):
        model(tf.constant([[30]]))
