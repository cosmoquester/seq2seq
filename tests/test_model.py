import pytest
import tensorflow as tf

from seq2seq.model import RNNSeq2Seq


def test_model():
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
        use_bidirectional=True,
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
