from typing import Tuple

import tensorflow as tf


def greedy_search(
    model: tf.keras.Model,
    encoder_input: tf.Tensor,
    bos_id: int,
    eos_id: int,
    max_sequence_length: int,
) -> tf.Tensor:
    """
    Generate sentences using decoder by beam searching.

    :param model: seq2seq model instance.
    :param encoder_input: seq2seq model inputs [BatchSize, EncoderSequenceLength].
    :param eos_
    :return: generated tensor shaped
    """
    batch_size = tf.shape(encoder_input)[0]
    decoder_input = tf.fill((batch_size, 1), bos_id)
    is_ended = tf.zeros([batch_size], tf.bool)
    while tf.shape(decoder_input)[1] < max_sequence_length and not tf.reduce_all(is_ended):
        # [BatchSize, VocabSize]
        output = model((encoder_input, decoder_input))

        # [BatchSize, 1]
        new_tokens = tf.cast(tf.argmax(output, axis=1), tf.int32)
        is_ended = tf.logical_or(is_ended, new_tokens == eos_id)
        new_tokens = tf.expand_dims(new_tokens, axis=1)

        # [BatchSize, DecoderSequenceLength + 1]
        decoder_input = tf.concat((decoder_input, new_tokens), axis=1)

    return decoder_input
