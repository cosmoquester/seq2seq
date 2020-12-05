from typing import Tuple

import tensorflow as tf


def greedy_search(
    model: tf.keras.Model,
    encoder_input: tf.Tensor,
    bos_id: int,
    eos_id: int,
    max_sequence_length: int,
    pad_id: int = 0,
) -> tf.Tensor:
    """
    Generate sentences using decoder by beam searching.

    :param model: seq2seq model instance.
    :param encoder_input: seq2seq model inputs [BatchSize, EncoderSequenceLength].
    :param bos_id: bos id for decoding.
    :param eos_id: eos id for decoding.
    :param max_sequence_length: max sequence length of decoded sequences.
    :param pad_id: when a sequence is shorter thans other sentences, the back token ids of the sequence is filled pad id.
    :return: generated tensor shaped.
    """
    batch_size = tf.shape(encoder_input)[0]
    decoder_input = tf.fill([batch_size, 1], bos_id)
    sequence_lengths = tf.fill([batch_size, 1], max_sequence_length)
    is_ended = tf.zeros([batch_size], tf.bool)
    while tf.shape(decoder_input)[1] < max_sequence_length and not tf.reduce_all(is_ended):
        # [BatchSize, VocabSize]
        output = model((encoder_input, decoder_input))

        # [BatchSize, 1]
        new_tokens = tf.cast(tf.argmax(output, axis=1), tf.int32)
        new_tokens = tf.where(is_ended, 0, new_tokens)
        is_ended = tf.logical_or(is_ended, new_tokens == eos_id)
        new_tokens = tf.expand_dims(new_tokens, axis=1)

        # [BatchSize, DecoderSequenceLength + 1]
        decoder_input = tf.concat((decoder_input, new_tokens), axis=1)

    return decoder_input
