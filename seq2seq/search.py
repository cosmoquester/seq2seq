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
    perplexity = tf.fill([batch_size, 1], 0.0)
    sequence_lengths = tf.fill([batch_size, 1], max_sequence_length)
    is_ended = tf.zeros([batch_size, 1], tf.bool)
    while tf.shape(decoder_input)[1] < max_sequence_length and not tf.reduce_all(is_ended):
        # [BatchSize, VocabSize]
        output = model((encoder_input, decoder_input))
        output = tf.nn.log_softmax(output, axis=1)

        # [BatchSize, 1]
        probs, new_tokens = tf.math.top_k(output)
        probs, new_tokens = tf.cast(probs, tf.float32), tf.cast(new_tokens, tf.int32)
        perplexity += probs
        new_tokens = tf.where(is_ended, 0, new_tokens)
        is_ended = tf.logical_or(is_ended, new_tokens == eos_id)
        sequence_lengths = tf.where(new_tokens == eos_id, tf.shape(decoder_input)[1], sequence_lengths)

        # [BatchSize, DecoderSequenceLength + 1]
        decoder_input = tf.concat((decoder_input, new_tokens), axis=1)

    perplexity = tf.squeeze(tf.exp(perplexity) ** tf.cast(-1 / sequence_lengths, tf.float32), axis=1)
    return decoder_input, perplexity
