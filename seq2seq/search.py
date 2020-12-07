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
    :return: generated tensor shaped. and ppl value of each generated sentences
    """
    batch_size = tf.shape(encoder_input)[0]
    decoder_input = tf.fill([batch_size, 1], bos_id)
    log_perplexity = tf.fill([batch_size, 1], 0.0)
    sequence_lengths = tf.fill([batch_size, 1], max_sequence_length)
    is_ended = tf.zeros([batch_size, 1], tf.bool)
    while tf.shape(decoder_input)[1] < max_sequence_length and not tf.reduce_all(is_ended):
        # [BatchSize, VocabSize]
        output = model((encoder_input, decoder_input))
        output = tf.nn.log_softmax(output, axis=1)

        # [BatchSize, 1]
        log_probs, new_tokens = tf.math.top_k(output)
        log_probs, new_tokens = tf.cast(log_probs, log_perplexity.dtype), tf.cast(new_tokens, tf.int32)
        log_perplexity += log_probs
        new_tokens = tf.where(is_ended, 0, new_tokens)
        is_ended = tf.logical_or(is_ended, new_tokens == eos_id)
        sequence_lengths = tf.where(new_tokens == eos_id, tf.shape(decoder_input)[1], sequence_lengths)

        # [BatchSize, DecoderSequenceLength + 1]
        decoder_input = tf.concat((decoder_input, new_tokens), axis=1)

    perplexity = tf.squeeze(
        tf.pow(tf.exp(log_perplexity), tf.cast(-1 / sequence_lengths, log_perplexity.dtype)), axis=1
    )
    return decoder_input, perplexity


def beam_search(
    model: tf.keras.Model,
    encoder_input: tf.Tensor,
    beam_size: int,
    bos_id: int,
    eos_id: int,
    max_sequence_length: int,
    pad_id: int = 0,
    alpha=1,
    beta=32,
) -> tf.Tensor:
    """
    Generate sentences using decoder by beam searching.

    :param model: seq2seq model instance.
    :param encoder_input: seq2seq model inputs [BatchSize, EncoderSequenceLength].
    :param beam_size: beam size for beam search.
    :param bos_id: bos id for decoding.
    :param eos_id: eos id for decoding.
    :param max_sequence_length: max sequence length of decoded sequences.
    :param pad_id: when a sequence is shorter thans other sentences, the back token ids of the sequence is filled pad id.
    :param alpha: length penalty control variable
    :param beta: length penalty control variable, meaning minimum length.
    :return: generated tensor shaped. and ppl value of each generated sentences
        decoder_input: (BatchSize, BeamSize, SequenceLength)
        perplexity: (BatchSize, BeamSize)
    """
    batch_size = tf.shape(encoder_input)[0]
    decoder_input = tf.fill([batch_size, 1], bos_id)
    log_perplexity = tf.fill([batch_size, 1], 0.0)

    def _to_sequence_lengths(decoder_single_input):
        eos_indices = tf.where(decoder_single_input == eos_id)
        if tf.size(eos_indices) == 0:
            return tf.size(decoder_single_input, tf.int32)
        return tf.cast(tf.math.reduce_min(eos_indices) + 1, tf.int32)

    def get_sequnce_lengths(decoder_input):
        original_shape = tf.shape(decoder_input)
        decoder_input = tf.reshape(decoder_input, (-1, original_shape[-1]))
        sequence_lengths = tf.map_fn(_to_sequence_lengths, decoder_input)
        return tf.reshape(sequence_lengths, original_shape[:-1])

    def has_eos(decoder_input):
        return tf.reduce_any(decoder_input == eos_id, axis=-1)

    while tf.shape(decoder_input)[1] < max_sequence_length and tf.reduce_any(tf.logical_not(has_eos(decoder_input))):
        # [BatchSize, VocabSize]
        output = model((encoder_input, decoder_input))
        output = tf.nn.log_softmax(output, axis=1)

        # [BatchSize, BeamSize] at first, [BatchSize * BeamSize, BeamSize] after second loops
        log_probs, new_tokens = tf.math.top_k(output, k=beam_size)

        # log_probs: [BatchSize, BeamSize] at first, [BatchSize, BeamSize ** 2] after second loops
        # new_tokens: [BatchSize, 1]at first, [BatchSize * BeamSize, 1] after second loops
        log_probs, new_tokens = tf.reshape(log_probs, [batch_size, -1]), tf.reshape(new_tokens, [-1, 1])
        is_end_sequences = tf.reshape(has_eos(decoder_input), tf.shape(log_probs))
        log_probs = tf.where(
            is_end_sequences,
            log_probs,
            log_probs + tf.cast(tf.repeat(log_perplexity, beam_size, axis=1), log_probs.dtype),
        )

        # Generate first token
        if tf.shape(decoder_input)[1] == 1:
            # [BatchSize * BeamSize, EncoderInputSequence]
            encoder_input = tf.repeat(encoder_input, beam_size, axis=0)

            # [BatchSize * BeamSize, 2]
            decoder_input = tf.concat([tf.fill([batch_size * beam_size, 1], bos_id), new_tokens], axis=1)
            log_perplexity = log_probs
            continue
        else:
            # [BatchSize * BeamSize, BeamSize, DecoderSequenceLength + 1]
            decoder_input = tf.reshape(
                tf.concat((tf.repeat(decoder_input, beam_size, axis=0), new_tokens), axis=1),
                [batch_size, beam_size * beam_size, -1],
            )

        length_penalty = tf.pow((1 + get_sequnce_lengths(decoder_input)) / (1 + beta), alpha)
        length_penalty = tf.cast(tf.reshape(length_penalty, log_probs.shape), log_probs.dtype)
        # [BatchSize, BeamSize]
        _, top_indices = tf.math.top_k(log_probs * length_penalty, k=beam_size)

        # [BatchSize * BeamSize, 2]
        indices_for_decoder_input = tf.concat(
            [
                tf.reshape(tf.repeat(tf.range(batch_size), beam_size), [batch_size * beam_size, 1]),
                tf.reshape(top_indices, [batch_size * beam_size, 1]),
            ],
            axis=1,
        )

        # [BatchSize * BeamSize, DecoderSequenceLength]
        decoder_input = tf.gather_nd(decoder_input, indices_for_decoder_input)
        log_perplexity = tf.reshape(tf.gather_nd(log_probs, indices_for_decoder_input), [batch_size, beam_size ** 2])

    decoder_input = tf.reshape(decoder_input, [batch_size, beam_size, -1])
    sequence_lengths = get_sequnce_lengths(decoder_input)
    decoder_input = tf.where(tf.sequence_mask(sequence_lengths, tf.reduce_max(sequence_lengths)), decoder_input, pad_id)
    perplexity = tf.pow(tf.exp(log_perplexity), tf.cast(-1 / sequence_lengths, log_perplexity.dtype))

    return decoder_input, perplexity
