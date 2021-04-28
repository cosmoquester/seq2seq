from functools import partial

import tensorflow as tf
import tensorflow_text as text


def get_dataset(dataset_file_path: str, tokenizer: text.SentencepieceTokenizer, auto_encoding: bool):
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path:
        - if auto_encoding, text dataset file path just containing text lines.
        - else, tsv dataset file path. formed (sentence1, sentence2) without header.
    :param tokenizer: SentencepieceTokenizer instance.
    :param auto_encoding: whether to use text lines dataset for auto encoding.
                            If true, open dataset files as txt and a lines is an example for auto encoding.
    """

    @tf.function
    def tokenize_fn(source_text, target_text):
        # Tokenize & Add bos, eos
        source_tokens = tokenizer.tokenize(source_text)
        target_tokens = tokenizer.tokenize(target_text)
        return source_tokens, target_tokens

    if auto_encoding:
        duplicate = tf.function(lambda text: (text, text))
        dataset = tf.data.TextLineDataset(
            dataset_file_path,
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
        ).map(duplicate)
    else:
        dataset = tf.data.experimental.CsvDataset(dataset_file_path, [tf.string, tf.string], field_delim="\t")

    return dataset.map(tokenize_fn)


def get_tfrecord_dataset(dataset_file_path: str) -> tf.data.Dataset:
    """ Read TFRecord dataset file and construct tensorflow dataset """

    decompose = tf.function(
        lambda serialized_example: (
            tf.io.parse_tensor(serialized_example[0], tf.int32),
            tf.io.parse_tensor(serialized_example[1], tf.int32),
        )
    )
    dataset = (
        tf.data.TFRecordDataset(dataset_file_path, "GZIP")
        .map(partial(tf.io.parse_tensor, out_type=tf.string))
        .map(decompose)
    )
    return dataset


@tf.function
def make_train_examples(source_tokens: tf.Tensor, target_tokens: tf.Tensor):
    """ Make training examples from source and target tokens. """
    # Make training example
    num_examples = tf.shape(target_tokens)[0] - 1

    # [NumExamples, EncoderSequence]
    encoder_input = tf.repeat([source_tokens], repeats=[num_examples], axis=0)
    # [NumExamples, DecoderSequence]
    decoder_input = target_tokens * tf.sequence_mask(tf.range(1, num_examples + 1), num_examples + 1, tf.int32)
    # [NumExamples]
    labels = target_tokens[1:]

    return (encoder_input, decoder_input), labels
