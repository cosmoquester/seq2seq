import tensorflow as tf
import tensorflow_text as text


def get_dataset(dataset_file_path: str, tokenizer: text.SentencepieceTokenizer, bos_id: int, eos_id: int):
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path: tsv dataset file path. formed (sentence1, sentence2) without header.
    :param tokenizer: SentencepieceTokenizer instance.
    :param bos_id: bos (begin of sentence) token id
    :param eos_id: eos (end of sentence) token id
    """

    @tf.function
    def mapping_fn(text1, text2):
        # Tokenize
        token1 = tokenizer.tokenize(text1)
        token2 = tokenizer.tokenize(text2)

        # Append Special tokens
        token1 = tf.concat([[bos_id], token1, [eos_id]], axis=0)
        token2 = tf.concat([[bos_id], token2, [eos_id]], axis=0)

        # Make training example
        num_examples = tf.shape(token2)[0] - 1
        encoder_input = tf.tile(tf.expand_dims(token1, axis=0), [num_examples, 1])
        decoder_input = token2 * tf.sequence_mask(tf.range(1, num_examples + 1), num_examples + 1, tf.int32)
        labels = token2 * tf.one_hot(tf.range(1, num_examples + 1), num_examples + 1, dtype=tf.int32)
        labels = tf.boolean_mask(labels, labels != 0)

        return (encoder_input, decoder_input), labels

    dataset = tf.data.experimental.CsvDataset(dataset_file_path, [tf.string, tf.string], field_delim="\t").map(
        mapping_fn
    )
    return dataset
