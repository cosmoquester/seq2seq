import tensorflow as tf
import tensorflow_text as text


def get_dataset(dataset_file_path: str, tokenizer: text.SentencepieceTokenizer, auto_encoding: bool):
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path: tsv dataset file path. formed (sentence1, sentence2) without header.
    :param tokenizer: SentencepieceTokenizer instance.
    :param auto_encoding: whether to use text lines dataset for auto encoding.
                            If true, open dataset files as txt and a lines is an example for auto encoding.
    """

    @tf.function
    def mapping_fn(text1, text2):
        # Tokenize & Add bos, eos
        token1 = tokenizer.tokenize(text1)
        token2 = tokenizer.tokenize(text2)

        # Make training example
        num_examples = tf.shape(token2)[0] - 1

        # [NumExamples, EncoderSequence]
        encoder_input = tf.repeat([token1], repeats=[num_examples], axis=0)
        # [NumExamples, DecoderSequence]
        decoder_input = token2 * tf.sequence_mask(tf.range(1, num_examples + 1), num_examples + 1, tf.int32)
        # [NumExamples]
        labels = token2[1:]

        return (encoder_input, decoder_input), labels

    if auto_encoding:
        dataset = tf.data.TextLineDataset(dataset_file_path, num_parallel_reads=tf.data.experimental.AUTOTUNE).map(
            lambda text: (text, text)
        )
    else:
        dataset = tf.data.experimental.CsvDataset(dataset_file_path, [tf.string, tf.string], field_delim="\t")

    return dataset.map(mapping_fn)
