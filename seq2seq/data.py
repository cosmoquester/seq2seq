import tensorflow as tf
import tensorflow_text as text


def get_dataset(dataset_file_path: str, tokenizer: text.SentencepieceTokenizer):
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path: tsv dataset file path. formed (sentence1, sentence2) without header
    """
    tokenize_fn = tf.function(lambda text, text2: (tokenizer.tokenize(text), tokenizer.tokenize(text2)))
    dataset = tf.data.experimental.CsvDataset(dataset_file_path, [tf.string, tf.string], field_delim="\t").map(
        tokenize_fn
    )
    return dataset
