import os

import pytest
import tensorflow as tf

from seq2seq.data import get_dataset


@pytest.fixture(scope="session")
def data_path():
    return os.path.join((os.path.dirname(__file__)), "data")


def test_get_dataset(data_path):
    class PseudoTokenizer:
        def tokenize(self, text):
            return tf.strings.unicode_decode(text, "UTF-8")

    dataset = get_dataset(os.path.join(data_path, "sample_dataset.txt"), PseudoTokenizer())

    data = next(iter(dataset))
    batch_data = next(iter(dataset.padded_batch(2)))

    assert tf.shape(data[0]) == [5]
    assert tf.shape(data[1]) == [13]
    tf.debugging.assert_equal(tf.shape(batch_data[0]), [2, 11])
    tf.debugging.assert_equal(tf.shape(batch_data[1]), [2, 13])
