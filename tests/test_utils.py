import numpy as np
import pytest
import tensorflow as tf

from seq2seq.utils import LRScheduler, sparse_categorical_crossentropy_nan_debug


@pytest.mark.parametrize(
    "total_steps,learning_rate,min_learning_rate,warm_up_rate",
    [(10, 1.1, 0.0, 0.0), (33, 1e-5, 1e-7, 0.3), (100, 100, 0, 0.5)],
)
def test_learning_rate_scheduler(total_steps, learning_rate, min_learning_rate, warm_up_rate):
    fn = LRScheduler(total_steps, learning_rate, min_learning_rate, warm_up_rate)

    for i in range(total_steps):
        learning_rate = fn(i)
    np.isclose(learning_rate, min_learning_rate, 1e-10, 0)


def test_sparse_categorical_crossentropy_nan_debug():
    y_true = tf.random.uniform([44], 0, 100, dtype=tf.int32)
    y_pred = tf.random.normal([44, 100])
    y_pred_with_nan = tf.concat([y_pred, tf.fill([1, 100], float("nan"))], axis=0)
    y_true_with_nan = tf.concat([y_true, [0]], axis=0)

    loss1 = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss2 = sparse_categorical_crossentropy_nan_debug(y_true, y_pred)
    loss3 = sparse_categorical_crossentropy_nan_debug(y_true_with_nan, y_pred_with_nan)
    tf.debugging.assert_equal(loss1, loss2, loss3)
