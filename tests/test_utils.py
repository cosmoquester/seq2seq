import numpy as np
import pytest

from seq2seq.utils import learning_rate_scheduler


@pytest.mark.parametrize(
    "num_epoch,learning_rate,min_learning_rate,warm_up_rate",
    [(1, 1.1, 0.0, 0.0), (33, 1e-5, 1e-7, 0.3), (100, 100, 0, 0.5)],
)
def test_learning_rate_scheduler(num_epoch, learning_rate, min_learning_rate, warm_up_rate):
    fn = learning_rate_scheduler(num_epoch, learning_rate, min_learning_rate, warm_up_rate)

    for i in range(num_epoch):
        learning_rate = fn(i, learning_rate)
    np.isclose(learning_rate, min_learning_rate, 1e-10, 0)
