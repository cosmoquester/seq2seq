import numpy as np
import pytest

from seq2seq.utils import LRScheduler


@pytest.mark.parametrize(
    "total_steps,learning_rate,min_learning_rate,warm_up_rate",
    [(10, 1.1, 0.0, 0.0), (33, 1e-5, 1e-7, 0.3), (100, 100, 0, 0.5)],
)
def test_learning_rate_scheduler(total_steps, learning_rate, min_learning_rate, warm_up_rate):
    fn = LRScheduler(total_steps, learning_rate, min_learning_rate, warm_up_rate)

    for i in range(total_steps):
        learning_rate = fn(i)
    np.isclose(learning_rate, min_learning_rate, 1e-10, 0)
