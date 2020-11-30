import numpy as np
import pytest

from sample_package.utils import learning_rate_scheduler


@pytest.mark.parametrize("num_epoch,learning_rate,min_learning_rate", [(1, 1.1, 0.0), (33, 1e-5, 1e-7), (100, 100, 0)])
def test_learning_rate_scheduler(num_epoch, learning_rate, min_learning_rate):
    fn = learning_rate_scheduler(num_epoch, learning_rate, min_learning_rate)

    for i in range(num_epoch):
        learning_rate = fn(i, learning_rate)
    np.isclose(learning_rate, min_learning_rate, 1e-10, 0)
