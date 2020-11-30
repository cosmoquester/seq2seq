import logging
import sys

import tensorflow as tf
from tensorflow.keras import backend as K


def learning_rate_scheduler(num_epochs, max_learning_rate, min_learninga_rate=1e-7):
    lr_delta = (max_learning_rate - min_learninga_rate) / num_epochs

    def _scheduler(epoch, lr):
        return lr - lr_delta

    return _scheduler


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger
