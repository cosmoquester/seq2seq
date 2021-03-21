import logging
import os
import sys
from collections import Counter
from typing import Iterable

import tensorflow as tf
from tensorflow.keras import backend as K


def learning_rate_scheduler(num_epochs, max_learning_rate, min_learninga_rate=1e-7):
    lr_delta = (max_learning_rate - min_learninga_rate) / num_epochs

    def _scheduler(epoch, lr):
        return max_learning_rate - lr_delta * epoch

    return _scheduler


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def path_join(*paths: Iterable[str]) -> str:
    """ Join paths to string local paths and google storage paths also """
    if paths[0].startswith("gs://"):
        return "/".join([path.strip("/") for path in paths])
    return os.path.join(*paths)


def get_device_strategy(device) -> tf.distribute.Strategy:
    """ Return tensorflow device strategy """
    # Use TPU
    if device == "TPU":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ["TPU_NAME"])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        return strategy

    # Use GPU
    if device == "GPU":
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) == 0:
            raise RuntimeError("Cannot find GPU!")
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        if len(devices) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

        return strategy

    # Use CPU
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def n_gram_precision(true_tokens, pred_tokens, n):
    true_n_grams = []
    pred_n_grams = []

    for i in range(len(true_tokens) - n + 1):
        true_n_grams.append(tuple(true_tokens[i : i + n]))
    for i in range(len(pred_tokens) - n + 1):
        pred_n_grams.append(tuple(pred_tokens[i : i + n]))

    true_n_gram_counter = Counter(true_n_grams)
    pred_n_gram_counter = Counter(pred_n_grams)

    correct = 0
    for true_n_gram in true_n_gram_counter:
        correct += min(pred_n_gram_counter[true_n_gram], true_n_gram_counter[true_n_gram])
    return correct / len(true_n_grams)


def calculat_bleu_score(true_tokens, pred_tokens):
    n_gram_score = 1.0
    for n in range(1, 5):
        n_gram_score *= n_gram_precision(true_tokens, pred_tokens, n)
    n_gram_score **= 0.25

    brevity_penalty = min(1.0, len(pred_tokens) / len(true_tokens))

    return n_gram_score * brevity_penalty
