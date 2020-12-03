import logging
import os
import sys
from typing import Iterable

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


def path_join(*paths: Iterable[str]) -> str:
    """ Join paths to string local paths and google storage paths also """
    if paths[0].startswith("gs://"):
        return "/".join([path.strip("/") for path in paths])
    return os.path.join(*paths)


def get_device_strategy(device) -> tf.distribute.Strategy:
    """ Return tensorflow device strategy """
    # Use TPU
    if device.upper() == "TPU":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ["TPU_NAME"])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        return strategy

    # Use GPU
    if device.upper() == "GPU":
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
