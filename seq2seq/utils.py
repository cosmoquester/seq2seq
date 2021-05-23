import logging
import os
import sys
from collections import Counter
from typing import Iterable, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Schedule learning rate linearly from max_learning_rate to min_learning_rate."""

    def __init__(
        self,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: Optional[float] = None,
        warmup_steps: Optional[int] = None,
    ):
        self.warmup_steps = max(int(total_steps * warmup_rate) if warmup_steps is None else warmup_steps, 1)
        self.increasing_delta = max_learning_rate / self.warmup_steps
        self.decreasing_delta = (max_learning_rate - min_learning_rate) / (total_steps - self.warmup_steps)
        self.max_learning_rate = tf.cast(max_learning_rate, tf.float32)
        self.min_learning_rate = tf.cast(min_learning_rate, tf.float32)

    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        lr = tf.reduce_min([step * self.increasing_delta, self.max_learning_rate - step * self.decreasing_delta])
        return tf.reduce_max([lr, self.min_learning_rate])


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def path_join(*paths: Iterable[str]) -> str:
    """Join paths to string local paths and google storage paths also"""
    if paths[0].startswith("gs://"):
        return "/".join([path.strip("/") for path in paths])
    return os.path.join(*paths)


def set_mixed_precision(device: str):
    """Set mixed precision on"""
    mixed_type = "mixed_bfloat16" if device == "TPU" else "mixed_float16"
    policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
    tf.keras.mixed_precision.experimental.set_policy(policy)


def get_device_strategy(device) -> tf.distribute.Strategy:
    """Return tensorflow device strategy"""
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


def sparse_categorical_crossentropy_nan_debug(y_true, y_pred):
    pred_nan = tf.math.is_nan(y_pred)
    if tf.math.reduce_any(pred_nan):
        tf.print(
            "\nWarning:",
            "The",
            tf.size(tf.where(pred_nan)),
            "number of output values are Nan!\n",
            output_stream=sys.stderr,
        )

    loss = tf.losses.sparse_categorical_crossentropy(y_true, y_pred, True)
    is_nan = tf.math.is_nan(loss)
    if tf.math.reduce_any(is_nan):
        tf.print(
            "\nWarning:", "The", tf.size(tf.where(is_nan)), "number of losses are Nan!\n", output_stream=sys.stderr
        )
        loss = tf.boolean_mask(loss, tf.logical_not(is_nan))
    return loss
