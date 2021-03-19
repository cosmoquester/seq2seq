import argparse
import json
import sys

import tensorflow as tf
import tensorflow_text as text

from seq2seq.data import get_dataset, get_tfrecord_dataset
from seq2seq.model import MODEL_MAP
from seq2seq.utils import get_device_strategy, get_logger, learning_rate_scheduler, path_join

# fmt: off
parser = argparse.ArgumentParser("This is script to train seq2seq model")
file_paths = parser.add_argument_group("File Paths")
file_paths.add_argument("--model-name", type=str, default="RNNSeq2SeqWithAttention", help="Seq2seq model name")
file_paths.add_argument("--model-config-path", type=str, default="resources/configs/rnn.json", help="model config file")
file_paths.add_argument("--dataset-path", required=True, help="a text file or multiple files ex) *.txt")
file_paths.add_argument("--pretrained-model-path", type=str, default=None, help="pretrained model checkpoint")
file_paths.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
file_paths.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model")

training_parameters = parser.add_argument_group("Training Parameters")
training_parameters.add_argument("--epochs", type=int, default=10)
training_parameters.add_argument("--steps-per-epoch", type=int, default=None)
training_parameters.add_argument("--learning-rate", type=float, default=2e-4)
training_parameters.add_argument("--min-learning-rate", type=float, default=1e-8)
training_parameters.add_argument("--batch-size", type=int, default=512)
training_parameters.add_argument("--dev-batch-size", type=int, default=512)
training_parameters.add_argument("--num-dev-dataset", type=int, default=30000)
training_parameters.add_argument("--shuffle-buffer-size", type=int, default=100000)
training_parameters.add_argument("--prefetch-buffer-size", type=int, default=100000)
training_parameters.add_argument("--max-sequence-length", type=int, default=256)

other_settings = parser.add_argument_group("Other settings")
other_settings.add_argument("--tensorboard-update-freq", type=int, help='log losses and metrics every after this value step')
other_settings.add_argument("--disable-mixed-precision", action="store_false", dest="mixed_precision", help="Use mixed precision FP16")
other_settings.add_argument("--auto-encoding", action="store_true", help="train by auto encoding with text lines dataset")
other_settings.add_argument("--use-tfrecord", action="store_true", help="train using tfrecord dataset")
other_settings.add_argument("--device", type=str, default="CPU", help="device to train model")
# fmt: on


def sparse_categorical_crossentropy(y_true, y_pred):
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


if __name__ == "__main__":
    args = parser.parse_args()
    strategy = get_device_strategy(args.device)

    logger = get_logger()

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("Use Mixed Precision FP16")

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.model_config_path, path_join(args.output_path, "model_config.json"))

    # Construct Dataset
    dataset_files = tf.io.gfile.glob(args.dataset_path)
    if not dataset_files:
        logger.error("[Error] Dataset path is invalid!")
        sys.exit(1)

    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    flat_fn = tf.function(lambda inputs, labels: (tf.data.Dataset.from_tensor_slices((inputs, labels))))
    filter_fn = tf.function(
        lambda inputs, labels: tf.math.logical_and(
            tf.shape(inputs[0])[1] < args.max_sequence_length, tf.size(labels) < args.max_sequence_length
        )
    )

    dataset = (
        get_dataset(dataset_files, tokenizer, args.auto_encoding)
        if not args.use_tfrecord
        else get_tfrecord_dataset(dataset_files)
    )
    dataset = (
        dataset.filter(filter_fn)
        .shuffle(args.shuffle_buffer_size)
        .flat_map(flat_fn)
        .prefetch(args.prefetch_buffer_size)
    )
    train_dataset = dataset.skip(args.num_dev_dataset).padded_batch(args.batch_size)
    dev_dataset = dataset.take(args.num_dev_dataset).padded_batch(max(args.batch_size, args.dev_batch_size))

    if args.steps_per_epoch:
        train_dataset = train_dataset.repeat()
        logger.info("Repeat dataset")

    with strategy.scope():

        # Model Initialize
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model = MODEL_MAP[args.model_name](**json.load(f))

        model((tf.keras.Input([None]), tf.keras.Input([None])))
        model.summary()

        # Load pretrained model
        if args.pretrained_model_path:
            model.load_weights(args.pretrained_model_path)
            logger.info("Loaded weights of model")

        # Model Compile
        model.compile(
            optimizer=tf.optimizers.Adam(args.learning_rate),
            loss=sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        logger.info("Model compiling complete")
        logger.info("Start training")

    # Training
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                path_join(
                    args.output_path,
                    "models",
                    "model-{epoch}epoch-{val_loss:.4f}loss_{val_sparse_categorical_accuracy:.4f}acc.ckpt",
                ),
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                path_join(args.output_path, "logs"),
                update_freq=args.tensorboard_update_freq if args.tensorboard_update_freq else "batch",
            ),
            tf.keras.callbacks.LearningRateScheduler(
                learning_rate_scheduler(args.epochs, args.learning_rate, args.min_learning_rate), verbose=1
            ),
        ],
    )
    logger.info("Finished training!")
