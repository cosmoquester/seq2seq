import argparse
import yaml
import sys

import tensorflow as tf
import tensorflow_text as text

from seq2seq.data import get_dataset, get_tfrecord_dataset, make_train_examples
from seq2seq.model import create_model
from seq2seq.utils import LRScheduler, get_device_strategy, get_logger, path_join

# fmt: off
parser = argparse.ArgumentParser("This is script to train seq2seq model")
file_paths = parser.add_argument_group("File Paths")
file_paths.add_argument("--model-name", type=str, default="RNNSeq2SeqWithAttention", help="Seq2seq model name")
file_paths.add_argument("--model-config-path", type=str, default="resources/configs/rnn.yml", help="model config file")
file_paths.add_argument("--dataset-path", required=True, help="a text file or multiple files ex) *.txt")
file_paths.add_argument("--pretrained-model-path", type=str, default=None, help="pretrained model checkpoint")
file_paths.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
file_paths.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model")

training_parameters = parser.add_argument_group("Training Parameters")
training_parameters.add_argument("--epochs", type=int, default=10)
training_parameters.add_argument("--steps-per-epoch", type=int, default=None)
training_parameters.add_argument("--learning-rate", type=float, default=2e-4)
training_parameters.add_argument("--min-learning-rate", type=float, default=1e-5)
training_parameters.add_argument("--warmup-steps", type=int)
training_parameters.add_argument("--warmup-rate", type=float, default=0.06)
training_parameters.add_argument("--batch-size", type=int, default=512)
training_parameters.add_argument("--dev-batch-size", type=int, default=512)
training_parameters.add_argument("--num-total-dataset", type=int, default=1000000)
training_parameters.add_argument("--num-dev-dataset", type=int, default=30000)
training_parameters.add_argument("--shuffle-buffer-size", type=int, default=100000)
training_parameters.add_argument("--prefetch-buffer-size", type=int, default=1000)
training_parameters.add_argument("--max-sequence-length", type=int, default=256)

other_settings = parser.add_argument_group("Other settings")
other_settings.add_argument("--tensorboard-update-freq", type=int, help='log losses and metrics every after this value step')
other_settings.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
other_settings.add_argument("--auto-encoding", action="store_true", help="train by auto encoding with text lines dataset")
other_settings.add_argument("--use-tfrecord", action="store_true", help="train using tfrecord dataset")
other_settings.add_argument("--debug-nan-loss", action="store_true", help="Trainin with this flag, print the number of Nan loss (not supported on TPU)")
other_settings.add_argument("--device", type=str, default="CPU", choices= ["CPU", "GPU", "TPU"], help="device to train model")
other_settings.add_argument("--max-over-sequence-policy", type=str, default="filter", choices=["filter", "slice"], help="Policy for sequences of which length is over the max")
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

    logger = get_logger(__name__)

    if args.mixed_precision:
        mixed_type = "mixed_bfloat16" if args.device == "TPU" else "mixed_float16"
        policy = tf.keras.mixed_precision.experimental.Policy(mixed_type)
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("Use Mixed Precision FP16")

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.model_config_path, path_join(args.output_path, "model_config.yml"))

    # Construct Dataset
    dataset_files = tf.io.gfile.glob(args.dataset_path)
    if not dataset_files:
        logger.error("[Error] Dataset path is invalid!")
        sys.exit(1)

    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    with strategy.scope():
        filter_fn = tf.function(
            lambda source_tokens, target_tokens: tf.math.logical_and(
                tf.size(source_tokens) < args.max_sequence_length, tf.size(target_tokens) < args.max_sequence_length
            )
        )
        slice_fn = tf.function(
            lambda source_tokens, target_tokens: (
                source_tokens[: args.max_sequence_length],
                target_tokens[: args.max_sequence_length],
            )
        )

        if args.use_tfrecord:
            dataset = get_tfrecord_dataset(dataset_files)
        else:
            dataset = get_dataset(dataset_files, tokenizer, args.auto_encoding)

        # Filter or Slice
        if args.max_over_sequence_policy == "filter":
            dataset = dataset.filter(filter_fn)
        else:
            dataset = dataset.map(slice_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = (
            dataset.shuffle(args.shuffle_buffer_size)
            .map(make_train_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .unbatch()
        )

        train_dataset = (
            dataset.skip(args.num_dev_dataset)
            .padded_batch(args.batch_size, (([args.max_sequence_length], [args.max_sequence_length]), ()))
            .prefetch(args.prefetch_buffer_size)
        )
        dev_dataset = dataset.take(args.num_dev_dataset).padded_batch(
            args.dev_batch_size, (([args.max_sequence_length], [args.max_sequence_length]), ())
        )

        if args.steps_per_epoch:
            train_dataset = train_dataset.repeat()
            logger.info("Repeat dataset")

        # Model Initialize
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model = create_model(args.model_name, yaml.load(f, yaml.SafeLoader))

        model((tf.keras.Input([None]), tf.keras.Input([None])))
        model.summary()

        # Load pretrained model
        if args.pretrained_model_path:
            model.load_weights(args.pretrained_model_path)
            logger.info("Loaded weights of model")

        # Model Compile
        total_steps = (args.num_total_dataset - args.num_dev_dataset) // args.batch_size
        learning_rate = LRScheduler(
            total_steps, args.learning_rate, args.min_learning_rate, args.warmup_rate, args.warmup_steps
        )
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            if not args.debug_nan_loss
            else sparse_categorical_crossentropy,
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
            ],
        )
        logger.info("Finished training!")
