import argparse
import sys

import tensorflow as tf
import tensorflow_text as text
import yaml

from seq2seq.data import filter_example, get_dataset, get_tfrecord_dataset, make_train_examples, slice_example
from seq2seq.model import create_model
from seq2seq.utils import (
    LRScheduler,
    get_device_strategy,
    get_logger,
    path_join,
    set_mixed_precision,
    sparse_categorical_crossentropy_nan_debug,
)

# fmt: off
parser = argparse.ArgumentParser("This is script to train seq2seq model")
arg_group = parser.add_argument_group("File Paths")
arg_group.add_argument("--model-name", type=str, default="transformer", help="Seq2seq model name")
arg_group.add_argument("--model-config-path", type=str, default="resources/configs/transformer.yml", help="model config file")
arg_group.add_argument("--dataset-path", required=True, help="a text file or multiple files ex) *.txt")
arg_group.add_argument("--pretrained-model-path", type=str, default=None, help="pretrained model checkpoint")
arg_group.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
arg_group.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model")

arg_group = parser.add_argument_group("Training Parameters")
arg_group.add_argument("--epochs", type=int, default=10)
arg_group.add_argument("--steps-per-epoch", type=int, default=None)
arg_group.add_argument("--learning-rate", type=float, default=2e-4)
arg_group.add_argument("--min-learning-rate", type=float, default=1e-5)
arg_group.add_argument("--warmup-steps", type=int)
arg_group.add_argument("--warmup-rate", type=float, default=0.06)
arg_group.add_argument("--batch-size", type=int, default=512)
arg_group.add_argument("--dev-batch-size", type=int, default=512)
arg_group.add_argument("--num-total-dataset", type=int, default=1000000)
arg_group.add_argument("--num-dev-dataset", type=int, default=30000)
arg_group.add_argument("--shuffle-buffer-size", type=int, default=20000)
arg_group.add_argument("--prefetch-buffer-size", type=int, default=1000)
arg_group.add_argument("--max-sequence-length", type=int, default=256)

arg_group = parser.add_argument_group("Other settings")
arg_group.add_argument("--tensorboard-update-freq", type=int, help='log losses and metrics every after this value step')
arg_group.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
arg_group.add_argument("--auto-encoding", action="store_true", help="train by auto encoding with text lines dataset")
arg_group.add_argument("--use-tfrecord", action="store_true", help="train using tfrecord dataset")
arg_group.add_argument("--debug-nan-loss", action="store_true", help="Trainin with this flag, print the number of Nan loss (not supported on TPU)")
arg_group.add_argument("--device", type=str, default="CPU", choices= ["CPU", "GPU", "TPU"], help="device to train model")
arg_group.add_argument("--max-over-sequence-policy", type=str, choices=["filter", "slice"], help="Policy for sequences of which length is over the max")
# fmt: on


def main(args: argparse.Namespace):
    strategy = get_device_strategy(args.device)

    logger = get_logger(__name__)

    if args.mixed_precision:
        set_mixed_precision(args.device)
        logger.info("[+] Use Mixed Precision FP16")

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.model_config_path, path_join(args.output_path, "model_config.yml"))

    dataset_files = tf.io.gfile.glob(args.dataset_path)
    if not dataset_files:
        raise RuntimeError("Dataset path is invalid!")

    logger.info("[+] Load Tokenizer")
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    with strategy.scope():
        logger.info("[+] Load Dataset")
        if args.use_tfrecord:
            dataset = get_tfrecord_dataset(dataset_files)
        else:
            dataset = get_dataset(dataset_files, tokenizer, args.auto_encoding)

        # Apply policy for sequences whose length is over than max sequence length
        if args.max_over_sequence_policy == "filter":
            logger.info(f"[+] Filter examples whose sequence length is over than {args.max_sequence_length}")
            dataset = dataset.filter(filter_example(args.max_sequence_length))
        elif args.max_over_sequence_policy == "slice":
            logger.info(f"[+] Slice examples whose sequence length is over than {args.max_sequence_length}")
            dataset = dataset.map(
                slice_example(args.max_sequence_length), num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        elif args.device == "TPU":
            raise RuntimeError(f"You should set max-over-sequence-policy with TPU!")

        # Make into Training Examples
        dataset = dataset.shuffle(args.shuffle_buffer_size).map(
            make_train_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = dataset.skip(args.num_dev_dataset).unbatch()
        dev_dataset = dataset.take(args.num_dev_dataset).unbatch()

        if args.steps_per_epoch:
            logger.info("[+] Repeat dataset")
            train_dataset = train_dataset.repeat()

        # Batching
        pad_length = None if args.device != "TPU" else args.max_sequence_length
        train_dataset = train_dataset.padded_batch(args.batch_size, (([pad_length], [pad_length]), ())).prefetch(
            args.prefetch_buffer_size
        )
        dev_dataset = dev_dataset.padded_batch(args.dev_batch_size, (([pad_length], [pad_length]), ()))

        logger.info("[+] Initialze Model")
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model = create_model(args.model_name, yaml.load(f, yaml.SafeLoader))

        model((tf.keras.Input([None]), tf.keras.Input([None])))
        model.summary()

        if args.pretrained_model_path:
            logger.info("[+] Load weights of trained model")
            checkpoint = tf.train.Checkpoint(model)
            checkpoint.restore(args.pretrained_model_path).expect_partial()

        logger.info("[+] Compile Model")
        total_steps = (args.num_total_dataset - args.num_dev_dataset) // args.batch_size
        learning_rate = LRScheduler(
            total_steps, args.learning_rate, args.min_learning_rate, args.warmup_rate, args.warmup_steps
        )
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            if not args.debug_nan_loss
            else sparse_categorical_crossentropy_nan_debug,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        logger.info("[+] Start training")
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
        logger.info("[+] Finished training!")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
