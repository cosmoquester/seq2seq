import argparse
import csv
import sys

import tensorflow as tf
import tensorflow_text as text
import yaml

from seq2seq.model import create_model
from seq2seq.search import Searcher
from seq2seq.utils import get_device_strategy, get_logger, set_mixed_precision

# fmt: off
parser = argparse.ArgumentParser("This is script to inferece (generate sentence) with seq2seq model")
arg_group = parser.add_argument_group("File Paths")
arg_group.add_argument("--model-name", type=str, default="RNNSeq2SeqWithAttention", help="Seq2seq model name")
arg_group.add_argument("--model-config-path", type=str, default="resources/configs/rnn.yml", help="model config file")
arg_group.add_argument("--dataset-path", required=True, help="a text file or multiple files ex) *.txt")
arg_group.add_argument("--model-path", type=str, required=True, help="pretrained model checkpoint")
arg_group.add_argument("--output-path", default="output", help="output file path to save generated sentences")
arg_group.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model")

arg_group = parser.add_argument_group("Inference Parameters")
arg_group.add_argument("--batch-size", type=int, default=512)
arg_group.add_argument("--prefetch-buffer-size", type=int, default=100)
arg_group.add_argument("--max-sequence-length", type=int, default=256)
arg_group.add_argument("--pad-id", type=int, default=0, help="Pad token id when tokenize with sentencepiece")
arg_group.add_argument("--beam-size", type=int, default=0, help="not given, use greedy search else beam search with this value as beam size")

arg_group = parser.add_argument_group("Other settings")
arg_group.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
arg_group.add_argument("--save-pair", action="store_true", help="save result as the pairs of original and decoded sentences")
arg_group.add_argument("--device", type=str, default="CPU", choices= ["CPU", "GPU", "TPU"], help="device")
# fmt: on


def main(args: argparse.Namespace):
    strategy = get_device_strategy(args.device)

    logger = get_logger(__name__)

    if args.mixed_precision:
        set_mixed_precision(args.device)
        logger.info("Use Mixed Precision FP16")

    # Construct Dataset
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("").numpy().tolist()

    dataset_files = tf.io.gfile.glob(args.dataset_path)
    if not dataset_files:
        logger.error("[Error] Dataset path is invalid!")
        sys.exit(1)

    with strategy.scope():
        dataset = (
            tf.data.TextLineDataset(dataset_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
            .map(tokenizer.tokenize)
            .padded_batch(args.batch_size)
        ).prefetch(args.prefetch_buffer_size)

        # Model Initialize & Load pretrained model
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model = create_model(args.model_name, yaml.load(f, yaml.SafeLoader))
        model.load_weights(args.model_path)
        searcher = Searcher(model, args.max_sequence_length, bos_id, eos_id, args.pad_id)
        logger.info("Loaded weights of model")

        # Inference
        logger.info("Start Inference")
        outputs = []

        for batch_input in dataset:
            if args.beam_size > 0:
                batch_output = searcher.beam_search(batch_input, args.beam_size)
                batch_output = batch_output[0][:, 0, :].numpy()
            else:
                batch_output = searcher.greedy_search(batch_input)[0].numpy()
            outputs.extend(batch_output)
        outputs = [tokenizer.detokenize(output).numpy().decode("UTF8") for output in outputs]
        logger.info("Ended Inference, Start to save...")

        # Save file
        if args.save_pair:
            with open(args.dataset_path) as f, open(args.output_path, "w") as fout:
                wtr = csv.writer(fout, delimiter="\t")
                wtr.writerow(["EncodedSentence", "DecodedSentence"])

                for input_sentence, decoded_sentence in zip(f.read().split("\n"), outputs):
                    wtr.writerow((input_sentence, decoded_sentence))
            logger.info(f"Saved (original sentence,decoded sentence) pairs to {args.output_path}")

        else:
            with open(args.output_path, "w") as fout:
                for line in outputs:
                    fout.write(line + "\n")
            logger.info(f"Saved generated sentences to {args.output_path}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
