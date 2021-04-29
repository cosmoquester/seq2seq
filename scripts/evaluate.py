import argparse
import json
import sys

import tensorflow as tf
import tensorflow_text as text
from tqdm import tqdm

from seq2seq.model import create_model
from seq2seq.search import Searcher
from seq2seq.utils import calculat_bleu_score, get_device_strategy, get_logger

# fmt: off
parser = argparse.ArgumentParser("This is script to inferece (generate sentence) with seq2seq model")
file_paths = parser.add_argument_group("File Paths")
file_paths.add_argument("--model-name", type=str, default="RNNSeq2SeqWithAttention", help="Seq2seq model name")
file_paths.add_argument("--model-config-path", type=str, default="resources/configs/rnn.json", help="model config file")
file_paths.add_argument("--dataset-path", required=True, help="a tsv file or multiple files ex) *.tsv")
file_paths.add_argument("--model-path", type=str, required=True, help="pretrained model checkpoint")
file_paths.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model")

inference_parameters = parser.add_argument_group("Inference Parameters")
inference_parameters.add_argument("--batch-size", type=int, default=512)
inference_parameters.add_argument("--prefetch-buffer-size", type=int, default=100)
inference_parameters.add_argument("--max-sequence-length", type=int, default=256)
inference_parameters.add_argument("--header", action="store_true", help="use this flag if dataset (tsv file) has header")
inference_parameters.add_argument("--beam-size", type=int, default=0, help="not given, use greedy search else beam search with this value as beam size")

other_settings = parser.add_argument_group("Other settings")
other_settings.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
other_settings.add_argument("--auto-encoding", action="store_true", help="evaluate by autoencoding performance dataset format is lines of texts (.txt)")
other_settings.add_argument("--device", type=str, default="CPU", help="device to train model")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()
    strategy = get_device_strategy(args.device)

    logger = get_logger(__name__)

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("Use Mixed Precision FP16")

    # Construct Dataset
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    with strategy.scope():
        dataset_files = tf.io.gfile.glob(args.dataset_path)
        if not dataset_files:
            logger.error("[Error] Dataset path is invalid!")
            sys.exit(1)
        if args.auto_encoding:
            scatter = lambda tokens: (tokens, tokens)
            dataset = (
                tf.data.TextLineDataset(dataset_files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
                .map(tokenizer.tokenize)
                .map(scatter)
            )
        else:
            tokenize = lambda inputs, outputs: ((tokenizer.tokenize(inputs), tokenizer.tokenize(outputs)))
            dataset = tf.data.experimental.CsvDataset(
                dataset_files, [tf.string, tf.string], header=args.header, field_delim="\t"
            ).map(tokenize)

        dataset = dataset.padded_batch(args.batch_size).prefetch(args.prefetch_buffer_size)

        # Model Initialize & Load pretrained model
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model = create_model(args.model_name, json.load(f))
        model((tf.keras.Input([None], dtype=tf.int32), tf.keras.Input([None], dtype=tf.int32)))
        model.load_weights(args.model_path)
        searcher = Searcher(model)
        logger.info("Loaded weights of model")

        # Evaluate
        bleu_sum = 0.0
        perplexity_sum = 0.0
        total = 0
        bos_id, eos_id = tokenizer.tokenize("").numpy().tolist()
        dataset_tqdm = tqdm(dataset)
        for batch_input, batch_true_answer in dataset_tqdm:
            num_batch = len(batch_true_answer)
            if args.beam_size > 0:
                batch_pred_answer, perplexity = searcher.beam_search(
                    batch_input, args.beam_size, bos_id, eos_id, args.max_sequence_length
                )
                batch_pred_answer = batch_pred_answer[:, 0, :]
            else:
                batch_pred_answer, perplexity = searcher.greedy_search(
                    batch_input, bos_id, eos_id, args.max_sequence_length
                )
            perplexity_sum += tf.math.reduce_sum(perplexity).numpy()

            for true_answer, pred_answer in zip(batch_true_answer, batch_pred_answer):
                bleu_sum += calculat_bleu_score(true_answer.numpy().tolist(), pred_answer.numpy().tolist())

            total += num_batch
            dataset_tqdm.set_description(f"Perplexity: {perplexity_sum / total}, BLEU: {bleu_sum / total}")

        logger.info("Finished evalaution!")
        logger.info(f"Perplexity: {perplexity_sum / total}, BLEU: {bleu_sum / total}")
