import argparse
import csv
import glob
import os

import tensorflow as tf
import tensorflow_text as text
from tqdm import tqdm

from seq2seq.data import serialize_example

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str, required=True, help="Input File glob pattern")
parser.add_argument("--output-dir", type=str, required=True, help="Output path file or directory")
parser.add_argument("--sp-model-path", type=str, required=True, help="Sentencepiece model path")
parser.add_argument("--auto-encoding", action="store_true", help="If use autoencoding, dataset is .txt format else .tsv format")
# fmt: on


def read_data(file_path: str, tokenizer: text.SentencepieceTokenizer, auto_encoding: bool):
    with open(file_path) as f:
        if auto_encoding:
            for line in f:
                tokens = tokenizer.tokenize(line.strip())
                yield tokens, tokens
        else:
            reader = csv.reader(f, delimiter="\t")
            for source_sentence, target_sentence in reader:
                source_tokens = tokenizer.tokenize(source_sentence)
                target_tokens = tokenizer.tokenize(target_sentence)
                yield source_tokens, target_tokens


if __name__ == "__main__":
    args = parser.parse_args()
    input_files = glob.glob(args.input_path)

    # Load Sentencepiece model
    with open(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    for file_path in tqdm(input_files):
        output_path = os.path.join(args.output_dir, os.path.splitext(file_path)[0] + ".tfrecord")

        # Write TFRecordFile
        with tf.io.TFRecordWriter(output_path) as writer:
            for source_tokens, target_tokens in read_data(file_path, tokenizer, args.auto_encoding):
                serialized_example = serialize_example(source_tokens, target_tokens)
                writer.write(serialized_example)
