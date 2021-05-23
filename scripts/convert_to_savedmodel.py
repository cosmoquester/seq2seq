import argparse
import os
import random
import sys

import tensorflow as tf
import tensorflow_text as text
import yaml
from tensorflow_serving.apis import predict_pb2, prediction_log_pb2

from seq2seq.model import create_model
from seq2seq.search import Searcher
from seq2seq.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="RNNSeq2Seq", help="Seq2seq model name")
parser.add_argument("--model-config-path", type=str, default="resources/configs/rnn.yml", help="model config file")
parser.add_argument("--model-weight-path", type=str, required=True, help="Model weight file path saved in training")
parser.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model", help="sp tokenizer model path")
parser.add_argument("--output-path", type=str, default="seq2seq-model/1", help="Savedmodel path")

search = parser.add_argument_group("Search Method Configs")
search.add_argument("--pad-id", type=int, default=0, help="Pad token id when tokenize with sentencepiece")
search.add_argument("--max-sequence-length", type=int, default=128, help="Max number of tokens including bos, eos")
search.add_argument("--alpha", type=float, default=1, help="length penalty control variable when beam searching")
search.add_argument("--beta", type=int, default=32, help="length penalty control variable when beam searching")
# fmt: on


def make_warmup_record(inputs, model_name, signature_name="serving_default"):
    predict_request = predict_pb2.PredictRequest()
    predict_request.model_spec.name = model_name
    predict_request.model_spec.signature_name = signature_name

    for key, value in inputs.items():
        predict_request.inputs[key].CopyFrom(tf.make_tensor_proto(value, get_tf_datatype(value)))

    log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=predict_request))
    return log.SerializeToString()


def get_tf_datatype(data):
    type_map = {int: tf.int32, float: tf.float32, str: tf.string}
    data_type = type(data)

    if data_type == list:
        return get_tf_datatype(data[0])
    if data_type in type_map:
        return type_map[data_type]
    raise RuntimeError(f"Data type:{type(data)} not supported!")


def random_string(length=16):
    string_ords = [random.randint(ord("가"), ord("힣")) for _ in range(length)]
    return "".join([chr(token_ord) for token_ord in string_ords])


def main(args: argparse.Namespace):
    logger = get_logger(__name__)

    with open(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("")
    logger.info("Loaded sentencepiece tokenizer")

    with open(args.model_config_path) as f:
        model = create_model(args.model_name, yaml.load(f, yaml.SafeLoader))
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(args.model_weight_path).expect_partial()
    searcher = Searcher(model, args.max_sequence_length, bos_id, eos_id, args.pad_id)
    logger.info("Loaded weights of model")

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string), tf.TensorSpec([], tf.int32)])
    def generate_with_beam_search(texts, beam_size):
        tokens = tokenizer.tokenize(texts).to_tensor(default_value=args.pad_id)
        decoded_tokens, perplexity = searcher.beam_search(tokens, beam_size, args.alpha, args.beta)
        sentences = tokenizer.detokenize(decoded_tokens).to_tensor()
        return {"sentences": sentences, "perplexity": perplexity}

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def generate_with_greedy_search(texts):
        tokens = tokenizer.tokenize(texts).to_tensor(default_value=args.pad_id)
        decoded_tokens, perplexity = searcher.greedy_search(tokens)
        sentences = tokenizer.detokenize(decoded_tokens)
        return {"sentences": sentences, "perplexity": perplexity}

    model.tokenizer = tokenizer
    tf.saved_model.save(
        model,
        args.output_path,
        signatures={"serving_default": generate_with_greedy_search, "beam_search": generate_with_beam_search},
    )
    logger.info(f"Saved model to {args.output_path}")

    warmup_record_path = os.path.join(args.output_path, "assets.extra", "tf_serving_warmup_requests")
    os.makedirs(os.path.dirname(warmup_record_path), exist_ok=True)
    with tf.io.TFRecordWriter(warmup_record_path) as writer:
        model_name = os.path.dirname(args.output_path)
        logger.info(f"Make warmup record with model name is {model_name}")
        writer.write(make_warmup_record({"texts": [random_string() for _ in range(5)]}, model_name))
        writer.write(
            make_warmup_record(
                {"texts": [random_string() for _ in range(5)], "beam_size": 3}, model_name, "beam_search"
            )
        )
    logger.info(f"Made warmup record and saved to {warmup_record_path}")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
