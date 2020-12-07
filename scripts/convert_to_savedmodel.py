import argparse
import json
from functools import partial

import tensorflow as tf
import tensorflow_text as text

from seq2seq.model import MODEL_MAP
from seq2seq.search import beam_search, greedy_search
from seq2seq.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="RNNSeq2Seq", help="Seq2seq model name")
parser.add_argument("--model-config-path", type=str, default="resources/configs/rnn.json", help="model config file")
parser.add_argument("--model-weight-path", type=str, required=True, help="Model weight file path saved in training")
parser.add_argument("--spm-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model", help="spm tokenizer model path")
parser.add_argument("--output-path", type=str, default="seq2seq-model/1", help="Savedmodel path")

search = parser.add_argument_group("Search Method Configs")
search.add_argument("--pad-id", type=int, default=0, help="Pad token id when tokenize with sentencepiece")
search.add_argument("--max-sequence-length", type=int, default=128, help="Max number of tokens including bos, eos")
search.add_argument("--alpha", type=int, default=1, help="length penalty control variable when beam searching")
search.add_argument("--beta", type=int, default=32, help="length penalty control variable when beam searching")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()

    logger = get_logger()

    with open(args.model_config_path) as f:
        model = MODEL_MAP[args.model_name](**json.load(f))
    model.load_weights(args.model_weight_path)
    logger.info("Loaded weights of model")

    with open(args.spm_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("")
    logger.info("Loaded sentencepiece tokenizer")

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string), tf.TensorSpec([], tf.int32)])
    def generate_with_beam_search(texts, beam_size):
        batch_size = tf.shape(texts)[0]
        tokens = tokenizer.tokenize(texts).to_tensor(default_value=args.pad_id)
        decoded_tokens, perplexity = beam_search(
            model, tokens, beam_size, bos_id, eos_id, args.max_sequence_length, args.pad_id, args.alpha, args.beta
        )
        sentences = tokenizer.detokenize(tf.reshape((decoded_tokens), [batch_size * beam_size, -1]))
        sentences = tf.reshape(sentences, [batch_size, beam_size, -1])
        return {"sentences": sentences, "perplexity": perplexity}

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def generate_with_greedy_search(texts):
        tokens = tokenizer.tokenize(texts).to_tensor(default_value=args.pad_id)
        decoded_tokens, perplexity = greedy_search(model, tokens, bos_id, eos_id, args.max_sequence_length, args.pad_id)
        sentences = tokenizer.detokenize(decoded_tokens)
        return {"sentences": sentences, "perplexity": perplexity}

    model.tokenizer = tokenizer
    tf.saved_model.save(
        model,
        args.output_path,
        signatures={"serving_default": generate_with_greedy_search, "beam_search": generate_with_beam_search},
    )
    logger.info(f"Saved model to {args.output_path}")
