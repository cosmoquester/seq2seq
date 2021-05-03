import argparse

import tensorflow as tf
import tensorflow_text as text
import yaml

from seq2seq.model import create_model
from seq2seq.search import Searcher
from seq2seq.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser("This is script to inferece (generate sentence) with seq2seq model")
file_paths = parser.add_argument_group("File Paths")
file_paths.add_argument("--model-name", type=str, default="RNNSeq2SeqWithAttention", help="Seq2seq model name")
file_paths.add_argument("--model-config-path", type=str, default="resources/configs/rnn.yml", help="model config file")
file_paths.add_argument("--model-path", type=str, required=True, help="pretrained model checkpoint")
file_paths.add_argument("--sp-model-path", type=str, default="resources/sp-model/sp_model_unigram_16K.model")

inference_parameters = parser.add_argument_group("Inference Parameters")
inference_parameters.add_argument("--batch-size", type=int, default=512)
inference_parameters.add_argument("--prefetch-buffer-size", type=int, default=100)
inference_parameters.add_argument("--max-sequence-length", type=int, default=256)
inference_parameters.add_argument("--pad-id", type=int, default=0, help="Pad token id when tokenize with sentencepiece")
inference_parameters.add_argument("--beam-size", type=int, default=0, help="not given, use greedy search else beam search with this value as beam size")

other_settings = parser.add_argument_group("Other settings")
other_settings.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
other_settings.add_argument("--device", type=str, default="CPU", help="device to train model")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()
    logger = get_logger(__name__)

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("Use Mixed Precision FP16")

    # Load Tokenizer
    with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    bos_id, eos_id = tokenizer.tokenize("").numpy().tolist()

    # Model Initialize & Load pretrained model
    with tf.io.gfile.GFile(args.model_config_path) as f:
        model = create_model(args.model_name, yaml.load(f, yaml.SafeLoader))
    model.load_weights(args.model_path)
    searcher = Searcher(model, args.max_sequence_length, bos_id, eos_id, args.pad_id)
    logger.info("Loaded weights of model")

    while True:
        text = input("Please Input Text: ")
        if not text:
            break

        encoder_input = tokenizer.tokenize(text)[tf.newaxis, :]
        encoder_input = tf.concat([encoder_input, encoder_input], axis=0)
        if args.beam_size > 0:
            outputs, ppls = searcher.beam_search(encoder_input, args.beam_size)
            outputs = [output.decode("UTF8") for output in tokenizer.detokenize(outputs[0, :, :]).numpy()]
            ppls = ppls[0, :].numpy()

            for i, (output, ppl) in enumerate(zip(outputs, ppls)):
                print(f"Rank {i+1}, Output: {output}, Perplexity: {ppl:.4f}")
        else:
            output, ppl = searcher.greedy_search(encoder_input)
            output = tokenizer.detokenize(output[0]).numpy().decode("UTF8")
            print(f"Output: {output}, Perplexity: {ppl.numpy()[0]:.4f}")
