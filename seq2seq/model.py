from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Dropout, Embedding, Masking, SimpleRNN

from .layer import BahdanauAttention, PositionalEncoding, TransformerDecoderLayer, TransformerEncoderLayer

RNN_CELL_MAP: Dict[str, tf.keras.layers.Layer] = {
    "SimpleRNN": SimpleRNN,
    "LSTM": LSTM,
    "GRU": GRU,
}


class TransformerSeq2Seq(tf.keras.Model):
    """
    Seq2seq model using RNN cell.

    Arguments:
        cell_type: String, one of (SimpleRNN, LSTM, GRU).
        vocab_size: Integer, the size of vocabulary.
        dim_embedding: Integer, model internal dimension.
        num_heads: Integer, the number of heads.
        num_encoder_layers: Integer, the number of seq2seq encoder layers.
        num_decoder_layers: Integer, the number of seq2seq decoder layers.
        dim_feedfoward: Integer, feedforward dimension.
        activation: Integer, feedforward activation function.
        dropout: Float dropout rate.
        positional_max_sequence: Integer, max sequence to pos encode.
        pad_id: Integer, the id of padding token.

    Call arguments:
        inputs: A tuple (encoder_tokens, decoder_tokens)
            encoder_tokens: A 3D tensor, with shape of `[BatchSize, EncoderSequenceLength]`.
                                all values are in [0, VocabSize).
            decoder_tokens: A 3D tensor, with shape of `[BatchSize, DecoderSequenceLength]`.
                                all values are in [0, VocabSize).
            encoder_attention_mask: Optional, A 2D tensor, with shape of `[BatchSize, EncoderSequenceLength]` as float type tensor.
            decoder_attention_mask: Optional, A 2D tensor, with shape of `[BatchSize, DecoderSequenceLength]` as float type tensor.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, VocabSize]`
    """

    def __init__(
        self,
        vocab_size: int,
        dim_embedding: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedfoward: int,
        activation: str,
        dropout: float,
        positional_max_sequence: int = 1024,
        pad_id: int = 0,
    ):
        super(TransformerSeq2Seq, self).__init__()

        self.embedding = Embedding(vocab_size, dim_embedding, name="embedding")
        self.pos_encode = PositionalEncoding(dim_embedding, positional_max_sequence, name="pos_encode")
        self.dropout = Dropout(dropout, name="dropout")
        args = dim_embedding, num_heads, dim_feedfoward, dropout, activation
        self.encoder = [TransformerEncoderLayer(*args, name=f"encoder_layer{i}") for i in range(num_encoder_layers)]
        self.decoder = [TransformerDecoderLayer(*args, name=f"decoder_layer{i}") for i in range(num_decoder_layers)]
        self.dense = Dense(vocab_size, name="feedforward")
        self.pad_id = pad_id

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None):
        if len(inputs) == 2:
            encoder_tokens, decoder_tokens = inputs
            encoder_attention_mask = tf.expand_dims(tf.cast(encoder_tokens == self.pad_id, tf.float32), axis=1)
            decoder_attention_mask = tf.expand_dims(tf.cast(decoder_tokens == self.pad_id, tf.float32), axis=1)
        else:
            encoder_tokens, decoder_tokens, encoder_attention_mask, decoder_attention_mask = inputs

        # [BatchSize, SequenceLength, DimEmbedding]
        encoder_input = self.embedding(encoder_tokens)
        decoder_input = self.embedding(decoder_tokens)

        # [BatchSize, SequenceLength, DimEmbedding]
        encoder_input = self.dropout(self.pos_encode(encoder_input))
        decoder_input = self.dropout(self.pos_encode(decoder_input))

        # [BatchSize, SequenceLength, DimEmbedding]
        for encoder_layer in self.encoder:
            encoder_input = encoder_layer(encoder_input, encoder_attention_mask)
        for decoder_layer in self.decoder:
            decoder_input = decoder_layer(decoder_input, encoder_input, encoder_attention_mask, decoder_attention_mask)

        # [BatchSize, VocabSize]
        output = self.dense(decoder_input[:, -1, :])
        return output


class RNNSeq2Seq(tf.keras.Model):
    """
    Seq2seq model using RNN cell.

    Arguments:
        cell_type: String, one of (SimpleRNN, LSTM, GRU).
        vocab_size: Integer, the size of vocabulary.
        hidden_dim: Integer, the hidden dimension size of SampleModel.
        num_encoder_layers: Integer, the number of seq2seq encoder.
        num_decoder_layers: Integer, the number of seq2seq decoder.
        dropout: Float dropout rate
        pad_id: Integer, the id of padding token.

    Call arguments:
        inputs: A tuple (encoder_tokens, decoder_tokens)
            encoder_tokens: A 3D tensor, with shape of `[BatchSize, EncoderSequenceLength]`.
                                all values are in [0, VocabSize).
            decoder_tokens: A 3D tensor, with shape of `[BatchSize, DecoderSequenceLength]`.
                                all values are in [0, VocabSize).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, VocabSize]`
    """

    def __init__(
        self,
        cell_type,
        vocab_size,
        hidden_dim,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        pad_id=0,
    ):
        super(RNNSeq2Seq, self).__init__()

        assert cell_type in RNN_CELL_MAP, "RNN type is not valid!"

        self.embedding = Embedding(vocab_size, hidden_dim, name="embedding")
        self.pad_masking = Masking(pad_id, name="masking")
        self.dropout = Dropout(dropout, name="dropout")
        self.encoder = [
            # SimpleRNN, LSTM, GRU
            Bidirectional(
                RNN_CELL_MAP[cell_type](
                    hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                ),
                name=f"encoder_layer{i}",
            )
            for i in range(num_encoder_layers)
        ]
        self.decoder = [
            # SimpleRNN, LSTM, GRU
            RNN_CELL_MAP[cell_type](
                hidden_dim * 2,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f"decoder_layer{i}",
            )
            for i in range(num_decoder_layers)
        ]

        self.dense = Dense(vocab_size, name="dense")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None):
        encoder_tokens, decoder_tokens = inputs

        # [BatchSize, SequenceLength, HiddenDim]
        encoder_input = self.dropout(self.pad_masking(self.embedding(encoder_tokens)))
        decoder_input = self.dropout(self.pad_masking(self.embedding(decoder_tokens)))

        # [BatchSize, SequenceLength, HiddenDim]
        states = None
        for encoder_layer in self.encoder:
            encoder_input, *states = encoder_layer(encoder_input, initial_state=states)

        # Concat Forward-Backward states
        if len(states) == 2:
            states = (tf.concat(states, axis=-1),)
        elif len(states) == 4:
            states = (tf.concat(states[::2], axis=-1), tf.concat(states[1::2], axis=-1))

        for decoder_layer in self.decoder:
            decoder_input, *states = decoder_layer(decoder_input, initial_state=states)

        # [BatchSize, VocabSize]
        output = self.dense(decoder_input[:, -1, :])
        return output


class RNNSeq2SeqWithAttention(tf.keras.Model):
    """
    Seq2seq model using RNN cell with attention.

    Arguments:
        cell_type: String, one of (SimpleRNN, LSTM, GRU).
        vocab_size: Integer, the size of vocabulary.
        hidden_dim: Integer, the hidden dimension size of SampleModel.
        num_encoder_layers: Integer, the number of seq2seq encoder.
        num_decoder_layers: Integer, the number of seq2seq decoder.
        dropout: Float, dropout rate.
        pad_id: Integer, the id of padding token.

    Call arguments:
        inputs: A tuple (encoder_tokens, decoder_tokens)
            encoder_tokens: A 3D tensor, with shape of `[BatchSize, EncoderSequenceLength]`.
                                all values are in [0, VocabSize).
            decoder_tokens: A 3D tensor, with shape of `[BatchSize, DecoderSequenceLength]`.
                                all values are in [0, VocabSize).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Output Shape:
        2D tensor with shape:
            `[BatchSize, VocabSize]`
    """

    def __init__(
        self,
        cell_type,
        vocab_size,
        hidden_dim,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        pad_id=0,
    ):
        super(RNNSeq2SeqWithAttention, self).__init__()

        assert cell_type in RNN_CELL_MAP, "RNN type is not valid!"

        self.embedding = Embedding(vocab_size, hidden_dim, name="embedding")
        self.pad_masking = Masking(pad_id, name="masking")
        self.dropout = Dropout(dropout, name="dropout")
        self.encoder = [
            # SimpleRNN, LSTM, GRU
            Bidirectional(
                RNN_CELL_MAP[cell_type](
                    hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                ),
                name=f"encoder_layer{i}",
            )
            for i in range(num_encoder_layers)
        ]
        self.decoder = [
            # SimpleRNN, LSTM, GRU
            RNN_CELL_MAP[cell_type](
                hidden_dim * 2,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f"decoder_layer{i}",
            )
            for i in range(num_decoder_layers)
        ]

        self.attention = BahdanauAttention(hidden_dim, name="attention")
        self.dense = Dense(vocab_size, name="dense")

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None):
        encoder_tokens, decoder_tokens = inputs

        # [BatchSize, SequenceLength, HiddenDim]
        encoder_input = self.dropout(self.pad_masking(self.embedding(encoder_tokens)))
        decoder_input = self.dropout(self.pad_masking(self.embedding(decoder_tokens)))

        # [BatchSize, SequenceLength, HiddenDim]
        states = None
        for encoder_layer in self.encoder:
            encoder_input, *states = encoder_layer(encoder_input, initial_state=states)
        context_decoder_input = decoder_input

        # Concat Forward-Backward states
        if len(states) == 2:
            states = (tf.concat(states, axis=-1),)
        elif len(states) == 4:
            states = (tf.concat(states[::2], axis=-1), tf.concat(states[1::2], axis=-1))

        for decoder_layer in self.decoder:
            decoder_input, *states = decoder_layer(context_decoder_input, initial_state=states)
            context = self.attention(states[0], encoder_input)
            context_decoder_input = tf.concat([tf.expand_dims(context, axis=1), decoder_input], axis=1)

        # [BatchSize, VocabSize]
        output = self.dense(decoder_input[:, -1, :])
        return output


MODEL_MAP: Dict[str, tf.keras.Model] = {
    "TransformerSeq2Seq": TransformerSeq2Seq,
    "RNNSeq2Seq": RNNSeq2Seq,
    "RNNSeq2SeqWithAttention": RNNSeq2SeqWithAttention,
}
