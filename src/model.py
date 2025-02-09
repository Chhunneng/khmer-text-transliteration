from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from config import EMBED_DIM, LSTM_UNITS


def build_model(eng_vocab_size, khm_vocab_size):
    """
    Builds and returns a sequence-to-sequence model for English to Khmer text transliteration.

    Args:
        eng_vocab_size (int): Size of the English vocabulary.
        khm_vocab_size (int): Size of the Khmer vocabulary.

    Returns:
        keras.Model: A Keras Model instance representing the sequence-to-sequence model.

    The model consists of:
    - An encoder with an embedding layer and an LSTM layer.
    - A decoder with an embedding layer, an LSTM layer, and a dense layer with softmax activation.

    The encoder processes the input English text and generates internal state vectors.
    The decoder uses these state vectors to generate the corresponding Khmer text.
    """
    # Encoder
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    enc_emb = Embedding(eng_vocab_size, EMBED_DIM, name="encoder_embedding")(
        encoder_inputs
    )
    encoder_lstm = LSTM(LSTM_UNITS, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(enc_emb)

    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    dec_emb = Embedding(khm_vocab_size, EMBED_DIM, name="decoder_embedding")(
        decoder_inputs
    )
    decoder_lstm = LSTM(
        LSTM_UNITS, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    decoder_dense = Dense(khm_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model"
    )
    return model
