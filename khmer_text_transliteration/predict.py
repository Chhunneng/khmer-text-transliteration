import numpy as np
import pickle
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from khmer_text_transliteration.data_processing import clean_text, normalize_khmer
from khmer_text_transliteration.config import *

# Load assets
with open(ASSETS_PATH, "rb") as f:
    assets = pickle.load(f)
eng_tokenizer = assets["eng_tokenizer"]
khm_tokenizer = assets["khm_tokenizer"]
max_eng_len = assets["max_eng_len"]
max_khm_len = assets["max_khm_len"]

# Load model
model = load_model(MODEL_PATH)

encoder_inputs = model.input[0]
encoder_outputs, state_h, state_c = model.get_layer("encoder_lstm").output
encoder_model = Model(encoder_inputs, [state_h, state_c])

# Decoder
decoder_inputs = model.input[1]
decoder_state_input_h = Input(shape=(LSTM_UNITS,))
decoder_state_input_c = Input(shape=(LSTM_UNITS,))
decoder_states = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = model.get_layer("decoder_lstm")(
    model.get_layer("decoder_embedding")(decoder_inputs), initial_state=decoder_states
)
decoder_outputs = model.get_layer("decoder_dense")(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states, [decoder_outputs, state_h, state_c]
)


def transliterate(text):
    """
    Transliterates the given text from English to Khmer.
    This function takes an input text in English, cleans it, encodes it, and then uses a trained encoder-decoder model to transliterate it into Khmer script.
    Args:
        text (str): The input text in English to be transliterated.
    Returns:
        str: The transliterated text in Khmer script. If the input text is empty or cannot be cleaned, an empty string is returned.
    Steps:
        1. Clean the input text using the `clean_text` function.
        2. Encode the cleaned text into sequences using the English tokenizer.
        3. Pad the encoded sequences to a fixed length.
        4. Use the encoder model to predict the initial states for the decoder.
        5. Initialize the target sequence with the start token for Khmer.
        6. Iteratively predict the next character using the decoder model until the stop condition is met (either the end token is generated or the maximum length is reached).
        7. Append each predicted character to the decoded output.
        8. Normalize the decoded output using the `normalize_khmer` function and return the result.
    """
    # Clean input
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Encode
    seq = eng_tokenizer.texts_to_sequences([cleaned])
    encoder_input = pad_sequences(seq, maxlen=max_eng_len + 1, padding="post")
    states = encoder_model.predict(encoder_input)

    # Start decoding
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = khm_tokenizer.word_index["\t"]
    stop_condition = False
    decoded = []

    while not stop_condition:
        output, h, c = decoder_model.predict([target_seq] + states)
        char_index = np.argmax(output[0, -1, :])
        char = khm_tokenizer.index_word.get(char_index, "")
        if char == "\n" or len(decoded) >= (max_khm_len + 1):
            stop_condition = True
        else:
            decoded.append(char)
            target_seq[0, 0] = char_index
            states = [h, c]

    return normalize_khmer("".join(decoded))


# Modified prediction function with temperature
def transliterate_variants(text, num_variants=3, temperature=0.7, max_attempts=20):
    """
    Generate multiple transliteration variants for a given text.
    This function attempts to generate a specified number of unique transliteration variants
    for the input text using a sequence-to-sequence model with dynamic temperature adjustment
    and top-k sampling.
    Args:
        text (str): The input text to be transliterated.
        num_variants (int, optional): The number of unique transliteration variants to generate. Default is 3.
        temperature (float, optional): The temperature parameter for sampling. Higher values result in more random samples. Default is 0.7.
        max_attempts (int, optional): The maximum number of attempts to generate unique variants. Default is 20.
    Returns:
        list: A list of unique transliteration variants. If the desired number of unique variants
              cannot be generated within the maximum attempts, a warning is printed and the available
              unique variants are returned.
    Raises:
        ValueError: If the input text is empty or cannot be cleaned.
    Example:
        >>> variants = transliterate_variants("example text", num_variants=5, temperature=0.8, max_attempts=30)
        >>> print(variants)
        ['variant1', 'variant2', 'variant3', 'variant4', 'variant5']
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Encode input
    seq = eng_tokenizer.texts_to_sequences([cleaned])
    encoder_input = pad_sequences(seq, maxlen=max_eng_len, padding="post")
    states = encoder_model.predict(encoder_input)

    unique_variants = set()
    attempt_count = 0

    while len(unique_variants) < num_variants and attempt_count < max_attempts:
        # Start decoding with fresh states each attempt
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = khm_tokenizer.word_index["\t"]
        current_states = [s.copy() for s in states]
        decoded = []
        attempt_count += 1

        # Dynamic temperature adjustment
        current_temp = temperature * (
            1 + attempt_count / 10
        )  # Increase temp slightly each attempt

        for _ in range(max_khm_len + 1):
            output, h, c = decoder_model.predict([target_seq] + current_states)

            # Apply temperature with top-k filtering
            probs = output[0, -1, :]
            probs = np.exp(np.log(probs) / current_temp)
            probs /= probs.sum()

            # Get top 3 candidates
            top_k = 3
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs /= top_probs.sum()  # Renormalize

            char_index = np.random.choice(top_indices, p=top_probs)
            char = khm_tokenizer.index_word.get(char_index, "")

            if char == "\n":
                break

            decoded.append(char)
            target_seq[0, 0] = char_index
            current_states = [h, c]

        variant = normalize_khmer("".join(decoded))
        if variant and variant not in unique_variants:
            unique_variants.add(variant)

    # Fallback if not enough unique variants
    if len(unique_variants) < num_variants:
        print(
            f"Warning: Only found {len(unique_variants)} unique variants after {max_attempts} attempts"
        )

    return list(unique_variants)[:num_variants]
