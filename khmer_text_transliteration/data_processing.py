import re
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def normalize_khmer(text):
    """
    Normalize Khmer text using Unicode Normalization Form C (NFC).

    This function takes a string of Khmer text and normalizes it using
    the NFC (Normalization Form C) standard. NFC is a standard for
    Unicode normalization that composes characters into their canonical
    composed form.

    Args:
        text (str): The Khmer text to be normalized.

    Returns:
        str: The normalized Khmer text.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text, is_khmer=False):
    """
    Cleans the input text by removing unwanted characters and normalizing it.

    Args:
        text (str): The input text to be cleaned.
        is_khmer (bool): A flag indicating whether the text is in Khmer script.
                         If True, the text will be cleaned and normalized for Khmer script.
                         If False, the text will be cleaned and converted to lowercase English letters.

    Returns:
        str: The cleaned and normalized text.

    Examples:
        >>> clean_text("Hello, World!")
        'helloworld'
        >>> clean_text("សួស្តី", is_khmer=True)
        'សួស្តី'
    """
    text = str(text).strip()
    if is_khmer:
        text = re.sub(r"[^\u1780-\u17FF]", "", text)
        text = normalize_khmer(text)
    else:
        text = re.sub(r"[^a-z]", "", text.lower())
    return text


def load_data(filename):
    """
    Load and process data from a file containing Khmer and English text pairs.

    The file should have lines in the format "khmer_text:english_text1,english_text2,...".
    Each line is split into Khmer and English parts, cleaned, and stored as pairs.

    Args:
        filename (str): The path to the file containing the text pairs.

    Returns:
        list of tuple: A list of tuples where each tuple contains a cleaned English text and a cleaned Khmer text.
    """
    pairs = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                khmer_part, eng_part = line.strip().split(":", 1)
                khmer_clean = clean_text(khmer_part, True)
                for eng in eng_part.split(","):
                    eng_clean = clean_text(eng.strip())
                    if eng_clean and khmer_clean:
                        pairs.append((eng_clean, khmer_clean))
    return pairs


def khmer_word(filename):
    """
    Extracts and processes Khmer words from a file.

    This function reads a file line by line, extracts the Khmer part of each line
    (assuming the line contains a ':' character), and processes the extracted text
    using the `clean_text` function. The processed Khmer words are then collected
    into a list and returned.

    Args:
        filename (str): The path to the file containing Khmer words.

    Returns:
        list: A list of processed Khmer words.

    Example:
        >>> khmer_words = khmer_word('khmer_words.txt')
        >>> print(khmer_words)
        ['ខ្មែរ', 'សួស្តី', 'អរុណសួស្តី']
    """
    khmer_words_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                khmer_part, _ = line.strip().split(":", 1)
                khmer_words_list.append(clean_text(khmer_part, True))
    return khmer_words_list


def flat_khmer_word(filename):
    """
    Reads a file containing Khmer words, processes each line, and returns a list of cleaned Khmer words.

    Args:
        filename (str): The path to the file containing Khmer words, with each word on a new line.

    Returns:
        list: A list of cleaned Khmer words.

    Example:
        >>> khmer_words = flat_khmer_word('khmer_words.txt')
        >>> print(khmer_words)
        ['សួស្តី', 'ជំរាបសួរ', 'អរុណសួស្តី']

    Note:
        The function assumes that the `clean_text` function is defined elsewhere in the codebase and is used to clean each line of text.
    """
    # read from file that each line is khmer word
    khmer_words_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            khmer_words_list.append(clean_text(line, True))
    return khmer_words_list


def prepare_tokenizers(dataset):
    """
    Prepares and fits tokenizers for English and Khmer text data.
    Args:
        dataset (list of tuples): A list of tuples where each tuple contains an English string and a Khmer string.
    Returns:
        tuple: A tuple containing two tokenizers:
            - eng_tokenizer: Tokenizer for English text, character-level, with no filters and an out-of-vocabulary token '<unk>'.
            - khm_tokenizer: Tokenizer for Khmer text, character-level, with no filters and an out-of-vocabulary token '<unk>'.
    """
    eng_tokenizer = Tokenizer(char_level=True, filters="", oov_token="<unk>")
    eng_tokenizer.fit_on_texts([eng for eng, _ in dataset])

    khm_tokenizer = Tokenizer(char_level=True, filters="", oov_token="<unk>")
    khm_tokenizer.fit_on_texts(["\t", "\n"] + [khm for _, khm in dataset])
    return eng_tokenizer, khm_tokenizer


def prepare_sequences(dataset, eng_tokenizer, khm_tokenizer):
    """
    Prepares sequences for encoder and decoder inputs and targets from the given dataset.
    Args:
        dataset (list of tuples): A list of tuples where each tuple contains an English sentence and its corresponding Khmer sentence.
        eng_tokenizer (Tokenizer): A Keras Tokenizer fitted on the English sentences.
        khm_tokenizer (Tokenizer): A Keras Tokenizer fitted on the Khmer sentences.
    Returns:
        tuple: A tuple containing:
            - encoder_data (numpy.ndarray): Padded sequences for encoder inputs.
            - decoder_input_data (numpy.ndarray): Padded sequences for decoder inputs.
            - decoder_target_data (numpy.ndarray): Padded sequences for decoder targets.
            - max_eng_len (int): Maximum length of the English sequences.
            - max_khm_len (int): Maximum length of the Khmer sequences.
    """
    encoder_inputs, decoder_inputs, decoder_targets = [], [], []
    max_eng_len = max(len(eng) for eng, _ in dataset) or 1
    max_khm_len = max(len(khm) for _, khm in dataset) or 1

    for eng, khm in dataset:
        enc_seq = eng_tokenizer.texts_to_sequences([eng])[0]
        encoder_inputs.append(enc_seq)

        khm_seq = khm_tokenizer.texts_to_sequences([khm])[0]
        decoder_input = [khm_tokenizer.word_index["\t"]] + khm_seq
        decoder_target = khm_seq + [khm_tokenizer.word_index["\n"]]

        decoder_inputs.append(decoder_input)
        decoder_targets.append(decoder_target)

    encoder_data = pad_sequences(encoder_inputs, maxlen=max_eng_len, padding="post")
    decoder_input_data = pad_sequences(
        decoder_inputs, maxlen=max_khm_len + 1, padding="post"
    )
    decoder_target_data = pad_sequences(
        decoder_targets, maxlen=max_khm_len + 1, padding="post"
    )

    return (
        encoder_data,
        decoder_input_data,
        decoder_target_data,
        max_eng_len,
        max_khm_len,
    )
