"""
This script trains a Khmer text transliteration model using a sequence-to-sequence architecture.

Modules:
    numpy: A package for scientific computing with Python.
    pickle: A module for serializing and de-serializing Python object structures.
    data_processing: A custom module for loading data and preparing tokenizers and sequences.
    model: A custom module for building the sequence-to-sequence model.
    config: A custom module containing configuration parameters.

Functions:
    load_data(filepath): Loads the dataset from the specified file.
    prepare_tokenizers(dataset): Prepares tokenizers for the English and Khmer texts.
    prepare_sequences(dataset, eng_tokenizer, khm_tokenizer): Prepares the sequences for the encoder and decoder.
    build_model(eng_vocab_size, khm_vocab_size): Builds the sequence-to-sequence model.

Workflow:
    1. Load and prepare the dataset.
    2. Prepare tokenizers for the English and Khmer texts.
    3. Prepare the sequences for the encoder and decoder.
    4. Build and compile the sequence-to-sequence model.
    5. Train the model using the prepared data.
    6. Save the trained model and associated assets (tokenizers and sequence lengths).

Configuration:
    The script uses configuration parameters defined in the `config` module, including:
        - BATCH_SIZE: The batch size for training.
        - EPOCHS: The number of epochs for training.
        - MODEL_PATH: The file path to save the trained model.
        - ASSETS_PATH: The file path to save the tokenizers and sequence lengths.
"""

import numpy as np
import pickle
from data_processing import load_data, prepare_tokenizers, prepare_sequences
from model import build_model
import config

# Load and prepare data
dataset = load_data("eng_khm_data.txt")
eng_tokenizer, khm_tokenizer = prepare_tokenizers(dataset)
encoder_data, decoder_input_data, decoder_target_data, max_eng_len, max_khm_len = (
    prepare_sequences(dataset, eng_tokenizer, khm_tokenizer)
)

# Build and compile model
model = build_model(
    eng_vocab_size=len(eng_tokenizer.word_index) + 1,
    khm_vocab_size=len(khm_tokenizer.word_index) + 1,
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train model
history = model.fit(
    [encoder_data, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    validation_split=0.2,
)

# Save model and assets
model.save(config.MODEL_PATH)
assets = {
    "eng_tokenizer": eng_tokenizer,
    "khm_tokenizer": khm_tokenizer,
    "max_eng_len": max_eng_len,
    "max_khm_len": max_khm_len,
}
with open(config.ASSETS_PATH, "wb") as f:
    pickle.dump(assets, f)
