import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBED_DIM = 32
LSTM_UNITS = 64
BATCH_SIZE = 16
EPOCHS = 50
BEAM_WIDTH = 3
MODEL_PATH = os.path.join(BASE_DIR, "models", "pretrained", "khmer_transliterator.keras")
ASSETS_PATH = os.path.join(BASE_DIR, "data", "processed", "khmer_transliteration_assets.pkl")
