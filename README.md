# Khmer Text Transliteration

A Python-based system for transliterating English text to Khmer script using sequence-to-sequence neural networks.

## Overview

This project provides tools to convert English phonetic text into Khmer script. It uses a sequence-to-sequence model with LSTM layers for transliteration.

## Features

- English to Khmer text transliteration
- Multiple prediction variants  
- Fuzzy matching and similarity search
- Web interface using Gradio

## Project Structure

### Pre-trained Models

The project includes pre-trained models located in `src/models/pretrained/`:

- `khmer_transliterator.keras`: A pre-trained sequence-to-sequence model for English to Khmer transliteration

### Training Assets

Tokenizer and model assets are stored in `data/processed/`:

- `khmer_transliteration_assets.pkl`: Contains the English and Khmer tokenizers, along with sequence length information

### Training Data

Raw data for training and reference is available in `data/raw/`:

- `eng_khm_data.txt`: Training data with English-Khmer word pairs
- `khmer_words.txt`: Dictionary of Khmer words
- `1000-most-common-khmer-words/`: Collection of common Khmer words for reference

### Training Process

The model training process is documented in the notebooks:

- `notebooks/khmer_seq2seq.ipynb`: Jupyter notebook containing the complete training pipeline, including:
  - Data preprocessing
  - Model architecture
  - Training configuration
  - Evaluation metrics
  - Example predictions

To train a new model or experiment with the existing one, refer to the training notebook for detailed instructions and parameters.

## Core Functions

### 1. Basic Transliteration

```python
from src.predict import transliterate

# Convert English text to Khmer
result = transliterate("somlor")  # Returns: សម្ល
```

### 2. Generate Multiple Variants

```python
from src.predict import transliterate_variants

# Get multiple possible transliterations
variants = transliterate_variants("srolanh", num_variants=3, temperature=0.7)
# Returns: ['ស្រឡាញ់', 'ស្រលាញ', 'ស្រលាញ់']
```

### 3. Find Similar Words

```python
from src.predict_with_clean import find_similar

# Find similar Khmer words
similar_words = find_similar("min", max_results=2)
# Returns: ['មិន', 'មីន']
```

### 4. TF-IDF Based Similarity Search

```python
from src.predict_with_clean import find_similar_tfidf

# Find similar words using TF-IDF
similar = find_similar_tfidf("min", max_results=2)
# Returns: ['មិន', 'មីន']
```

### 5. Last Result Prediction

```python
from src.predict_with_clean import predict_last_result

# Get final predictions with scoring
results = predict_last_result("snam", num_results=3)
# Returns: ['ស្នាម', 'ស្នំ', 'សម្នាម']
```

## Requirements

- TensorFlow 2.x
- NumPy
- scikit-learn
- python-Levenshtein
- rapidfuzz
- gradio (for web interface)

## Installation

```bash
pip install -r requirements.txt
```

## License

MIT License
