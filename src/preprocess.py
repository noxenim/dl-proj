"""
Text Preprocessing Module for Sentiment Analysis.
Handles text cleaning, tokenization, and sequence padding.
"""

import re
import html
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text: str) -> str:
    """
    Cleans raw customer review text:
    - Decodes HTML entities (e.g. &amp; -> &)
    - Strips HTML tags (<br />, <p>, etc.)
    - Removes URLs
    - Normalizes punctuation and keeps apostrophes/contractions for sentiment context
    - Lowercases text
    - Removes superfluous whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove non-alphabetical characters except standard punctuation used in contractions
    text = re.sub(r"[^a-zA-Z0-9\s'!\?.,-]", ' ', text)
    
    # Expand or normalize common contraction patterns if needed, keep 'not', "n't"
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_tokenizer(vocab_size: int = 15000, oov_token: str = "<OOV>") -> Tokenizer:
    """Creates a Keras Tokenizer configured with vocab size and out-of-vocabulary token."""
    return Tokenizer(num_words=vocab_size, oov_token=oov_token)


def fit_and_save_tokenizer(texts, tokenizer_path: str, vocab_size: int = 15000) -> Tokenizer:
    """Fits tokenizer on review texts and saves it to disk via pickle."""
    tokenizer = create_tokenizer(vocab_size=vocab_size)
    tokenizer.fit_on_texts(texts)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Loads a pre-fitted Tokenizer from disk."""
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def transform_texts(texts, tokenizer: Tokenizer, max_len: int = 100) -> np.ndarray:
    """
    Converts list of raw texts to padded sequences:
    1. Clean text
    2. Convert to integer token sequences
    3. Pad / truncate to fixed length
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded
