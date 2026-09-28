"""
Inference Module for Customer Review Sentiment Analysis.
Provides the SentimentPredictor class for single and batch predictions.
"""

import os
import json
import numpy as np
import tensorflow as tf
from src.preprocess import clean_text, load_tokenizer, transform_texts


class SentimentPredictor:
    """Encapsulates the trained BiLSTM model and Tokenizer for seamless sentiment inference."""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "model.keras")
        self.tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pickle")
        self.config_path = os.path.join(artifacts_dir, "config.json")
        
        self.model = None
        self.tokenizer = None
        self.config = {
            "max_len": 100,
            "vocab_size": 15000,
            "threshold": 0.5
        }
        self._load()
    
    def _load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config.update(json.load(f))
        
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = load_tokenizer(self.tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}. Run train.py first.")
            
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Run train.py first.")

    def predict_one(self, review_text: str) -> dict:
        """
        Predicts sentiment for a single review string.
        Returns:
            dict containing:
            - sentiment: 'Positive' or 'Negative'
            - confidence: float between 0.50 and 1.00 (the model's confidence in its chosen class)
            - probability: raw sigmoid output in [0, 1] (probability of Positive)
            - cleaned_text: preprocessed text
            - word_count: token count
        """
        cleaned = clean_text(review_text)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "probability": 0.5,
                "cleaned_text": "",
                "word_count": 0
            }
        
        max_len = self.config.get("max_len", 100)
        padded = transform_texts([review_text], self.tokenizer, max_len=max_len)
        prob = float(self.model.predict(padded, verbose=0)[0][0])
        
        is_positive = prob >= self.config.get("threshold", 0.5)
        sentiment = "Positive" if is_positive else "Negative"
        confidence = prob if is_positive else (1.0 - prob)
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probability": round(prob, 4),
            "cleaned_text": cleaned,
            "word_count": len(cleaned.split())
        }

    def predict_batch(self, reviews_list: list, batch_size: int = 128) -> list:
        """
        Performs fast vectorized batch inference over a list of reviews.
        """
        max_len = self.config.get("max_len", 100)
        threshold = self.config.get("threshold", 0.5)
        
        padded = transform_texts(reviews_list, self.tokenizer, max_len=max_len)
        raw_probs = self.model.predict(padded, batch_size=batch_size, verbose=0).flatten()
        
        results = []
        for text, prob in zip(reviews_list, raw_probs):
            prob = float(prob)
            is_pos = prob >= threshold
            sentiment = "Positive" if is_pos else "Negative"
            conf = prob if is_pos else (1.0 - prob)
            results.append({
                "sentiment": sentiment,
                "confidence": round(conf, 4),
                "probability": round(prob, 4)
            })
        return results
