"""
Training Pipeline for Customer Review Sentiment Analysis using BiLSTM.
Loads Amazon Reviews Polarity, cleans & tokenizes text, trains BiLSTM model,
evaluates on held-out test data, and exports all artifacts & evaluation plots.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import tensorflow as tf

from src.preprocess import (
    clean_text,
    fit_and_save_tokenizer,
    transform_texts
)
from src.model import build_bilstm_model


def run_training_pipeline(
    data_path: str = "data/amazon_polarity_sample.parquet",
    artifacts_dir: str = "artifacts",
    sample_size: int = 24000,
    vocab_size: int = 15000,
    max_len: int = 100,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    epochs: int = 4,
    batch_size: int = 64,
    random_state: int = 42
):
    print("=" * 70)
    print(" CUSTOMER REVIEW SENTIMENT ANALYSIS - BiLSTM TRAINING PIPELINE")
    print("=" * 70)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Load Dataset
    print(f"\n[1/6] Loading Amazon Reviews dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Total reviews available in parquet: {len(df):,}")
    
    # Sample balanced subset for fast, robust training
    half_sample = sample_size // 2
    pos_df = df[df["label"] == 1].sample(n=half_sample, random_state=random_state)
    neg_df = df[df["label"] == 0].sample(n=half_sample, random_state=random_state)
    subset_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Combine title and review content
    subset_df["full_text"] = subset_df["title"].fillna("") + " " + subset_df["content"].fillna("")
    print(f"Prepared balanced subset: {len(subset_df):,} reviews (50% positive, 50% negative)")
    
    # 2. Text Preprocessing & Cleaning
    print("\n[2/6] Cleaning and preprocessing review texts...")
    t0 = time.time()
    cleaned_texts = [clean_text(t) for t in subset_df["full_text"].values]
    labels = subset_df["label"].values.astype(np.float32)
    print(f"Text cleaning completed in {time.time() - t0:.2f}s")
    
    # Train / Test Split (80% Train, 20% Test)
    train_texts, test_texts, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.20, random_state=random_state, stratify=labels
    )
    print(f"Training set: {len(train_texts):,} reviews | Testing set: {len(test_texts):,} reviews")
    
    # 3. Tokenization & Padding
    print(f"\n[3/6] Fitting Keras Tokenizer (vocab_size={vocab_size:,})...")
    tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pickle")
    tokenizer = fit_and_save_tokenizer(train_texts, tokenizer_path, vocab_size=vocab_size)
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Total unique words indexed: {len(tokenizer.word_index):,}")
    
    print(f"Transforming text sequences to padded arrays (max_len={max_len})...")
    x_train = transform_texts(train_texts, tokenizer, max_len=max_len)
    x_test = transform_texts(test_texts, tokenizer, max_len=max_len)
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    
    # 4. Build Model
    print("\n[4/6] Constructing Bidirectional LSTM Architecture...")
    model = build_bilstm_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_len=max_len,
        lstm_units=lstm_units,
        dropout_rate=0.3,
        learning_rate=0.001
    )
    model.summary()
    
    # Callbacks
    model_save_path = os.path.join(artifacts_dir, "model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 5. Train Model
    print(f"\n[5/6] Training BiLSTM on {len(x_train):,} samples for {epochs} epochs...")
    train_start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - train_start
    print(f"Model training finished in {train_duration:.2f}s ({train_duration/60:.2f} minutes)")
    
    # Save model explicitly to ensure latest weights
    model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")
    
    # 6. Evaluation on Test Data
    print("\n[6/6] Evaluating model on unseen test set...")
    y_pred_probs = model.predict(x_test, batch_size=batch_size, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    y_true = y_test.astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 50)
    print(" TEST SET EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    print("Confusion Matrix:\n", cm)
    
    # Save Metrics JSON
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_true),
        "training_time_seconds": round(train_duration, 2)
    }
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Config JSON
    config = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "batch_size": batch_size,
        "epochs": epochs,
        "threshold": 0.5,
        "training_samples": len(train_texts),
        "test_samples": len(test_texts)
    }
    with open(os.path.join(artifacts_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    # Save History JSON
    hist_dict = {
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])]
    }
    with open(os.path.join(artifacts_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=4)
        
    # Generate & Save Plots
    print("\nGenerating training curves and confusion matrix plots...")
    _save_plots(history, cm, artifacts_dir)
    print(f"All artifacts successfully saved to {os.path.abspath(artifacts_dir)}")
    return metrics


def _save_plots(history, cm, artifacts_dir):
    """Generates publication-quality evaluation plots."""
    # Plot 1: Training & Validation Loss/Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    
    # Accuracy Plot
    axes[0].plot(epochs_range, history.history["accuracy"], 'o-', color='#2563EB', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs_range, history.history["val_accuracy"], 's--', color='#10B981', label='Validation Accuracy', linewidth=2)
    axes[0].set_title("Model Accuracy Across Epochs", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim([0.7, 1.0])
    
    # Loss Plot
    axes[1].plot(epochs_range, history.history["loss"], 'o-', color='#DC2626', label='Training Loss', linewidth=2)
    axes[1].plot(epochs_range, history.history["val_loss"], 's--', color='#F59E0B', label='Validation Loss', linewidth=2)
    axes[1].set_title("Model Loss Across Epochs", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Binary Crossentropy Loss", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    hist_plot_path = os.path.join(artifacts_dir, "training_history.png")
    plt.savefig(hist_plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {hist_plot_path}")
    
    # Plot 2: Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    labels = ["Negative", "Positive"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold', pad=14)
    plt.xlabel("Predicted Sentiment Label", fontsize=12)
    plt.ylabel("Actual True Label", fontsize=12)
    plt.tight_layout()
    cm_plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_plot_path}")


if __name__ == "__main__":
    run_training_pipeline()
