"""
BiLSTM Model Architecture for Sentiment Analysis.
Implements Embedding -> SpatialDropout1D -> Bidirectional LSTM -> Dense -> Sigmoid.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_bilstm_model(
    vocab_size: int = 15000,
    embedding_dim: int = 64,
    max_len: int = 100,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Builds and compiles a Bidirectional LSTM neural network for binary sentiment classification.
    
    Architecture:
    1. Input layer: accepts padded sequence of token IDs of length max_len.
    2. Embedding layer: transforms discrete token IDs into dense semantic vectors.
    3. SpatialDropout1D: drops entire 1D feature maps across channels to prevent co-adaptation.
    4. Bidirectional(LSTM): processes tokens forward and backward, capturing past and future context.
    5. Dense (ReLU) + Dropout: feature transformation and regularization.
    6. Dense (Sigmoid): outputs probability of positive sentiment [0, 1].
    """
    inputs = layers.Input(shape=(max_len,), name="input_sequence")
    
    # Embedding Layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embedding"
    )(inputs)
    
    # 1D Spatial Dropout for sequence embeddings
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
    
    # Bidirectional LSTM Layer
    x = layers.Bidirectional(
        layers.LSTM(
            units=lstm_units,
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=False,
            name="lstm_core"
        ),
        name="bidirectional_lstm"
    )(x)
    
    # Fully connected hidden layer
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)
    
    # Output Sigmoid Layer for Binary Classification
    outputs = layers.Dense(1, activation="sigmoid", name="output_sentiment")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Customer_Review_BiLSTM")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    test_model = build_bilstm_model()
    test_model.summary()
