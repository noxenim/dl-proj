# A MAJOR PROJECT REPORT ON
# CUSTOMER REVIEW SENTIMENT ANALYSIS USING BIDIRECTIONAL LONG SHORT-TERM MEMORY (BiLSTM)

---

### **A Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of**
### **BACHELOR OF TECHNOLOGY / MASTER OF SCIENCE**
### **IN**
### **COMPUTER SCIENCE AND ENGINEERING / ARTIFICIAL INTELLIGENCE**

<br>

**Submitted by:**
- **Student Name:** [Your Name]
- **Roll Number / PRN:** [Your Roll Number]
- **Department:** Department of Computer Science & Engineering
- **Institution:** [Your College / University Name]

**Under the Guidance of:**
- **Project Guide:** [Guide Name, Designation]
- **Department:** Department of Computer Science & Engineering
- **Academic Year:** 2025 – 2026

---

<div style="page-break-after: always;"></div>

# TABLE OF CONTENTS

1. **Abstract**
2. **Introduction**
   - 2.1 Problem Statement
   - 2.2 Project Motivation & Significance
   - 2.3 Key Objectives
   - 2.4 Scope and Limitations
3. **Explanation of the Project**
   - 3.1 Background: From Bag-of-Words to Sequential Deep Learning
   - 3.2 Theoretical Formulation of LSTM and BiLSTM
   - 3.3 Text Preprocessing & Embedding Pipeline
   - 3.4 Deep Learning Model Architecture
   - 3.5 Training Methodology & Optimization
   - 3.6 Web Application & System Flow
4. **Code Implementation**
   - 4.1 Data Cleaning & Preprocessing (`src/preprocess.py`)
   - 4.2 BiLSTM Neural Network Model (`src/model.py`)
   - 4.3 Training & Evaluation Pipeline (`train.py`)
   - 4.4 Real-time Inference Engine (`src/inference.py`)
   - 4.5 Interactive Streamlit Web Application (`app.py`)
5. **Results and Evaluation**
   - 5.1 Quantitative Performance Metrics
   - 5.2 Confusion Matrix Breakdown
   - 5.3 Training and Validation Learning Dynamics
   - 5.4 Qualitative Inference Case Studies
   - 5.5 Web Application Demonstration
6. **Conclusion and Future Scope**
7. **Viva Voce Preparation & Important Questions**
8. **References**

---

<div style="page-break-after: always;"></div>

# 1. ABSTRACT

In contemporary e-commerce and retail ecosystems, customer feedback serves as a pivotal driver of product refinement, customer retention, and brand sentiment monitoring. However, manually auditing millions of high-velocity customer reviews across diverse digital storefronts is intractable. Traditional Natural Language Processing (NLP) methodologies—such as Bag-of-Words (BoW) paired with Naive Bayes or Support Vector Machines (SVM)—rely heavily on term frequencies while discarding syntactical word order, semantic context, and negation structures (e.g., *"not bad"* vs. *"not good"*).

To overcome these structural limitations, this project presents an end-to-end Deep Learning system for binary customer review sentiment classification using a **Bidirectional Long Short-Term Memory (BiLSTM)** network. Built upon the benchmark **Amazon Reviews Polarity** dataset, our pipeline implements HTML stripping, tokenization, vocabulary mapping, sequence padding, and a learnable 64-dimensional Word Embedding layer. The bidirectional recurrent core processes text sequences in both chronological and anti-chronological order, capturing contextual dependencies preceding and following each token. 

Trained using the Adam optimizer with Binary Crossentropy loss, the model attains a test accuracy of **86.79%**, a precision of **90.03%**, and an F1-score of **86.24%** across 4,800 unseen test reviews. Beyond offline experimentation, the project delivers a production-ready, interactive **Streamlit web application** featuring single-review real-time sentiment scoring with visual confidence gauges, bulk CSV review batch classification with progress tracking, an executive sentiment analytics dashboard, and automated performance reporting.

---

# 2. INTRODUCTION

## 2.1 Problem Statement
Customer reviews contain nuanced linguistic elements including colloquialisms, slang, domain-specific adjectives, and intricate negation patterns. Standard classification algorithms treat documents as unordered collections of tokens, making them vulnerable to misclassifying sentences where polarity is modulated by negations or contrastive conjunctions (*"The camera looks decent, but the battery died within two hours"*). The problem addressed by this project is to develop an automated sequential deep learning model capable of capturing semantic nuances and temporal dependencies in unstructured textual reviews, accurately categorizing them into **Positive** or **Negative** sentiment polarities with associated confidence scores.

## 2.2 Project Motivation & Significance
For modern business operations, understanding customer sentiment provides actionable intelligence for:
1. **Quality Assurance**: Promptly discovering recurring hardware faults or software defects in newly launched products.
2. **Customer Service Triage**: Automatically prioritizing dissatisfied customer grievances to reduce churn rate.
3. **Market Competitive Intelligence**: Benchmarking customer satisfaction against competing brands.
4. **Automated Feedback Aggregation**: Generating real-time sentiment metrics for management dashboards without manual human intervention.

## 2.3 Key Objectives
- **Data Engineering**: Construct an automated data preprocessing pipeline that cleans, normalizes, and tokenizes raw customer reviews.
- **Deep Neural Architecture**: Implement a custom Bidirectional LSTM architecture incorporating Word Embeddings, Spatial Dropout, and Dense classification layers.
- **Robust Model Training**: Train the model on balanced samples from the Amazon Reviews Polarity dataset with EarlyStopping and ModelCheckpoint mechanisms.
- **Comprehensive Evaluation**: Rigorously evaluate performance through accuracy, precision, recall, F1-score, and confusion matrix analysis.
- **Interactive Application Deployment**: Build an intuitive, responsive Streamlit web application supporting single review inference, batch CSV ingestion, data filtering, and dynamic visualization.

## 2.4 Scope and Limitations
- **Scope**: Supervised binary sentiment classification (Positive vs. Negative) of English-language product and service customer reviews.
- **Limitations**: The model is trained on binary polarities and does not currently extract multi-aspect granularity (e.g., separating shipping sentiment from product quality sentiment in the same sentence) or neutral labels.

---

# 3. EXPLANATION OF THE PROJECT

## 3.1 Background: From Bag-of-Words to Sequential Deep Learning
Early text classification models utilized discrete lexical features:
- **One-Hot Encoding**: Represented words as high-dimensional, orthogonal sparse vectors with dimension $|V|$. It suffered from the curse of dimensionality and possessed zero semantic proximity between synonyms (e.g., $\text{dist}(\text{excellent}, \text{great}) = \text{dist}(\text{excellent}, \text{horrible})$).
- **Bag-of-Words / TF-IDF**: Measured word frequencies while discarding word ordering completely. A phrase such as *"not good, really terrible"* and *"not terrible, really good"* shared almost identical term frequencies despite conveying diametrically opposed sentiments.
- **Standard Recurrent Neural Networks (RNNs)**: Introduced cyclical connections to process sequential inputs token-by-token. However, when backpropagating gradients over extended sequences, repeated matrix multiplications caused gradients to either vanish exponentially to zero or explode towards infinity, preventing standard RNNs from capturing long-term dependencies.

## 3.2 Theoretical Formulation of LSTM and BiLSTM

### Long Short-Term Memory (LSTM) Mechanics
LSTMs circumvent the vanishing gradient problem by maintaining an internal memory cell vector $C_t$ regulated by three non-linear gates:

$$\begin{aligned}
\text{Forget Gate:} \quad & f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
\text{Input Gate:} \quad & i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\text{Candidate State:} \quad & \tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \\
\text{Cell State Update:} \quad & C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
\text{Output Gate:} \quad & o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
\text{Hidden State:} \quad & h_t = o_t \odot \tanh(C_t)
\end{aligned}$$

Where:
- $\sigma$ denotes the sigmoid activation function producing values in $[0, 1]$.
- $\odot$ represents element-wise Hadamard product.
- $h_t$ is the hidden state vector passed to the subsequent time step.

### Why Bidirectional LSTM (BiLSTM)?
A standard unidirectional LSTM processes text strictly in the forward direction ($t = 1 \to T$). Consequently, when interpreting word $x_t$, the network only possesses historical context. 
In human language, however, the sentiment of a word frequently depends on subsequent words (future context). For instance:
- *"Although the design was **flawed**, the customer support team was **exceptional**."*

A **Bidirectional LSTM** employs two independent recurrent hidden layers:
1. **Forward LSTM Layer**: Computes hidden state sequence $\overrightarrow{h}_t$ from $t = 1$ to $T$.
2. **Backward LSTM Layer**: Computes hidden state sequence $\overleftarrow{h}_t$ from $t = T$ to $1$.

The final representation for each token is the concatenation of both hidden states:
$$h_t = \left[ \overrightarrow{h}_t \,;\, \overleftarrow{h}_t \right]$$

This dual-directional formulation enables the network to make holistic sentiment inferences based on the complete contextual window of the review.

```
                    ┌─────────────────────────┐
                    │     Output (Sigmoid)    │
                    └────────────▲────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │     Dense (64, ReLU)    │
                    └────────────▲────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Bidirectional LSTM    │
                    │   [ h_forward ; h_back ]│
                    └───────▲─────────▲───────┘
                     ┌──────┘         └──────┐
            Forward LSTM Layer        Backward LSTM Layer
            ( t = 1 ──► T )           ( t = T ──► 1 )
                     └──────┬─────────┬──────┘
                    ┌───────┴─────────┴───────┐
                    │    SpatialDropout1D     │
                    └────────────▲────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │     Word Embedding      │
                    │  (15,000 Vocab x 64 D)  │
                    └────────────▲────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Padded Token Sequence │
                    │   (Fixed Length: 100)   │
                    └─────────────────────────┘
```

## 3.3 Text Preprocessing & Embedding Pipeline
1. **HTML & Entity Stripping**: Removes web markup, XML artifacts, and unescapes entities (e.g., `&amp;` to `&`).
2. **Noise Removal**: Eliminates hyperlinks and unneeded non-alphanumeric punctuation while carefully preserving apostrophes and negation markers (`don't`, `can't`, `not`, `never`).
3. **Case Normalization**: Converts all text to lowercase to ensure vocabulary consistency (`Great` and `great` map to the same token index).
4. **Tokenization**: Maps the top 15,000 most frequent vocabulary words to contiguous integer indices $1, 2, \dots, 15000$. Out-of-vocabulary terms are mapped to an explicit `<OOV>` token.
5. **Sequence Padding / Truncation**: Standardizes review lengths to $L = 100$ tokens. Shorter reviews are post-padded with zeros ($0$), and longer reviews are truncated.

## 3.4 Deep Learning Model Architecture
- **Input Layer**: Accepts integer vectors of shape `(batch_size, 100)`.
- **Embedding Layer**: Projects integer tokens into continuous 64-dimensional vectors `(batch_size, 100, 64)`.
- **SpatialDropout1D (0.2)**: Regularization technique that drops entire feature channels across the sequence dimension, preventing co-adaptation of correlated word representations.
- **Bidirectional LSTM Layer**: Comprises 64 forward units and 64 backward units, returning a concatenated 128-dimensional output vector `(batch_size, 128)`.
- **Dense Layer (64 units, ReLU)**: Applies non-linear transformations to extract high-level feature combinations.
- **Standard Dropout (0.3)**: Mitigates overfitting on dense parameters.
- **Dense Output Layer (1 unit, Sigmoid)**: Emits scalar $p \in [0, 1]$ representing the estimated probability that the customer review expresses positive sentiment.

## 3.5 Training Methodology & Optimization
- **Loss Function**: Binary Crossentropy:
  $$\mathcal{L}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Optimization**: Adam (Adaptive Moment Estimation) optimizer with learning rate $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.
- **Early Stopping**: Monitors validation loss (`val_loss`) with a patience of 2 epochs to prevent over-fitting and automatically restores the best model weights.
- **Model Checkpointing**: Persists the highest-performing model snapshot (`model.keras`) based on peak validation accuracy.

---

# 4. CODE IMPLEMENTATION

The project is structured with modularity, clean code practices, and separation of concerns.

## 4.1 Data Cleaning & Preprocessing (`src/preprocess.py`)
This module handles all string normalization, Keras tokenizer serialization, and sequence conversion:

```python
import re
import html
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text: str) -> str:
    """Cleans raw customer review text by stripping HTML, URLs, and excess spaces."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s'!\?.,-]", ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fit_and_save_tokenizer(texts, tokenizer_path: str, vocab_size: int = 15000) -> Tokenizer:
    """Fits Keras Tokenizer on corpus and serializes to disk."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer

def transform_texts(texts, tokenizer: Tokenizer, max_len: int = 100) -> np.ndarray:
    """Converts raw string list into padded numerical sequence tensor."""
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded
```

## 4.2 BiLSTM Neural Network Model (`src/model.py`)
Defines the functional neural network architecture:

```python
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
    """Constructs and compiles the Bidirectional LSTM classification model."""
    inputs = layers.Input(shape=(max_len,), name="input_sequence")
    
    # 1. Embedding Layer
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="word_embedding")(inputs)
    
    # 2. Regularization via SpatialDropout
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
    
    # 3. Bidirectional LSTM Layer
    x = layers.Bidirectional(
        layers.LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.0, return_sequences=False),
        name="bidirectional_lstm"
    )(x)
    
    # 4. Dense feature extraction & classification
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sentiment")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Customer_Review_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

## 4.3 Training & Evaluation Pipeline (`train.py`)
Orchestrates end-to-end dataset ingestion, model fitting, metric computation, and artifact persistence:

```python
# Excerpt from train.py:
def run_training_pipeline():
    df = pd.read_parquet("data/amazon_polarity_sample.parquet")
    # Stratified balanced sampling
    pos_df = df[df["label"] == 1].sample(n=12000, random_state=42)
    neg_df = df[df["label"] == 0].sample(n=12000, random_state=42)
    subset_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    subset_df["full_text"] = subset_df["title"].fillna("") + " " + subset_df["content"].fillna("")
    
    cleaned_texts = [clean_text(t) for t in subset_df["full_text"].values]
    labels = subset_df["label"].values.astype(np.float32)
    
    train_texts, test_texts, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    tokenizer = fit_and_save_tokenizer(train_texts, "artifacts/tokenizer.pickle", vocab_size=15000)
    x_train = transform_texts(train_texts, tokenizer, max_len=100)
    x_test = transform_texts(test_texts, tokenizer, max_len=100)
    
    model = build_bilstm_model(vocab_size=15000, embedding_dim=64, max_len=100, lstm_units=64)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath="artifacts/model.keras", monitor="val_accuracy", save_best_only=True)
    ]
    
    history = model.fit(x_train, y_train, validation_split=0.15, epochs=4, batch_size=64, callbacks=callbacks)
    
    # Test evaluation
    y_pred_probs = model.predict(x_test, batch_size=64).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    # Compute accuracy, precision, recall, f1, and confusion matrix
```

## 4.4 Real-time Inference Engine (`src/inference.py`)
Encapsulates model scoring for single inputs and vectorized batch processing:

```python
class SentimentPredictor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.tokenizer = load_tokenizer(os.path.join(artifacts_dir, "tokenizer.pickle"))
        self.model = tf.keras.models.load_model(os.path.join(artifacts_dir, "model.keras"))
        self.max_len = 100
        
    def predict_one(self, review_text: str) -> dict:
        cleaned = clean_text(review_text)
        padded = transform_texts([review_text], self.tokenizer, max_len=self.max_len)
        prob = float(self.model.predict(padded, verbose=0)[0][0])
        is_positive = prob >= 0.5
        return {
            "sentiment": "Positive" if is_positive else "Negative",
            "confidence": prob if is_positive else (1.0 - prob),
            "probability": prob,
            "cleaned_text": cleaned,
            "word_count": len(cleaned.split())
        }
```

---

# 5. RESULTS AND EVALUATION

## 5.1 Quantitative Performance Metrics
The BiLSTM network was evaluated on a strictly held-out test split of **4,800 customer reviews** from the Amazon Reviews Polarity benchmark dataset.

| Evaluation Metric | Test Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **86.79%** | Overall percentage of correct sentiment classifications across both classes. |
| **Precision (Positive)** | **90.03%** | Proportion of reviews predicted as positive that were truly positive. |
| **Recall (Positive)** | **82.75%** | Proportion of actual positive reviews correctly identified by the network. |
| **F1-Score** | **86.24%** | Harmonic mean of precision and recall, demonstrating balanced performance. |
| **Test Sample Size** | **4,800 Reviews** | Stratified test split (2,400 Positive, 2,400 Negative). |

### Complete Classification Report
```
              precision    recall  f1-score   support

    Negative       0.84      0.91      0.87      2400
    Positive       0.90      0.83      0.86      2400

    accuracy                           0.87      4800
   macro avg       0.87      0.87      0.87      4800
weighted avg       0.87      0.87      0.87      4800
```

## 5.2 Confusion Matrix Breakdown
The confusion matrix on the 4,800 unseen test reviews is summarized below:

| | Predicted Negative | Predicted Positive | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative** | **2,180** (TN) | **220** (FP) | 2,400 |
| **Actual Positive** | **414** (FN) | **1,986** (TP) | 2,400 |
| **Total Predicted** | 2,594 | 2,206 | 4,800 |

- **True Negatives (TN = 2,180)**: 90.83% of all genuine negative reviews were accurately identified.
- **False Positives (FP = 220)**: Low false alarm rate of only 9.17% on negative feedback.
- **True Positives (TP = 1,986)**: Over 82.7% of genuine positive reviews were identified.
- **False Negatives (FN = 414)**: Mildly under-predicted on subtle positive phrasing.

## 5.3 Training and Validation Learning Dynamics
During model fitting:
- In **Epoch 1**, the model achieved a training accuracy of **91.69%** and a validation accuracy of **87.36%**, demonstrating rapid convergence enabled by the Bidirectional LSTM recurrence.
- By **Epoch 2 & 3**, validation accuracy stabilized between **86.0%** and **87.4%**.
- The `EarlyStopping` callback automatically halted training at Epoch 3 and restored the best model weights, ensuring optimal generalization while avoiding over-fitting.

## 5.4 Qualitative Inference Case Studies
To evaluate real-world contextual reasoning, several challenging sentences were tested:

| Test Review Text | True Sentiment | Predicted Sentiment | Confidence | Contextual Analysis |
| :--- | :---: | :---: | :---: | :--- |
| *"The product quality is excellent. Delivery was fast, and I am very satisfied with my purchase."* | Positive | **Positive** | **93.35%** | Correctly associates strong positive adjectives (*"excellent"*, *"satisfied"*). |
| *"Stopped working completely after three days. Cheap plastic and terrible customer service!"* | Negative | **Negative** | **98.06%** | High-confidence detection of critical failure and poor support. |
| *"The product is good."* | Positive | **Positive** | **92.04%** | Baseline positive sentiment benchmark. |
| *"The product is not good at all, very bad."* | Negative | **Negative** | **69.91%** | Successfully resolves the negation *"not good"*, overriding the word *"good"*. |
| *"A masterpiece of engineering, loved every minute of using it."* | Positive | **Positive** | **95.09%** | Correctly handles metaphorical positive phrasing (*"masterpiece"*). |
| *"Total waste of money, broke on the first day."* | Negative | **Negative** | **99.33%** | Overwhelming negative classification for severe product defect. |

## 5.5 Web Application Demonstration
The deployed Streamlit application features five dedicated modules:
1. **Single Review Analyzer**: Features an interactive text console, quick preset buttons, real-time prediction badge, and an animated Plotly probability gauge meter.
2. **Bulk Reviews Processor**: Provides drag-and-drop CSV upload, automated review column detection, vectorized batch inference with progress animation, metric summaries, and one-click analyzed CSV export.
3. **Executive Sentiment Dashboard**: Displays key metric KPI cards, an interactive sentiment donut chart, a probability distribution histogram, and product category comparisons.
4. **Model Performance Hub**: Displays live test metrics, the saved confusion matrix heatmap, and training/validation loss/accuracy curves.
5. **Viva & Documentation Guide**: In-app theoretical notes and viva voce Q&A for academic review.

---

# 6. CONCLUSION AND FUTURE SCOPE

## 6.1 Conclusion
This project successfully designed, implemented, and deployed a deep learning-based **Customer Review Sentiment Analysis** application powered by a **Bidirectional Long Short-Term Memory (BiLSTM)** network. By pairing a learnable word embedding layer with bidirectional recurrent units, the model demonstrated an ability to capture syntactic context and tricky negation structures that traditional machine learning algorithms fail to handle. Achieving **86.79% accuracy** and **90.03% precision** on the Amazon Reviews Polarity benchmark, the system bridges theoretical deep learning research with practical software engineering through a responsive Streamlit web application.

## 6.2 Future Scope & Enhancements
1. **Aspect-Based Sentiment Analysis (ABSA)**: Extending the model to extract multi-granular sentiment for discrete attributes (e.g., distinguishing between *"Battery: Negative"* and *"Screen: Positive"* in a single electronic review).
2. **Transformer Fine-Tuning**: Comparing the BiLSTM baseline against fine-tuned BERT or RoBERTa architectures for even higher contextual accuracy.
3. **Multi-Class Granularity**: Transitioning from binary polarity to a 5-star rating prediction scale (1-star through 5-stars).
4. **Multilingual Support**: Integrating multilingual embeddings to process customer reviews in global languages.

---

# 7. VIVA VOCE PREPARATION & IMPORTANT QUESTIONS

### Q1. Why is BiLSTM preferred over standard Feedforward Neural Networks (ANN)?
**Answer:** ANNs treat input features as independent vectors of fixed dimensions with no temporal or positional relationships. Language, however, is sequential—the position and order of words dictate meaning. BiLSTM maintains sequential state memory across time steps, enabling the network to learn grammar, context, and dependencies between words.

### Q2. What is the fundamental difference between LSTM and BiLSTM?
**Answer:** A standard LSTM processes sequences chronologically from left to right ($t = 1 \dots T$). A Bidirectional LSTM runs two concurrent LSTM layers: one forwards ($1 \dots T$) and one backwards ($T \dots 1$). Their outputs are concatenated at each time step ($h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$), giving the network simultaneous access to both past and future context.

### Q3. How does the LSTM Cell solve the Vanishing Gradient problem?
**Answer:** Standard RNNs repeatedly multiply gradients by the weight matrix $W$ during Backpropagation Through Time (BPTT), causing gradients to decay exponentially ($W^T \to 0$). LSTMs introduce the **Cell State ($C_t$)**, where updates are predominantly **additive** ($C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$). Because additions distribute gradients equally without exponential attenuation, long-range gradients can flow backwards without vanishing.

### Q4. What do the three gates in an LSTM cell do?
**Answer:**
1. **Forget Gate ($f_t$):** Calculates a sigmoid mask $[0, 1]$ determining how much previous cell memory to erase.
2. **Input Gate ($i_t$):** Regulates how much newly arrived information from current input $x_t$ should enter the cell state.
3. **Output Gate ($o_t$):** Decides what filtered portion of the updated cell state $C_t$ is emitted as the hidden state $h_t$.

### Q5. What is the difference between Word Embeddings and One-Hot Encoding?
**Answer:** One-Hot Encoding produces high-dimensional, sparse vectors where all vectors are equidistant and orthogonal, carrying zero semantic information. Word Embeddings project words into a dense, lower-dimensional space (e.g., 64 dimensions), where semantically related words are located close to each other, allowing the model to generalize across synonyms.

### Q6. Why was SpatialDropout1D utilized after the Embedding Layer?
**Answer:** Standard Dropout drops individual scalar elements within each embedding vector randomly. Because values inside a single word embedding are closely tied, dropping individual features does not effectively regularize the network. SpatialDropout1D drops entire 1D word feature maps, preventing co-adaptation across adjacent tokens and improving generalization.

### Q7. Why is Binary Crossentropy used as the loss function instead of Mean Squared Error (MSE)?
**Answer:** For binary classification with a Sigmoid output, MSE produces non-convex optimization surfaces with plateaus where gradients vanish when predictions are confidently wrong. Binary Crossentropy penalizes incorrect probabilistic predictions logarithmically, producing steep gradients that guarantee fast and reliable convergence.

### Q8. What does an F1-score of 86.24% indicate?
**Answer:** The F1-score is the harmonic mean of Precision (90.03%) and Recall (82.75%). It demonstrates that the model maintains both high accuracy when declaring positive sentiment (low false alarms) while successfully capturing a large majority of actual positive reviews (low false negatives).

---

# 8. REFERENCES
1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
2. Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM networks. *Neural Networks*, 18(5-6), 602-610.
3. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems (NeurIPS)* (Amazon Reviews Polarity dataset benchmark).
4. Chollet, F. et al. (2015). *Keras: Deep Learning for humans*. GitHub repository.
5. Abadi, M. et al. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*.
