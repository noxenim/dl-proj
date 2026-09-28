# Customer Review Sentiment Analysis using Deep Learning (BiLSTM)

An end-to-end, production-grade Deep Learning project that analyzes customer reviews and predicts positive or negative sentiment using a custom **Bidirectional Long Short-Term Memory (BiLSTM)** neural network trained on the **Amazon Reviews Polarity** dataset.

---

## 🌟 Key Features

1. **Custom BiLSTM Architecture**:
   - Built from scratch with TensorFlow and Keras.
   - Text Tokenization + Sequence Padding.
   - 64-dimensional Word Embedding Layer.
   - SpatialDropout1D + Bidirectional LSTM (64 forward + 64 backward units).
   - Dense hidden layer + Dropout (0.3) + Sigmoid output for binary classification.

2. **Interactive Streamlit Web Application**:
   - **✍️ Single Review Analysis**: Real-time review classification with confidence gauge meter, raw probabilities, and cleaned sequence details.
   - **📂 Bulk Review Analysis**: Upload CSV files or use pre-loaded multi-domain customer reviews for fast batch inference with progress tracking and CSV export.
   - **📊 Sentiment Dashboard**: Executive KPIs, donut charts, probability distribution histograms, and product category breakdowns.
   - **🔬 Model Performance & Architecture**: Live test set metrics (Accuracy, Precision, Recall, F1-Score), confusion matrix heatmap, and training curves.
   - **📚 Academic Viva Guide**: Curated theoretical and practical Q&A for project defense.

3. **High Performance**:
   - **86.79%** Test Accuracy on unseen Amazon product reviews.
   - **90.03%** Precision on positive customer reviews.
   - **86.24%** F1-Score.

---

## 📁 Project Structure

```
dl/
├── artifacts/                  # Trained model artifacts & plots
│   ├── model.keras             # Trained BiLSTM Keras model
│   ├── tokenizer.pickle        # Fitted Keras Tokenizer
│   ├── config.json             # Model hyperparameters & settings
│   ├── metrics.json            # Accuracy, precision, recall, F1, confusion matrix
│   ├── history.json            # Epoch-by-epoch loss & accuracy
│   ├── training_history.png    # Loss & Accuracy learning curves
│   └── confusion_matrix.png    # Test set confusion matrix heatmap
├── data/                       # Datasets
│   └── amazon_polarity_sample.parquet # Benchmark Amazon Reviews Polarity dataset
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning, tokenization, padding pipeline
│   ├── model.py                # BiLSTM neural network architecture
│   └── inference.py            # SentimentPredictor inference engine
├── app.py                      # Streamlit interactive web application
├── train.py                    # Complete training and evaluation script
├── sample_reviews.csv          # Sample dataset for bulk review testing
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── PROJECT_REPORT.md           # Comprehensive Academic Project Report
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. (Optional) Re-train the BiLSTM Model
To retrain the model with customized epochs, sample size, or hyperparameters:
```bash
python train.py
```
The script will process the Amazon dataset, train the network, compute test evaluation metrics, and automatically regenerate all artifacts and visualization plots.

---

## 📊 Evaluation Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **86.79%** |
| **Precision** | **90.03%** |
| **Recall** | **82.75%** |
| **F1-Score** | **86.24%** |
| **Test Set Size** | 4,800 Amazon Reviews |

---

## 🎓 Academic Submission

For submission and viva preparation, refer to:
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): Complete formal report including First Page, Abstract, Introduction, Theoretical Analysis, Code walkthrough, and Results.
