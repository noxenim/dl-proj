
"""
Customer Review Sentiment Analysis Web Application.
Built with Streamlit and deep learning (Bidirectional LSTM).
"""

import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Sentify AI | Customer Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Light adaptive modern CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .positive-badge {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .negative-badge {
        display: inline-block;
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .review-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Initializes and caches the trained BiLSTM inference engine."""
    from src.inference import SentimentPredictor
    try:
        predictor = SentimentPredictor(artifacts_dir="artifacts")
        return predictor, None
    except Exception as e:
        return None, str(e)


def load_artifacts_data():
    """Loads metrics, configuration, and history JSON files if available."""
    metrics_path = "artifacts/metrics.json"
    config_path = "artifacts/config.json"
    history_path = "artifacts/history.json"
    
    metrics = None
    config = None
    history = None
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
            
    return metrics, config, history


# Sidebar Navigation
st.sidebar.markdown("## 🧠 Sentify BiLSTM")
st.sidebar.caption("Customer Sentiment Analysis System")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "✍️ Single Review Analysis",
        "📂 Bulk Reviews (CSV Upload)",
        "📊 Sentiment Dashboard",
        "🔬 Model Evaluation & Architecture",
        "📚 Project Viva & Report Summary"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Model Specs")
st.sidebar.markdown("""
- **Algorithm**: Bidirectional LSTM
- **Dataset**: Amazon Reviews Polarity
- **Vocab Size**: 15,000 words
- **Embedding Dim**: 64 vectors
- **Context Window**: Bidirectional
""")

predictor, error_msg = get_predictor()

# ==============================================================================
# 1. SINGLE REVIEW ANALYSIS
# ==============================================================================
if nav_choice == "✍️ Single Review Analysis":
    st.markdown('<div class="main-header">Real-Time Review Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Test individual customer reviews using our trained Bidirectional Long Short-Term Memory neural network.</div>', unsafe_allow_html=True)
    
    if error_msg:
        st.error(f"⚠️ Model artifacts not ready: {error_msg}. Please run `python train.py` first.")
    else:
        # Predefined Quick Presets
        st.markdown("**💡 Quick Examples:**")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        default_text = "The product quality is excellent. Delivery was fast, and I am very satisfied with my purchase."
        
        if col_btn1.button("⭐ Glowing Review"):
            default_text = "The product quality is excellent. Delivery was fast, and I am very satisfied with my purchase."
        if col_btn2.button("❌ Defective / Frustrated"):
            default_text = "Stopped working completely after three days. Cheap plastic and terrible customer service!"
        if col_btn3.button("🔄 Complex / Negation"):
            default_text = "The packaging was nice, but the item itself was not good at all. I would not buy this again."
        if col_btn4.button("🎧 Tech / Gadgets"):
            default_text = "Sound stage is immersive with punchy bass. Best noise cancellation at this price point."
            
        review_input = st.text_area(
            "Customer Review Text:",
            value=default_text,
            height=130,
            placeholder="Type or paste any product or service review here..."
        )
        
        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        
        if review_input.strip() and (analyze_btn or review_input):
            with st.spinner("Analyzing text sequence through BiLSTM layers..."):
                res = predictor.predict_one(review_input)
                
            sentiment = res["sentiment"]
            confidence = res["confidence"]
            prob = res["probability"]
            
            st.divider()
            st.markdown("### 🎯 Prediction Results")
            
            col_res1, col_res2, col_res3 = st.columns([1.2, 1.8, 1])
            
            with col_res1:
                st.markdown("<br>", unsafe_allow_html=True)
                if sentiment == "Positive":
                    st.markdown(f'<div class="positive-badge">🟢 Positive Sentiment</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="negative-badge">🔴 Negative Sentiment</div>', unsafe_allow_html=True)
                
                st.markdown(f"#### **Confidence: {confidence * 100:.1f}%**")
                st.caption(f"Raw Sigmoid Output: `{prob:.4f}`")
                
            with col_res2:
                # Gauge / Progress Bar
                st.markdown("**Positive Sentiment Probability Meter:**")
                st.progress(float(prob))
                
                # Visual probability split
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': "%", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#10B981" if prob >= 0.5 else "#EF4444"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FEE2E2"},
                            {'range': [50, 100], 'color': "#D1FAE5"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_res3:
                st.markdown("**Sequence Breakdown**")
                st.markdown(f"- **Word Count:** `{res['word_count']}`")
                st.markdown(f"- **Max Length:** `100 tokens`")
                st.markdown(f"- **Padding:** `Post-padded`")
                st.markdown(f"- **RNN Cells:** `64 Bidirectional`")
                
            with st.expander("🔍 Cleaned & Normalized Tokens"):
                st.code(res["cleaned_text"], language="text")

# ==============================================================================
# 2. BULK REVIEWS ANALYSIS
# ==============================================================================
elif nav_choice == "📂 Bulk Reviews (CSV Upload)":
    st.markdown('<div class="main-header">Bulk Review Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a dataset or CSV file to batch classify thousands of customer reviews simultaneously.</div>', unsafe_allow_html=True)
    
    if error_msg:
        st.error(f"⚠️ Model artifacts not ready: {error_msg}")
    else:
        col_up, col_samp = st.columns([2, 1])
        
        with col_up:
            uploaded_file = st.file_uploader("Upload CSV File containing reviews", type=["csv"])
            
        with col_samp:
            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                use_headphones = st.button("🎧 Single-Product (30 Reviews)", use_container_width=True)
            with col_b2:
                use_sample = st.button("🛍️ Multi-Product (25 Reviews)", use_container_width=True)
            
        df_to_analyze = None
        if uploaded_file is not None:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded successfully ({len(df_to_analyze)} rows).")
        elif use_headphones and os.path.exists("bulk_test_reviews.csv"):
            df_to_analyze = pd.read_csv("bulk_test_reviews.csv")
            st.info("Loaded AuraSound Pro Headphones review test dataset (30 diverse reviews).")
        elif use_sample and os.path.exists("sample_reviews.csv"):
            df_to_analyze = pd.read_csv("sample_reviews.csv")
            st.info("Loaded pre-configured multi-product review dataset (25 reviews).")
                
        if df_to_analyze is not None:
            st.write("### Data Preview")
            st.dataframe(df_to_analyze.head(4), use_container_width=True)
            
            # Select review column
            possible_cols = [c for c in df_to_analyze.columns if any(k in c.lower() for k in ["review", "text", "content", "comment", "feedback"])]
            default_col_idx = 0
            if possible_cols:
                default_col_idx = df_to_analyze.columns.get_loc(possible_cols[0])
                
            text_column = st.selectbox(
                "Select column containing customer reviews:",
                df_to_analyze.columns.tolist(),
                index=default_col_idx
            )
            
            if st.button("⚡ Run BiLSTM Batch Inference", type="primary"):
                reviews_list = df_to_analyze[text_column].astype(str).tolist()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Preprocessing review sequences...")
                
                t_start = time.time()
                batch_preds = predictor.predict_batch(reviews_list)
                progress_bar.progress(100)
                elapsed = time.time() - t_start
                status_text.text(f"Batch prediction completed in {elapsed:.2f} seconds!")
                
                # Append results to DataFrame
                df_to_analyze["Predicted_Sentiment"] = [p["sentiment"] for p in batch_preds]
                df_to_analyze["Confidence"] = [f"{p['confidence']*100:.1f}%" for p in batch_preds]
                df_to_analyze["Positive_Probability"] = [p["probability"] for p in batch_preds]
                
                # Store in session state for Dashboard view
                st.session_state["analyzed_df"] = df_to_analyze
                
                st.divider()
                st.markdown("### 📊 Bulk Summary")
                
                pos_count = (df_to_analyze["Predicted_Sentiment"] == "Positive").sum()
                neg_count = (df_to_analyze["Predicted_Sentiment"] == "Negative").sum()
                total = len(df_to_analyze)
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Total Reviews", f"{total:,}")
                col_m1.caption(f"Processed in {elapsed:.2f}s")
                col_m2.metric("Positive Reviews", f"{pos_count:,}", f"{(pos_count/total)*100:.1f}%")
                col_m3.metric("Negative Reviews", f"{neg_count:,}", f"-{(neg_count/total)*100:.1f}%")
                avg_conf = np.mean([p["confidence"] for p in batch_preds])
                col_m4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                
                # Filter view
                filter_choice = st.radio("Filter Results Table:", ["All", "Positive Only", "Negative Only"], horizontal=True)
                
                filtered_df = df_to_analyze
                if filter_choice == "Positive Only":
                    filtered_df = df_to_analyze[df_to_analyze["Predicted_Sentiment"] == "Positive"]
                elif filter_choice == "Negative Only":
                    filtered_df = df_to_analyze[df_to_analyze["Predicted_Sentiment"] == "Negative"]
                    
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download analyzed results
                csv_data = df_to_analyze.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analyzed CSV with Predictions",
                    data=csv_data,
                    file_name="sentify_analyzed_reviews.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==============================================================================
# 3. SENTIMENT DASHBOARD
# ==============================================================================
elif nav_choice == "📊 Sentiment Dashboard":
    st.markdown('<div class="main-header">Sentiment Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive visual analytics displaying sentiment distribution, confidence dynamics, and category insights.</div>', unsafe_allow_html=True)
    
    # Retrieve analyzed data or load default sample
    df_dashboard = None
    if "analyzed_df" in st.session_state:
        df_dashboard = st.session_state["analyzed_df"]
    elif os.path.exists("sample_reviews.csv") and predictor is not None:
        df_temp = pd.read_csv("sample_reviews.csv")
        batch_preds = predictor.predict_batch(df_temp["customer_review"].tolist())
        df_temp["Predicted_Sentiment"] = [p["sentiment"] for p in batch_preds]
        df_temp["Confidence_Val"] = [p["confidence"] for p in batch_preds]
        df_temp["Positive_Probability"] = [p["probability"] for p in batch_preds]
        df_dashboard = df_temp
        
    if df_dashboard is None:
        st.info("No data available yet. Please run Single Analysis or Bulk Analysis first.")
    else:
        pos_cnt = (df_dashboard["Predicted_Sentiment"] == "Positive").sum()
        neg_cnt = (df_dashboard["Predicted_Sentiment"] == "Negative").sum()
        tot = len(df_dashboard)
        
        # Summary KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Analyzed", f"{tot:,}")
        kpi2.metric("Positive Sentiment", f"{pos_cnt:,}", f"{(pos_cnt/tot)*100:.1f}%")
        kpi3.metric("Negative Sentiment", f"{neg_cnt:,}", f"-{(neg_cnt/tot)*100:.1f}%")
        kpi4.metric("Net Sentiment Score", f"{((pos_cnt - neg_cnt)/tot)*100:+.1f}%")
        
        st.divider()
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # Donut Chart
            fig_donut = px.pie(
                values=[pos_cnt, neg_cnt],
                names=["Positive", "Negative"],
                color=["Positive", "Negative"],
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
                hole=0.55,
                title="Overall Sentiment Distribution"
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_c2:
            # Probability / Confidence Distribution
            if "Positive_Probability" in df_dashboard.columns:
                fig_hist = px.histogram(
                    df_dashboard,
                    x="Positive_Probability",
                    nbins=20,
                    color="Predicted_Sentiment",
                    color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
                    title="Model Probability Distribution (0 = Strong Negative, 1 = Strong Positive)",
                    labels={"Positive_Probability": "Predicted Probability of Positive Sentiment"}
                )
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
                
        # Category Breakdown if category column exists
        category_col = None
        for c in df_dashboard.columns:
            if "category" in c.lower():
                category_col = c
                break
                
        if category_col:
            st.markdown("### 🏷️ Sentiment by Product Category")
            cat_df = df_dashboard.groupby([category_col, "Predicted_Sentiment"]).size().reset_index(name="Count")
            fig_cat = px.bar(
                cat_df,
                x=category_col,
                y="Count",
                color="Predicted_Sentiment",
                barmode="group",
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444"},
                title=f"Review Sentiments Across {category_col.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

# ==============================================================================
# 4. MODEL EVALUATION & ARCHITECTURE
# ==============================================================================
elif nav_choice == "🔬 Model Evaluation & Architecture":
    st.markdown('<div class="main-header">BiLSTM Architecture & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics on unseen test data from the Amazon Reviews Polarity dataset.</div>', unsafe_allow_html=True)
    
    metrics, config, history = load_artifacts_data()
    
    if metrics:
        st.markdown("### 🏆 Test Set Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        m3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        m4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")
        st.caption(f"Evaluated on {metrics.get('total_test_samples', 0):,} held-out customer reviews.")
        
    st.divider()
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("#### 📈 Training & Validation Curves")
        if os.path.exists("artifacts/training_history.png"):
            st.image("artifacts/training_history.png", use_container_width=True)
        else:
            st.info("Training history plot will appear after training finishes.")
            
    with col_p2:
        st.markdown("#### 🎯 Confusion Matrix")
        if os.path.exists("artifacts/confusion_matrix.png"):
            st.image("artifacts/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Confusion matrix plot will appear after training finishes.")
            
    st.divider()
    st.markdown("### 🧱 Deep Learning Architecture Breakdown")
    st.markdown("""
| Layer | Layer Type | Output Dimension | Description & Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | `InputLayer` | `(None, 100)` | Padded sequence of word token indices (fixed length 100). |
| **Embedding** | `Embedding` | `(None, 100, 64)` | Maps 15,000 discrete vocabulary tokens into dense 64-D vectors. |
| **SpatialDropout1D** | `SpatialDropout1D` | `(None, 100, 64)` | Drops entire 1D word feature maps (rate=0.2) to prevent co-adaptation. |
| **BiLSTM** | `Bidirectional(LSTM)`| `(None, 128)` | 64 forward + 64 backward LSTM units reading sequential context both ways. |
| **Dense** | `Dense (ReLU)` | `(None, 64)` | Non-linear dense layer extracting higher-level sentiment features. |
| **Dropout** | `Dropout` | `(None, 64)` | 30% dropout regularization preventing overfitting on dense features. |
| **Output** | `Dense (Sigmoid)` | `(None, 1)` | Sigmoid activation function outputting positive sentiment probability [0, 1]. |
    """)

# ==============================================================================
# 5. PROJECT VIVA & REPORT SUMMARY
# ==============================================================================
elif nav_choice == "📚 Project Viva & Report Summary":
    st.markdown('<div class="main-header">Academic Report & Viva Voce Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive guide covering theory, implementation, and frequently asked viva questions.</div>', unsafe_allow_html=True)
    
    with st.expander("📖 1. Project Abstract", expanded=True):
        st.markdown("""
        **Abstract:**
        Customer review sentiment analysis is an indispensable component of modern e-commerce and customer experience management. 
        Traditional machine learning models (e.g., Naive Bayes, Logistic Regression, Bag of Words) fail to capture syntactic ordering, word dependencies, and negations (e.g., *"not good"* vs *"good"*). 
        In this project, we design, train, and deploy an end-to-end Deep Learning system based on a **Bidirectional Long Short-Term Memory (BiLSTM)** network trained on the benchmark **Amazon Reviews Polarity dataset**. 
        Our solution incorporates text normalization, tokenization, a learnable word embedding layer, bidirectional temporal feature extraction, and a production-grade Streamlit web application capable of real-time single review scoring and high-throughput bulk CSV analysis.
        """)
        
    with st.expander("❓ 2. Top Viva Voce Questions & Answers", expanded=True):
        st.markdown("""
        **Q1: Why use Bidirectional LSTM instead of a standard unidirectional LSTM?**
        - *Answer:* A unidirectional LSTM only processes words from left to right, meaning a word only has access to past context. A Bidirectional LSTM runs two separate hidden layers (one forward, one backward) and concatenates their outputs. In customer reviews, sentiment cues can appear before or after key adjectives (e.g., *"Not only was it defective, it broke"* vs *"It broke even though it was brand new"*). BiLSTM captures contextual cues from both directions.

        **Q2: How does LSTM solve the vanishing gradient problem in traditional RNNs?**
        - *Answer:* Traditional RNNs suffer from vanishing/exploding gradients during backpropagation through time (BPTT) due to continuous matrix multiplications. LSTMs introduce an internal **Cell State** ($C_t$) that acts as an information highway, regulated by three gates:
          1. **Forget Gate ($f_t$):** Decides what information to discard.
          2. **Input Gate ($i_t$):** Decides which new information to store.
          3. **Output Gate ($o_t$):** Determines what part of the cell state forms the output.
          Because information can flow across time steps through additive updates rather than purely multiplicative ones, gradients remain stable over long sequences.

        **Q3: What role does the Embedding Layer play?**
        - *Answer:* Instead of using high-dimensional sparse representations like One-Hot Encoding (which has dimension equal to vocabulary size and zero semantic correlation between vectors), the Embedding Layer maps words into dense, continuous vectors (e.g., 64 dimensions). Words with similar semantic meanings and usage patterns develop proximate geometric locations in vector space during training.

        **Q4: Why is SpatialDropout1D used instead of regular Dropout after the Embedding layer?**
        - *Answer:* Regular Dropout independently drops individual elements across the embedding vector. SpatialDropout1D drops entire 1D word feature channels. In NLP embeddings, adjacent values in a single word embedding are strongly correlated; standard dropout simply shifts the weight burden to adjacent elements. SpatialDropout1D prevents co-adaptation across the whole embedding map.

        **Q5: What is the purpose of padding and truncation?**
        - *Answer:* Neural networks require uniform batch tensor dimensions. Customer reviews vary in length from 5 words to hundreds of words. By establishing a fixed maximum length (`max_len = 100`), shorter reviews are padded with zeros (`0`), while excessively long reviews are truncated.
        """)
        
    with st.expander("🛠️ 3. Technology Stack & Dataset Details"):
        st.markdown("""
        - **Programming Language:** Python 3.13
        - **Deep Learning Framework:** TensorFlow 2.21 / Keras 3
        - **NLP Tokenization:** Keras Tokenizer (Sequences + Padding)
        - **Web Application:** Streamlit 1.51
        - **Visualizations:** Plotly Express & Seaborn
        - **Dataset:** Amazon Reviews Polarity (Stanford / Hugging Face)
        """)
