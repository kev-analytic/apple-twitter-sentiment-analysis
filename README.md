# 🍏 Apple Twitter Sentiment Analysis

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify tweets mentioning **Apple Inc.** into **positive**, **neutral**, or **negative** sentiment categories. It follows the **CRISP-DM** methodology to guide the entire process — from understanding the problem to model deployment and evaluation.

---

## CRISP-DM Methodology Overview

### 1. Business Understanding

Customer sentiment on social media is a powerful indicator of brand reputation and public perception. This project aims to automate the analysis of such sentiments to:
- Identify patterns in public opinion.
- Monitor reactions to Apple’s announcements and products.
- Provide actionable intelligence for decision-makers in PR, marketing, and strategy.

---

### 2. Data Understanding

**Dataset**: Apple Twitter Sentiment DFE (CrowdFlower)

**Key Columns**:
- `text`: Raw tweet text.
- `sentiment`: Target variable (positive, negative, neutral).
- `sentiment:confidence`: Confidence in label.

**Initial Observations**:
- Dataset has 3,886 tweets.
- Noisy or irrelevant metadata removed.
- Malformed or low-confidence sentiment entries were cleaned.
- Texts include social media artifacts (e.g., hashtags, URLs).

---

### 3. Data Preparation

**Steps Taken**:
- Filtered dataset to retain only relevant rows and columns.
- Renamed and casted columns for consistency.
- Removed nulls and duplicates.
- Cleaned text: removed punctuation, links, emojis, and stopwords.
- Applied tokenization and lemmatization using NLTK.
- Engineered new features:
  - Tweet length
  - Word/sentence counts
  - Lexical diversity

**Augmentations**:
- SMOTE applied to handle class imbalance.
- TF-IDF vectorization for model input.
- did text augmentation for the minority class

---

### 4. Exploratory Data Analysis (EDA)

**Analyses Conducted**:
- Sentiment distribution.
- Box plots of tweet length, word count, and sentence count across sentiment classes.
- Frequency distributions of top words per sentiment class.
- Word clouds visualizing key terms for each sentiment.

These insights informed the preprocessing strategy and revealed imbalances and linguistic traits.

---

### 5. Modeling

**Preprocessing Pipelines** were built for different data conditions:
- Raw TF-IDF features
- SMOTE-balanced data
- Augmented feature set (lengths, diversity, etc.)

**Models Trained**:
- Logistic Regression
- Random Forest
- XGBoost

**Validation**:
- Accuracy, Precision, Recall, F1-score
- Cross-validation scores
- ROC AUC and confusion matrices

**Best Model**: Logistic Regression (with augmented features and hyperparameter tuning)

---

### 6. Hyperparameter Tuning

Used `GridSearchCV` to tune top models on:
- Logistic Regression (C, penalty)
- XGBoost (max_depth, n_estimators)

The best-performing model was Logistic Regression with tuned hyperparameters and engineered features.

---

### 7. Model Evaluation

**Final Model**: Tuned Logistic Regression

**Evaluation Metrics**:
- Classification Report
- Confusion Matrix
- ROC Curve
- Feature Importance

**Insights**:
- High precision and recall on neutral and positive classes.
- Strong interpretability due to linear model.
- TF-IDF weights provided useful feature importance analysis.

---

### 8. Deployment

Model was saved using `joblib`, allowing reuse in production pipelines or real-time applications.

Future deployment considerations:
- Integrate with Twitter API for live tracking.
- Build a dashboard for real-time sentiment analytics.

---

### 9. Recommendations

1. **Deploy** the tuned logistic regression model.
2. **Continuously retrain** the model with new Twitter data to adapt to language changes.
3. **Experiment with deep learning models** for potential gains (e.g., BERT).
4. Use the system in **product feedback analysis**, **brand monitoring**, or **market research** pipelines.

---

### 10. Limitations & Assumptions

- **TF-IDF assumptions** may miss contextual meaning.
- **Class imbalance** required external handling (SMOTE).
- **Noisy social media data** reduces precision.
- **Sentiment subjectivity** introduces label noise.
- **Data stationarity** is assumed, though public opinion shifts rapidly.
- **Model simplicity** trades off accuracy for interpretability.

---

## Requirements

To run this notebook, install the required Python packages:

```bash
pip install -r requirements.txt
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
xgboost
joblib
