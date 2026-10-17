# 📰 Smart Fake News Detection System

An end-to-end Natural Language Processing (NLP) and Machine Learning system designed to identify and classify misleading or fabricated news articles using textual and stylistic patterns.

---

## 🚀 Key Features
- **High-Accuracy Classification**: Evaluated on over 44,000 real-world news articles.
- **TF-IDF Feature Extraction**: Implements unigram and bigram tokenization with stopword filtering and WordNet lemmatization.
- **Passive-Aggressive Classifier**: Online learning algorithm optimized for fast convergence and high noise tolerance on large text corpora.
- **Interactive Web Interface**: Streamlit-powered dashboard offering real-time credibility scoring and confidence metrics.

---

## 🛠️ Tech Stack
- **Language**: Python 3.12
- **Machine Learning**: Scikit-learn
- **NLP & Preprocessing**: NLTK (Stopwords, WordNet Lemmatizer), Regular Expressions
- **Web Interface**: Streamlit
- **Data Handling**: Pandas, NumPy

---

## 📊 Dataset & Performance
- **Dataset**: [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (~44,898 articles from verified Reuters feeds and unverified online sources).
- **Test Accuracy**: ~99% on balanced test evaluation.
- **F1-Score / Precision / Recall**: 0.99+ across both real and fake classes.

---

## 💻 Local Setup & Execution

### 1. Clone the repository
```bash
git clone [https://github.com/purnenduchhotaray-bot/smart-fake-news-detector.git](https://github.com/purnenduchhotaray-bot/smart-fake-news-detector.git)
cd smart-fake-news-detector
