# 🛡️ Cyber Shield — AI Cyberbullying & Toxic Content Detection

> **Real-Time NLP & Machine Learning Platform with Browser Extension & Web Dashboard**  
> An intelligent cyberbullying detection ecosystem designed to detect, classify, and mitigate toxic content across social platforms in real-time using an ensemble of NLP preprocessors, TF-IDF vectorizers, and optimized machine learning models (SVM, Random Forest, XGBoost).

[![Python](https://img.shields.io/badge/Language-Python_3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![Flask](https://img.shields.io/badge/API-Flask_RESTful-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn_&_XGBoost-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Chrome Extension](https://img.shields.io/badge/Extension-Manifest_V3-4285F4?style=flat&logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Overview & Small Description

**Cyber Shield** is a production-oriented machine learning safety solution built to combat online harassment, hate speech, and cyberbullying across digital ecosystems.

The platform provides a dual-interface architecture:
1. **Interactive Web Dashboard**: An enterprise dark-themed web application allowing users and moderators to submit single comments or bulk text batches for instant multi-class toxic sentiment classification and confidence scoring.
2. **Chrome Browser Extension (Manifest V3)**: A lightweight, non-intrusive client-side background listener that dynamically monitors DOM nodes on Twitter/X, Facebook, YouTube, Reddit, and Instagram, automatically highlighting or blurring abusive content in real time.

---

## ✨ Features

- **High-Performance ML Ensemble**: Evaluates and compares three distinct classifiers — **Linear Support Vector Machine (LinearSVC)**, **Random Forest**, and **XGBoost** — to select optimal latency-accuracy trade-offs.
- **Context-Aware NLP Preprocessing Pipeline**: Cleans noisy social media text by decoding leetspeak, slang normalization, emoji sentiment translation, contraction expansion, and URL scrubbing.
- **Multi-Level Severity Classification**: Classifies statements into `Clean`, `Low Severity`, `Medium Severity`, and `High Severity / Hate Speech` with granular probability thresholds.
- **High-Throughput REST API**: Powered by Flask with CORS support, serving predictions under $15\text{ms}$ per request with thread-safe model caching.
- **Manifest V3 Chrome Extension**: Real-time DOM observer targeting major social platforms with customizable blur/warning overlay toggles and local badge counters.
- **Explainable Metrics & Visualizations**: Pre-computed confusion matrices, ROC-AUC curves, precision-recall graphs, and performance benchmarks stored in `reports/`.

---

## 📂 File Structure

```text
Cyber-Shield/
├── .gitignore                    # Excludes caches, virtual environments & raw artifacts
├── LICENSE                       # MIT Open Source License
├── README.md                     # Project documentation & operational manual
├── requirements.txt              # Pinned Python package dependencies
├── test_fix.py                   # Automated API and inference pipeline verification
├── train_pipeline.py             # End-to-end model training, tuning & evaluation script
├── api/                          # Backend RESTful inference service
│   ├── __init__.py
│   └── server.py                 # Flask server, prediction endpoints & static routing
├── datasets/                     # Benchmark datasets & unified training corpora
│   ├── combined_hate_speech_dataset.csv
│   ├── cyberbullying_tweets.csv
│   ├── spam.csv
│   └── unified_dataset.csv
├── extension/                    # Manifest V3 Chrome Extension
│   ├── background.js             # Service worker handling API dispatch
│   ├── content.css               # Content blur and alert badges styling
│   ├── content.js                # DOM observer for Twitter, YouTube, Reddit & Instagram
│   ├── manifest.json             # Chrome extension manifest configuration
│   ├── popup.html                # Extension toolbar popup interface
│   ├── popup.js                  # User preferences & threshold controllers
│   └── icons/                    # Multi-resolution extension icons (16px, 48px, 128px)
├── models/                       # Serialized joblib models & vectorizers
│   ├── best_model_info.joblib    # Metadata of champion model
│   ├── svm_linearsvc.joblib      # Production LinearSVC classifier
│   ├── random_forest.joblib      # Random Forest ensemble
│   ├── tfidf_vectorizer.joblib   # Fitted n-gram TF-IDF vectorizer
│   └── xgboost.joblib            # Gradient boosted decision trees model
├── reports/                      # Evaluation figures & training summaries
│   └── confusion_matrices.png    # Comparative confusion matrix analysis
├── src/                          # Modular core NLP & ML engineering package
│   ├── data_loader.py            # Dataset ingestion, schema unification & cleaning
│   ├── evaluation.py             # Accuracy, precision, recall & F1 evaluation metrics
│   ├── feature_engineering.py   # Tokenization, stop-words & n-gram TF-IDF pipeline
│   ├── models.py                 # Model training definitions & hyperparameter grids
│   ├── preprocessing.py          # Regex normalization, slang mapping & lemmatization
│   └── utils.py                  # Logger setup & serialization helpers
└── website/                      # Standalone client application
    ├── app.js                    # Dynamic AJAX submission & score visualizer
    ├── index.html                # Dark-mode dashboard UI
    └── style.css                 # Glassmorphic modern layout & components
```

---

## 🚀 Setup & Deployment

### Prerequisites
- **Python 3.10+**
- **pip** and `venv`
- Google Chrome or Chromium-based browser (for extension)

### 1. Backend Server Setup
```bash
# Clone the repository
git clone https://github.com/AnshulKanodia/Cyber-Shield.git
cd Cyber-Shield

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask REST API & Web UI
python api/server.py
```
The web dashboard is now accessible at `http://localhost:5000`.

### 2. Loading the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle on **Developer mode** in the top right corner.
3. Click **Load unpacked**.
4. Select the `extension/` folder inside this repository.
5. Cyber Shield is active and will analyze social media text in your browser.

### 3. Model Re-Training (Optional)
To retrain the models with updated datasets:
```bash
python train_pipeline.py
```

---

## 🛠️ Tech Stack & Language Breakdown

| Component | Technology |
|---|---|
| **Programming Language** | Python 3.10+, JavaScript (ES6+), HTML5, CSS3 |
| **Backend Framework** | Flask, Flask-CORS |
| **Machine Learning** | Scikit-Learn, XGBoost, Joblib |
| **NLP Pipeline** | NLTK, Regular Expressions, TF-IDF Vectorization |
| **Browser Extension** | Chrome Manifest V3, MutationObserver API |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the [LICENSE](LICENSE) file for details.
