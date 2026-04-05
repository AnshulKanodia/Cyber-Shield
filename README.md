# Cyber Shield — Cyberbullying Detection System

A real-time cyberbullying detection system powered by NLP and Machine Learning, featuring both a **standalone website** and a **Chrome browser extension** for social media monitoring.

## 🛡️ Features

- **Dual Mode** — Works as both a standalone website and Chrome extension
- **NLP Text Preprocessing Pipeline** — Cleans and normalizes social media text (slang, emojis, contractions, etc.)
- **3 ML Model Comparison** — SVM, Random Forest, and XGBoost with comprehensive evaluation
- **REST API** — Flask-based API for real-time predictions
- **Standalone Website** — Full-page premium dark-themed web app at `http://localhost:5000`
- **Chrome Extension** — Scans Twitter, Facebook, YouTube, Reddit, and Instagram in real-time
- **Severity Classification** — High, Medium, Low severity levels with confidence scores

## 📁 Project Structure

```
Cyber Shield/
├── datasets/                    # Training datasets
├── src/
│   ├── data_loader.py          # Load & merge datasets into unified format
│   ├── preprocessing.py        # NLP cleaning pipeline
│   ├── feature_engineering.py  # TF-IDF vectorization
│   ├── models.py               # SVM, Random Forest, XGBoost
│   ├── evaluation.py           # Metrics, confusion matrices, ROC curves
│   └── utils.py                # Shared helpers, slang dictionary
├── api/
│   └── server.py               # Flask REST API + website server
├── website/
│   ├── index.html              # Standalone web app
│   ├── style.css               # Premium dark theme styles
│   └── app.js                  # Client-side analysis logic
├── extension/
│   ├── manifest.json           # Chrome extension manifest v3
│   ├── popup.html/css/js       # Extension popup UI
│   ├── content.js/css          # Content script for social media scanning
│   ├── background.js           # Service worker
│   └── icons/                  # Extension icons
├── models/                     # Saved trained models (.joblib)
├── reports/                    # Generated evaluation charts
├── train_pipeline.py           # Main training script
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train_pipeline.py
```

This will:
- Load and merge all datasets into a unified format
- Preprocess text using the NLP pipeline
- Train SVM, Random Forest, and XGBoost models
- Evaluate and compare all models
- Save the best model and generate reports in `reports/`

### 3. Start the API Server

```bash
python api/server.py
```

The API will be available at `http://localhost:5000`

#### 🌐 Website Mode

Open `http://localhost:5000` in any browser — you'll see the full Cyber Shield website with:
- Hero section with animated shield
- Text analyzer with real-time analysis
- Category breakdown (Safe, Bullying, Toxic, Spam, Scam)
- Features showcase and How It Works guide
- Analysis history (stored in browser)

#### 🔌 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Predict single text |
| `/api/predict/batch` | POST | Predict multiple texts |
| `/api/model/info` | GET | Current model info |

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are such a loser, nobody likes you"}'
```

### 4. Install the Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The Cyber Shield icon will appear in your toolbar

## 📊 Datasets Used

| Dataset | Rows | Type |
|---------|------|------|
| `cyberbullying_tweets.csv` | 47,692 | Multi-class → binary |
| `combined_hate_speech_dataset.csv` | 29,550 | Binary hate labels |
| `train.csv` (Jigsaw Toxic) | 159,571 | Multi-label toxic → binary (subsampled) |

## 🧠 ML Models

| Model | Description |
|-------|-------------|
| **SVM (LinearSVC)** | Linear Support Vector Machine with balanced class weights |
| **Random Forest** | 300 trees with balanced class weights |
| **XGBoost** | Gradient boosted trees with automatic positive class weighting |

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices per model
- ROC curves overlay
- Side-by-side metric comparison chart

## 🔧 Tech Stack

- **Python 3.x** — Core language
- **scikit-learn** — SVM, Random Forest, TF-IDF, evaluation
- **XGBoost** — Gradient boosting
- **NLTK** — Tokenization, lemmatization, stopwords
- **Flask** — REST API
- **Chrome Extension (Manifest V3)** — Browser integration

## 📝 License

This project is for educational purposes — VIT Bhopal University.
