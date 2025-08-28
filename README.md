![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# Sentiment Analysis in Low-Resource Languages
> MSc Dissertation Project (2024) – Birmingham City University  
> Publicly released on GitHub in 2025

This project explores **sentiment analysis** for **low-resource languages** (focus: Hindi → English).  
It compares **translation pipelines** (Google Translate vs HuggingFace MarianMT) and evaluates multiple **ML/DL classifiers**.

---

## 👨‍🎓 Project Info
- **Dissertation Title:** *Exploring Sentiment Analysis in Low-Resource Language: Unveiling Limitations in Translation Libraries*  

---

## 🚀 Features
- Preprocessing for Hindi and English datasets  
- Translation pipelines:  
  - **Google Translate API**  
  - **HuggingFace MarianMT Transformers**  
- Data Augmentation (GPT-Neo for balancing classes)  
- Data Balancing utilities  
- Model Training:  
  - Naïve Bayes  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - LSTM / BiLSTM  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score  
- Visual comparisons (confusion matrices, accuracy plots)  

---

## 📊 Dataset
All datasets (~10 MB total) are included in `/data`.
⚠️ Note: These datasets are provided for academic/research purposes only. 
Scripts are kept in their original dissertation form (2024)

### Structure
- **English (original):** `covid.csv` (~6k tweets, Kaggle)  
- **Hindi (original):** `hi-train.csv`, `hi-test.csv`, `hi-valid.csv`, `hindi-data-combined.csv` (~5k reviews, IIT Patna via Kaggle)  
- **Translated:**  
  - `google-data.csv` (Google Translate)  
  - `transformers-data.csv` (HuggingFace MarianMT)  
- **Augmented:** `covidaug.csv` (synthetic balancing with GPT-Neo)  
- **Balanced:**  
  - `eng-data-bal.csv`  
  - `google-data-bal.csv`  
  - `transformers-data-bal.csv`  

---

## 📂 Repository Structure
```
sentiment-analysis-on-low-resource-languages/
├─ src/
│  ├─ HindiDataPreparatation.py
│  ├─ GoogleTranslation.py
│  ├─ TransformersTranslation.py
│  ├─ DataAugmentation.py
│  ├─ DataBalance.py
│  └─ SentimentAnalysisModel.py
├─ data/
│  ├─ english/
│  ├─ hindi/
│  ├─ translated/
│  ├─ augmented/
│  └─ balanced/
├─ docs/
│  └─ ProjectReport.pdf
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/ShouvikDas01/Sentiment-Analysis-on-Low-Resource-Languages.git
cd sentiment-analysis-on-low-resource-languages

# Create environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage
### 1. Data Preparation
```bash
python src/HindiDataPreparatation.py
```

### 2. Translation
- Google Translate:
```bash
python src/GoogleTranslation.py
```
- HuggingFace Transformers:
```bash
python src/TransformersTranslation.py
```

### 3. Data Augmentation
```bash
python src/DataAugmentation.py
```

### 4. Data Balancing
```bash
python src/DataBalance.py
```

### 5. Model Training
```bash
python src/SentimentAnalysisModel.py
```

Outputs: accuracy, precision, recall, F1-score, confusion matrices.  

---

## 📈 Results (Summary)
- **English dataset** performed best across classifiers.  
- **SVM** and **Logistic Regression** gave highest F1-scores post-tuning.  
- **LSTM** worked well on English but struggled on translated datasets.  
- Translation quality had a direct impact on classification performance.  

Full results: see [`docs/ProjectReport.pdf`](./docs/ProjectReport.pdf).  

---

## 📜 Timeline
- Research & Implementation: 2023–24  
- Open-sourced: 2025  

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).  

---

## ✍️ Author
**Shouvik Das** · MSc Computer Science (Distinction, 2023–24)  
[GitHub](https://github.com/ShouvikDas01) · [LinkedIn](https://www.linkedin.com/in/shouvikdas01)

