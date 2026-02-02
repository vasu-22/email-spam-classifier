# 📧 Email Spam Classification using Machine Learning

This project builds and evaluates multiple **classification models** to detect whether an email is **Spam** or **Not Spam** using **Count Vectorizer** for text feature extraction.

---

## 🚀 Project Overview
Email spam detection is a classic **binary classification** problem in NLP.  
In this project, raw email text is converted into numerical features and passed through various machine learning models to identify the best-performing classifier.

---

## 🧠 Models Used
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Gradient Boosting

---

## 🧾 Dataset
The dataset contains:
- `text` → Email content
- `label` → spam / not spam

Labels are encoded as:
- spam → 1
- not spam → 0

---

## 🔧 Feature Engineering
We use **Count Vectorizer** to convert text into numerical features:
- Removes English stopwords
- Keeps top 5000 most frequent words
- Produces a sparse matrix of word counts

---

## 📊 Evaluation Metrics
Because spam datasets are often **imbalanced**, we evaluate models using:
- Accuracy
- Precision
- Recall
- F1 Score (**primary metric**)

---

## 🏆 Results
Linear models such as **Logistic Regression** and **SVM** perform best due to:
- High-dimensional sparse text data
- Near-linear separability of spam keywords

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
