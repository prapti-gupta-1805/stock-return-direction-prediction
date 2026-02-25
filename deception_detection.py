# ============================================================
# Linguistic Fingerprinting for Deception Detection
# Final Comparative + Explainable + BERT Benchmark Pipeline
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import string
import nltk
import shap
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Download tokenizer resources (NLTK 3.8+ fix)
nltk.download('punkt')
nltk.download('punkt_tab')

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_csv("data.csv")

df = df.rename(columns={'text_': 'text'})
df = df[['text', 'label']].dropna()

# Map labels: CG = fake (1), OR = genuine (0)
df['label'] = df['label'].map({'CG': 1, 'OR': 0})
df = df.dropna(subset=['label'])

df['label'] = df['label'].astype(int)
df['text'] = df['text'].astype(str)

print("Dataset Loaded:", df.shape)
print(df['label'].value_counts())

# ============================================================
# 2️⃣ LINGUISTIC FEATURE ENGINEERING
# ============================================================

class LinguisticFeatures(BaseEstimator, TransformerMixin):

    def get_features(self, text):
        words = nltk.word_tokenize(text)
        sentences = re.split(r'[.!?]+', text)

        num_words = len(words)
        num_sentences = len([s for s in sentences if s.strip()])
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        lexical_diversity = len(set(words)) / len(words) if words else 0
        punct_count = sum(c in string.punctuation for c in text)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return [num_words, num_sentences, avg_word_len,
                lexical_diversity, punct_count, upper_ratio]

    def fit(self, X, y=None): return self

    def transform(self, X):
        return np.array([self.get_features(t) for t in X])

# ============================================================
# 3️⃣ TF-IDF + LINGUISTIC HYBRID FEATURES
# ============================================================

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_text = tfidf.fit_transform(df['text'])

ling = LinguisticFeatures().transform(df['text'])
ling_scaled = StandardScaler().fit_transform(ling)

X = hstack([X_text, ling_scaled])
y = df['label']

# ============================================================
# 4️⃣ TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 5️⃣ CLASSICAL MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss')
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# ============================================================
# 6️⃣ CROSS-VALIDATION
# ============================================================

print("\nCross-Validation:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f}")

# ============================================================
# 7️⃣ SHAP EXPLAINABILITY (Dense Sample Fix)
# ============================================================

print("\nRunning SHAP Explainability...")

sample_size = min(300, X_train.shape[0])
X_dense = X_test[:sample_size].toarray()

explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = explainer.shap_values(X_dense)

shap.summary_plot(shap_values, X_dense)

# ============================================================
# 8️⃣ BERT BASELINE
# ============================================================

print("\nTraining BERT Baseline...")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

train_dataset = Dataset.from_dict({"text": list(train_texts), "labels": list(train_labels)})
test_dataset  = Dataset.from_dict({"text": list(test_texts),  "labels": list(test_labels)})

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset  = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./bert_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="no",
    logging_steps=50,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

print("\nBERT Accuracy:", accuracy_score(test_labels, y_pred))
print(classification_report(test_labels, y_pred))