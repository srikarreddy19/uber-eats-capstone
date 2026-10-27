from pathlib import Path
import re
import nltk
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

for folder in [OUT, REPORTS, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

reviews = pd.read_csv(RAW / "reviews.csv")
reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
reviews = reviews[reviews["review_text"].str.strip() != ""].copy()
reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
reviews = reviews.dropna(subset=["rating"])
reviews = reviews[reviews["rating"].between(1, 5)].copy()

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never"}

def clean_text(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    cleaned = []
    for word in words:
        if word not in stop_words:
            try:
                word = lemmatizer.lemmatize(word)
            except LookupError:
                pass
            cleaned.append(word)
    return " ".join(cleaned)

reviews["cleaned_review"] = reviews["review_text"].apply(clean_text)
reviews = reviews[reviews["cleaned_review"].str.strip() != ""].copy()
reviews["sentiment"] = np.where(
    reviews["rating"] >= 4, "positive",
    np.where(reviews["rating"] <= 2, "negative", "neutral")
)
reviews.to_csv(OUT / "cleaned_reviews.csv", index=False)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
X = vectorizer.fit_transform(reviews["cleaned_review"])
save_npz(OUT / "review_tfidf_matrix.npz", X)
pd.DataFrame({"term": vectorizer.get_feature_names_out()}).to_csv(
    OUT / "tfidf_terms.csv", index=False
)

y = reviews["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LinearSVC(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

joblib.dump(
    {"vectorizer": vectorizer, "model": model},
    MODELS / "review_sentiment_model.joblib"
)

labels = ["negative", "neutral", "positive"]
report = classification_report(
    y_test, predictions, labels=labels, zero_division=0
)
matrix = confusion_matrix(y_test, predictions, labels=labels)

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
    ax=ax, cmap="Blues", colorbar=False
)
plt.title("Review Sentiment Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES / "sentiment_confusion_matrix.png", dpi=150)
plt.close()

sia = SentimentIntensityAnalyzer()
reviews["vader_compound"] = reviews["review_text"].apply(
    lambda text: sia.polarity_scores(text)["compound"]
)
reviews["vader_sentiment"] = np.where(
    reviews["vader_compound"] >= 0.05, "positive",
    np.where(reviews["vader_compound"] <= -0.05, "negative", "neutral")
)
vader_accuracy = accuracy_score(reviews["sentiment"], reviews["vader_sentiment"])

def top_terms(texts, count=15):
    if len(texts) == 0:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000)
    matrix = vec.fit_transform(texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    return [terms[i] for i in scores.argsort()[::-1][:count]]

negative_terms = top_terms(reviews.loc[reviews["sentiment"] == "negative", "cleaned_review"])
positive_terms = top_terms(reviews.loc[reviews["sentiment"] == "positive", "cleaned_review"])

reviews.to_csv(OUT / "reviews_with_vader.csv", index=False)

with open(REPORTS / "sentiment_evaluation.txt", "w") as f:
    f.write("DAY 3 - CUSTOMER REVIEW SENTIMENT ANALYSIS\n\n")
    f.write("Supervised model: Linear SVM\n")
    f.write(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}\n\n")
    f.write(report)
    f.write("\n\nConfusion matrix (negative, neutral, positive):\n")
    f.write(str(matrix))
    f.write(f"\n\nVADER accuracy against rating-derived labels: {vader_accuracy:.4f}\n")
    f.write("\nTop terms in negative reviews:\n" + ", ".join(negative_terms))
    f.write("\n\nTop terms in positive reviews:\n" + ", ".join(positive_terms))
    f.write("\n\nNote: sentiment labels are derived from ratings (1-2 negative, 3 neutral, 4-5 positive).")
    f.write("\nVADER is a lexicon-based comparison, not a trained model.")

print("Day 3 completed.")
print(f"Reviews processed: {len(reviews)}")
print(f"SVM test accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"VADER accuracy: {vader_accuracy:.3f}")
print("Saved cleaned reviews, TF-IDF matrix, model, evaluation report and confusion matrix.")
