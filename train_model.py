import os
import glob
import pickle
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------------------
# Download IMDB dataset from Kaggle
# -------------------------------

print("Downloading dataset...")

dataset_path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

print("Dataset downloaded to:", dataset_path)


# -------------------------------
# Find the CSV file automatically
# -------------------------------

csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")

dataset_file = csv_files[0]

print("Using dataset file:", dataset_file)


# -------------------------------
# Load dataset
# -------------------------------

df = pd.read_csv(dataset_file)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# -------------------------------
# Features and labels
# -------------------------------

X = df["review"]
y = df["sentiment"]

# Convert positive and negative into 1 and 0
y = y.map({
    "positive": 1,
    "negative": 0
})


# -------------------------------
# TF-IDF Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# Random Forest Classifier
# -------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("Training model...")

model.fit(X_train, y_train)


# -------------------------------
# Evaluate model
# -------------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)


# -------------------------------
# Save trained model
# -------------------------------

with open("sentiment_model_randomforest.pkl", "wb") as file:
    pickle.dump(model, file)


# -------------------------------
# Save TF-IDF vectorizer
# -------------------------------

with open("sentiment_randomforest_TFIDF.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("Model and vectorizer saved successfully!")