from pathlib import Path
import pickle


# Get the project directory
BASE_DIR = Path(__file__).resolve().parent


# Model file paths
MODEL_PATH = BASE_DIR / "sentiment_model_randomforest.pkl"
VECTORIZER_PATH = BASE_DIR / "sentiment_randomforest_TFIDF.pkl"


def load_model():
    """Load the trained model and TF-IDF vectorizer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run train_model.py first."
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}. "
            "Run train_model.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


# Load model and vectorizer
model, tfidf_vectorizer = load_model()


def predict_sentiment(review):
    """Predict whether a movie review is positive or negative."""

    review_vector = tfidf_vectorizer.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        return "positive review"

    return "negative review"
print('done')