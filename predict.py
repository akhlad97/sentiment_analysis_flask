import os
import joblib
from huggingface_hub import hf_hub_download


MODEL_NAME = "sentiment_model_randomforest.pkl"
TFIDF_NAME = "sentiment_randomforest_TFIDF.pkl"

REPO_ID = "akhlad/sentiment-analysis-randomforest"


def load_model():

    model_path = MODEL_NAME
    tfidf_path = TFIDF_NAME

    # If model files are not available locally,
    # download them from Hugging Face.
    if not os.path.exists(model_path):
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_NAME
        )

    if not os.path.exists(tfidf_path):
        tfidf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=TFIDF_NAME
        )

    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_path)

    return model, tfidf_vectorizer


model, tfidf_vectorizer = load_model()


def predict_sentiment(text):

    transformed_text = tfidf_vectorizer.transform([text])

    prediction = model.predict(transformed_text)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"