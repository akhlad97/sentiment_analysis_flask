import os

from flask import Flask, render_template, request
from predict import predict_sentiment


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review", "").strip()

    if not review:
        return render_template(
            "index.html",
            prediction="Please enter a movie review.",
            review=""
        )

    result = predict_sentiment(review)

    return render_template(
        "index.html",
        prediction=result,
        review=review
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )