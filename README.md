# Sentiment Analysis Using Random Forest and Flask

## Project Overview

This project is a web-based Sentiment Analysis application that predicts whether a movie review is **Positive** or **Negative**.

The machine learning model is trained using a **Random Forest Classifier** and **TF-IDF Vectorization**. The trained model is integrated with a **Flask web application**.

## download IMDB dataset from kaggel 
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Features

* Predicts Positive or Negative sentiment
* Uses TF-IDF for text vectorization
* Uses Random Forest for classification
* Simple Flask web interface
* Accepts movie reviews as user input
* Displays prediction results on the webpage

## Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML
* CSS
* Git and GitHub

## Project Structure

```text
sentimentanalysis_using_randomforest/
│
├── app.py
├── predict.py
├── train_model.py
├── train_model.ipynb
├── sentiment_randomforest_TFIDF.pkl
├── IMDB_dataset.txt
├── README.md
├── requirements.txt
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/akhlad97/sentiment_analysis_flask.git
```

### 2. Navigate to the project folder

```bash
cd sentiment_analysis_flask
```

### 3. Install the required libraries

```bash
pip install flask scikit-learn pandas numpy
```

## Train the Model

Run the following command:

```bash
python train_model.py
```

This will train the Random Forest model and generate the required model files.

## Run the Flask Application

Run:

```bash
python app.py
```

If `python` is not recognized on Windows, try:

```bash
py app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000/
```

## Example Reviews

### Positive Review

> This movie was absolutely amazing. The story was engaging and the acting was excellent.

### Negative Review

> This movie was very disappointing. The story was boring and the acting was poor.

## How It Works

1. The user enters a movie review.
2. The review is sent to the Flask application.
3. The TF-IDF vectorizer converts the text into numerical features.
4. The Random Forest model predicts the sentiment.
5. The result is displayed as Positive or Negative.

## Future Improvements

* Improve model accuracy
* Add more sentiment categories
* Deploy the application online
* Improve the user interface
* Add Docker support

## Author

**Moh Akhlad**
