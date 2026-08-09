import pickle 
with open ('sentiment_model_randomforest.pkl','rb') as file :
    model=pickle.load(file)

with open ('sentiment_randomforest_TFIDF.pkl','rb') as file:
    tfidfcount_vectorizer=pickle.load(file)

def predict_sentiment(review):
    
    review_vector = tfidfcount_vectorizer.transform([review])
    prediction=model.predict(review_vector)
    if prediction[0]==1:
        return 'postive review'
    else:
        return "negative review"