import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


##load the dataset 
df=pd.read_csv(r'D:\spyderProject\deep learning\sentimentanalysis_using_randomforest\IMDB_dataset.txt')

#features and labels
X=df['review']
y=df['sentiment']

# convert positive and negative into 0 and 1
y=y.map({'positive':1,'negative':0})
vectorizer=TfidfVectorizer(stop_words='english',max_features=5000)
X=vectorizer.fit_transform(X)

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

## random forest classifier 
model=RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test,prediction)

print("Accuracy:",accuracy)

# save the train model

with open ('sentiment_model_randomforest.pkl','wb') as file:
    pickle.dump(model,file)


## save vetorizer
with open('sentiment_randomforest_TFIDF.pkl', 'wb') as file :
    pickle.dump(vectorizer,file)

print("Model Saved")