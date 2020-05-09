import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine
# import nltk to do nlp pipeline
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
#import sklearn toolkit to do ml pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

database_file = argv[0]
modelname = argv[1]

# load data from sql database
engine = create_engine('sqlite:///'+database_file)
df = pd.read_sql('disasterresponse', con=engine)
X = df.iloc[:, 1].tolist()
y = df.iloc[:, 4:]

# write a tokenization function to process text data
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", text.lower)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# build a machine learning pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

# train pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
pipeline.fit(X_train, y_train)

# test model
y_pred = pipeline.predict(X_test)
for i, col in enumerate(y_test.columns):
    labels = [0, 1]
    target_names = [col+'_0', col+'_1']
    print(classification_report(y_test.iloc[:,i], y_pred[:,i], labels=labels, target_names=target_names))

# improve model
parameters = {
    'vect__ngram_range': ((1,1), (1,2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__min_samples_split': [2, 3, 4]
}

cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)
y_pred=cv.predict(X_test)

# save model as pickle file
model_path = "./"+modelname
pickle.dump(cv, open(model_path, 'wb'))
