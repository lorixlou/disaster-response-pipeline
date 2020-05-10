import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine
# import nltk to do nlp pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#import sklearn toolkit to do ml pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    # load data from sql database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('disasterresponse', con=engine)
    X = df.iloc[:, 1].tolist()
    y = df.iloc[:, 4:]
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    # write a tokenization function to process text data
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    # build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # improve model
    parameters = {
        'vect__ngram_range': ((1,1), (1,2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # test model
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        labels = [0, 1]
        target_names = [col+'_0', col+'_1']
        print(classification_report(y_test.iloc[:,i], y_pred[:,i], labels=labels, target_names=target_names))

    print("Best Parameters for GridSearchCV", cv.best_params_)

    return None

def save_model(model, model_filepath):
    # save model as pickle file
    model_path = "./"+modelname
    pickle.dump(model, open(model_path, 'wb'))

    return None

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
