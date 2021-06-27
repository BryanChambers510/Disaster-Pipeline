import sys
import os
import json
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'wordnet'])
le = preprocessing.LabelEncoder()
import warnings
warnings.simplefilter('ignore')
import subprocess
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('termcolor')
from termcolor import colored, cprint


def load_data(database_filepath):
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: targets, message categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    
    X = df['message']
    y = df.iloc[:, 4:]
    
    category_names = df.iloc[:, 4:].columns
    return X, y, category_names

def tokenize(text):
    """
    inputs:
    messages
       
    Returns:
    list of words into numbers of same meaning
    """
    # Converting everything to lower case (Normalization step)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Remove stop words 
    stop_words = stopwords.words("english")
  
    tokens = [ t for t in tokens if t not in stopwords.words("english")]
    
    # Lemmatization (break down to simple root word such as went -> go)
    lemmed = [WordNetLemmatizer().lemmatize(t) for t in tokens]
     
    return lemmed


def build_model():
    """
    pipeline that optimizes the NLP algorithm
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1],
                'tfidf__use_idf': [True],
                'clf__estimator__n_estimators': [10],
                'clf__estimator__min_samples_split': [5]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\n Results of the test data set ({}):\n'.format(colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))

def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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