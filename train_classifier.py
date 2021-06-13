import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    engine = create_engine('database_filepath', echo = False)
    df = pd.read_sql("SELECT * FROM Messages", engine)

def tokenize(text):
    """
    inputs:
    messages
       
    Returns:
    list of words into numbers of same meaning
    """
    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # normalization word tokens and remove stop words
    normalizer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normalized = [normalizer.stem(word) for word in tokens if word not in stop_words]
    
    return normalized


def build_model():
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators =10)))
])

    parameters = {'vect__min_df': [1],
              'tfidf__use_idf':[True],
              'clf__estimator__n_estimators':[10], 
              'clf__estimator__min_samples_split':[5]}

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    Y_test = np.array(Y_test)

    accuracy = []
    precision = []
    f1_score = []
    recall = []

    (rows, cols) = Y_test.shape
    n = len(category_names)

    Y_predict = model.predict(X_test)
    
    #iterate through the columns:
    
    for i in range(n):
        
        #get True positives (Tp), False positives (Fp), and False negative (Fn)):
        
        Tp = sum((Y_predict[:,i] ==1) & (Y_test[:,i]==1))
        Fp = sum((Y_predict[:,i] ==1) & (Y_test[:,i] ==0))
        Fn = sum((Y_test[:,i] ==1) & (Y_predict[:,i]==0))
        
        # Find the precision, accuracy, recall, and f_1 scores:
        
        accuracy_i = sum((Y_test[:,i] == Y_predict[:,i])/rows)
        
        precision_i = Tp/(Tp + Fp)
        
        recall_i = Tp/(Tp + Fn)
        
        f_i = Tp/(Tp + (1/2)*(Fp + Fn))
        
        accuracy.append(accuracy_i)
        
        precision.append(precision_i)
        
        recall.append(recall_i)
        
        f1_score.append(f_i)
        
    #store everything into 'matrix', transpose it and convert it into a matrix
    matrix = []
    matrix.append([accuracy, precision, recall, f1_score])
    matrix = np.transpose(matrix)
    matrix = np.asmatrix(np.array(matrix))
    
    #create a data frame that conveniently displays all the results    
    data_matrix = pd.DataFrame(data = matrix, index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
        
    return data_matrix

def save_model(model, model_filepath):
   
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
