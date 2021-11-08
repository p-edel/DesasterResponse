import sys

from sqlalchemy import create_engine
import pandas as pd

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk import pos_tag

nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords']) # download ressources

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """ load data from database """
    
    # create sqlite engine
    engine = create_engine(f"sqlite:///{database_filepath}")
    
    # read data from db
    df = pd.read_sql_table('Messages', f"sqlite:///{database_filepath}")
    
    # extract information
    category_names = df.columns[4:]
    X = df["message"]
    Y = df[category_names]
    
    return X, Y, category_names


def tokenize(text):
    """ normalize, tokenize and lemmatize messages """
    
    # normalize
    text = text.lower()

    # tokenize
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]

    # lemmatize
    tokens_lem = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]
    
    return tokens_lem


def build_model():
    """ build ML Model Pipeline with Count-Vectorizer, TfidfTranformer and RandomForest Classifier """
    
    # build ML-Pipeline 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=10))),
    ])
    
    # define parameter-grid
    parameters = {
        'vect__max_df': [0.5, 1.0],
        'clf__estimator__n_estimators': [5, 10],
    }
    
    # build gridsearch Model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """ make predictions on test set and evaluate metrics """
    
    #predictions on text set
    y_pred = model.predict(X_test)
    
    # evaluate metrics for each output category
    for i in range(y_pred.shape[1]):
        print(f"classification report for column: {category_names[i]}")
        print(classification_report(Y_test.values[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """ save model """
    # save model as pickle object
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ run ML-Workflow  """
    
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