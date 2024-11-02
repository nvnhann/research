import pandas as pd
import argparse
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_file):
    """
    Load data from a SQLite database and separate into input (X) and output (Y) variables.

    Args:
    database_filepath (str): Path to SQLite database file.

    Returns:
    X (pandas.core.series.Series): Input data (messages).
    Y (pandas.core.frame.DataFrame): Output data (categories).
    """
    # Load data from database 
    engine = create_engine(f'sqlite:///{database_file}')
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y

def tokenize(text):
    """
    Tokenizes input text by converting it to lowercase, removing stopwords,
    and lemmatizing each token.
    
    Args:
    - text (str): the input text to be tokenized
    
    Returns:
    - tokens (list of str): the list of tokens after text processing
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def build_model():
    """
    Builds a machine learning pipeline using scikit-learn's Pipeline class and tunes the hyperparameters of the
    model using GridSearchCV.

    Returns:
    -------
    A GridSearchCV object that can be used to fit the model and make predictions.
    """
    # Define the pipeline
    vectorizer = CountVectorizer(tokenizer=tokenize)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10)))
    ])

    # Define the parameter grid to search over
    param_grid = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [100]
    }

    # Define the GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=2)

    return grid_search

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the performance of a machine learning model on a test dataset using classification_report.

    Parameters:
    ----------
    model : object
        The fitted machine learning model to be evaluated.
    X_test : array-like of shape (n_samples, n_features)
        The feature matrix of the test dataset.
    Y_test : array-like of shape (n_samples, n_outputs)
        The target matrix of the test dataset.

    Returns:
    -------
    None
        Prints the classification report for each output column.
    """

    # Make predictions on the test dataset
    y_pred = model.predict(X_test)

    # Print the classification report for each output column
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

def save_model(model, model_filepath):
    """
    Exports the final machine learning model as a pickle file.

    Parameters:
    ----------
    model : object
        The fitted machine learning model to be saved.
    model_filepath : str
        The file path to save the model as a pickle file.

    Returns:
    -------
    None
        Saves the model as a pickle file.
    """

    # Save the model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train a machine learning model to classify disaster response messages.')

    # Add the command line arguments
    parser.add_argument('database_filepath', type=str, help='Filepath of the SQLite database containing cleaned message and category data.')
    parser.add_argument('model_filepath', type=str, help='Filepath to save the trained model as a pickle file.')

    # Parse the arguments
    args = parser.parse_args()

    # Extract the arguments
    database_filepath = args.database_filepath
    model_filepath = args.model_filepath

    # Call the functions to train and evaluate the model
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
    print('Building model...')
    model = build_model()
    model.fit(X_train, Y_train)
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test)
    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    # Save the trained model
    save_model(model, model_filepath)
    print('Trained model saved to {}!'.format(model_filepath))

if __name__ == '__main__':
    main()
