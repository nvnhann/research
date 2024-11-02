import json
import plotly
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/InsertDatabaseName.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')

def index():
    """
    OUTPUT:
    Renders the master html template
    """
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    
    y_counts = df.iloc[:, 4:].sum().sort_values(ascending=True)
    y_names = list(y_counts.index)
    
    # create visuals
    graphs = [
        
        {
            'data': [
                Bar(
                    x=y_counts,
                    y=y_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Count"
                },
                'margin': {
                    'l':200,
                    'b':100
                },
                'height': 1000,
            }
        },
        
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'margin': {
                    't':100,
                },
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')

def go():
    """
    OUTPUT:
    Renders the go html template, where the user goes after inputing the message
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    OUTPUT:
    a link with the web app hosted locally
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()