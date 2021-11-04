import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

# from sklearn.externals import joblib -> deprecated! changed to "import joblib"

import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ tokenize text """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ define plots """
    # extract data needed for visuals
    # # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],

    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    # ]
    
    # NEW: counts by category
    df2 = df[df.columns[4:]].sum()
    cat_counts = df2.tolist()
    cat_names = df2.index.str.replace("_"," ").tolist()
    
    # NEW: counts per gategory and genre
    df3 =df.groupby('genre').sum()[df.columns[4:]]
    cat_genre_counts = df3.to_numpy()
    
    # define graphs
    graphs = [
        {
            'data': [
                {
                    'type' : 'bar',
                    'x': cat_names,
                    'y': cat_counts
                }
            ],
            'layout': {
                'title': 'Counts per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                {
                    'type' : 'heatmap',
                    'z' : cat_genre_counts,
                    'x' : cat_names,
                    'y' : genre_names
                }
            ],
            'layout': {
                'title': 'Counts per Category and genre',
                'yaxis': {
                    'title': "genre"
                },
                'xaxis': {
                    'title': "Category"
                }
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
    """ show classification results """
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
    """ run flask app"""
    app.run(host='0.0.0.0', port=3001, debug=True)
    # app.run(debug=True)

if __name__ == '__main__':
    main()