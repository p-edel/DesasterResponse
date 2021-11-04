# Disaster Response Pipeline Project

### Projekt Summary

This Project produces a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data. 
The model is trained on disaster data provided by Figure Eight.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    > `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
    > `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    > `python run.py`

3. Go to http://0.0.0.0:3001/

### Files and Descriptions

    .
    ├───app
    │   │   run.py                      # runs ETL and ML Pipeline         
    │   │   plots.ipynb                 # test-notebook for plots
    │   │
    │   └───templates                   
    │           go.html                 # template to display classification results
    │           master.html             # master html template
    │
    ├───data
    │       DisasterResponse.db         # DB containing messages after ETL process
    │       disaster_categories.csv     # raw category data
    │       disaster_messages.csv       # raw message data
    │       process_data.py             # ETL pipeline
    │
    └───models
            classifier.pkl              # trained model
            train_classifier.py         # ML pipeline
