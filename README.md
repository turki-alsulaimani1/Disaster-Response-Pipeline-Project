# Disaster-Response-Pipeline-Project
This project is part of Udacity Data Scientist Nanodegree 

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

## Installations:
  - Install:
    - Anconda
    - nltk
    
          in the notebook use this command to install it:
          nltk.download()
        
    
    
  - using the following Packeges:
    - pandas
    - plotly
    - Flask
    - sqlalchemy
    - nltk
    - json
    - pickle
## Project Motivation:
   #### As Part of the Data science Nanodegree. I used the Figure Eight Data for disaster data.This project will show off software skills, including the ability to create basic data pipelines and write clean, organized code!
   ###  Three component:
   #### ETL Pipeline:
   - Merges the two datasets.
   - Cleans the data.
   - Stores it in a SQLite database.
   #### ML Pipeline:
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file
   #### Flask Web App
   
## Folder:
Workspace
Files:
data_process.py.
train_classifier.py
run.py

## acknowledgments and licinses:
I thank Udacity for great teaching materials and for Figure Eight for providing this data.
