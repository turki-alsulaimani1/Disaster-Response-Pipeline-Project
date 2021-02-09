"""
    Here we import all the need it libraries to run the code correctly
    
"""
import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df.message
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    return X,Y

def tokenize(text):
    """
    Input:
        1-Text.
    Output:
        return a list of words after lemmatizing each word.

    """
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Input:
        Nothing.
    Output:
        Return a pilpline with all the transformers and estimators
        
            ***************************
                        Note      
            ***************************
         I didn't use the Grid Search because it takes a long time to run 
         with either few or many parameters and have an accuracy similar to the regular way.
         
         ***If you want to run the Frid Search remove the docString and change the return values to cv.
         
        """
    pipeline = Pipeline([
    ("vect",CountVectorizer(tokenizer=tokenize)),
    ("tfidf",TfidfTransformer()),
    ("mocls",MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    """parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False)
        #'mocls__estimator__n_estimators': [50, 100, 200],
        
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)"""
    return pipeline


def evaluate_model(model, X_test, Y_test):
    Y_predict = model.predict(X_test)
    Y_predict_pd = pd.DataFrame(Y_predict, columns = Y_test.columns)

    for i in Y_test.columns:
        print(i)
        print(classification_report(Y_test[i], Y_predict_pd[i]))
    accuracy = (Y_predict == Y_test.values).mean()
    print(accuracy)
    


def save_model(model, model_filepath):
    """
    Input:
        1-Model  trained.
        2-The pathfile to save to.
        
        """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath) 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)       

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