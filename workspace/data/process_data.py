"""
    Here we import all the need it libraries to run the code correctly
    
"""
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    input:
        Read the the path files for the two datasets:
        1-messages_filepath.
        2-categories_filepath.
        
        Merge the two datasets together using the merge function.
    Output:
        return the merged dataset

    """ 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id")
    return df


def clean_data(df):
    """ 
        Input:
            dataset
            
        process:
            1-Split the contents of categories column in ; .
            2-take the first row of categories and use the slice method to 
            take the first word in make it the columns names.
            3-Use the slice method to take the last number in string and leave it while removing the other contents.
            4-Change the Datatype of these columns to int.
            5-Drop any duplicate row.
        Output:
            return a cleaned wrangled dataset.
    
    """
    categories = df.categories.str.split(";",expand=True)
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = (lambda x: x.str.slice(0,-2,1))(row) 
    categories = categories.rename(columns=category_colnames)
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)
    
    # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    df.drop(labels="categories",axis = 1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    df = df[df["related"] != 2]
    return df


def save_data(df, database_filename):
    """ 
        Input: 
        1-dataset.
        2- a file path to store the database file.
        
        Process:
            Using the create_engine to create a database file
            and then save the dataset into the file.
    
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("DisasterResponse", engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df= load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()