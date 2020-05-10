from sys import argv
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load disaster meassages and categories csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the messages and categories dataframes on the key = id
    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    # split the values in the categories column on the ";" symbol
    categories = df['categories'].str.split(';', expand=True)

    # use the first row as the column names
    row = categories.iloc[0]
    category_colnames = [(lambda x: x[:-2])(r) for r in row]
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(int)

    # drop the original cattegories column from 'df'
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new 'categories'
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(keep='first', inplace=True)

    return df

def save_data(df, database_filename):
    # save the clean dataset into an sqlite dataset
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disasterresponse', engine, index=False)

    return None

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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
