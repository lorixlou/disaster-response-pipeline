import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

message_file = argv[0]
categories_file = argv[1]
database_file = argv[2]

# load disaster meassages and categories csv files
messages = pd.read_csv(message_file)
categories = pd.read_csv(categories_file)

# merge the messages and categories dataframes on the key = id
df = pd.merge(messages, categories, on='id')

# split the values in the categories column on the ";" symbol
categories = df['categoreis'].str.split(';', expand=True)

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

# save the clean dataset into an sqlite dataset
engine = create_engine('sqlite:///'+database_file)
df.to_sql('disasterresponse', engine, index=False)
