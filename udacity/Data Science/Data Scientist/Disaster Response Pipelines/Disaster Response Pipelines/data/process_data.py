import pandas as pd
import argparse
from sqlalchemy import create_engine

def load_data(messages_file, categories_file):
    """
    Load and merge data from two csv files containing messages and categories.

    Args:
    - messages_file (str): Path to the csv file containing messages data.
    - categories_file (str): Path to the csv file containing categories data.

    Returns:
    - df (pandas.DataFrame): Merged dataframe containing messages and categories data.
    """
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)
    df = messages.merge(categories, how ='outer', on =['id'])
    return df

def clean_data(df):
    """
    Clean categories column in dataframe and split into separate columns.

    Args:
    - df (pandas.DataFrame): Dataframe containing categories column.

    Returns:
    - df (pandas.DataFrame): Cleaned dataframe with categories column split into separate columns.
    """
    # Split categories column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract category column names and set as column names for categories dataframe
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :]
    categories.columns = category_colnames
    
    # Convert category values from strings to integers
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    # Replace values of 2 in the 'related' column with 1
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    
    # Drop original categories column and merge with cleaned categories dataframe
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # drop rows or columns where all values are missing
    return df

def save_data(df, database_file):
    """
    Save dataframe to a SQLite database.

    Args:
    - df (pandas.DataFrame): Dataframe to be saved to database.
    - database_filepath (str): Path to the SQLite database.

    Returns:
    - None
    """
    # Create a connection to the database
    engine = create_engine(f'sqlite:///{database_file}')
    
    # Save dataframe to database
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(
        description='Load and clean data from CSV files and save to a SQLite database.')
    
    # Add arguments
    parser.add_argument('messages_file', type=str, help='Path to CSV file containing messages data.')
    parser.add_argument('categories_file', type=str, help='Path to CSV file containing categories data.')
    parser.add_argument('database_file', type=str, help='Path to SQLite database file to save data to.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call load_data and clean_data functions
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
           .format(args.messages_file, args.categories_file))
    df = load_data(args.messages_file, args.categories_file)
    print('Cleaning data...')
    df = clean_data(df)
    print('Saving data...\n    DATABASE: {}'.format(args.database_file))
    # Save cleaned data to database
    save_data(df, args.database_file)
    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()