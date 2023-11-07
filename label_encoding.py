import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def create_label_encoded_columns(df, column_names, save=False):
    """
    Create label encoded columns for a list of column names or a single column name
    using sklearn's LabelEncoder. It uses the same encoder for all the given columns.
    Optionally saves the LabelEncoder to disk.

    Parameters:
    df (pd.DataFrame): Dataframe containing the columns to encode.
    column_names (list or str): List of column names or a single column name to be label encoded.
    save (bool): If True, saves the LabelEncoder object to disk. Default is False.

    Returns:
    tuple: (LabelEncoder, pd.DataFrame) 
           The LabelEncoder object used for encoding and the dataframe with the new encoded columns added.
    """
    # Initialize the LabelEncoder
    le = LabelEncoder()
    
    # If a single column name is given, convert it to a list
    if isinstance(column_names, str):
        column_names = [column_names]
    
    # Combine all the unique values from the columns into a single list
    all_values = pd.Series(dtype=str)
    for column in column_names:
        all_values = pd.concat([all_values, df[column].fillna('missing').astype(str)])
    unique_values = all_values.unique()
    
    # Fit the label encoder on all unique values
    le.fit(unique_values)
    
    # Transform each column and add it to the dataframe
    for column in column_names:
        df[f'{column}_encoded'] = le.transform(df[column].fillna('missing').astype(str))
    
    # Save the LabelEncoder to disk if save is True
    if save:
        joblib.dump(le, f'label_encoder_{column_names}.joblib')

    return le, df
