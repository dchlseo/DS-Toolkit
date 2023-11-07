import pandas as pd
import numpy as np

def impute_null(df, column_to_impute, reference_columns=None, strategy='median', scope='overall'):
    """
    Imputes null values in a DataFrame column based on one or more reference columns 
    or the overall dataset using a specified strategy.
    
    Parameters:
    - df: DataFrame containing the data.
    - column_to_impute: The name of the column in which null values are to be imputed.
    - reference_columns: The name or list of names of the columns to group by for calculating 
                         the imputation value. This is used when the scope is 'group'.
                         Default is None.
    - strategy: The strategy to use for imputation ('median', 'mean', 'mode', or a function).
                Default is 'median'.
    - scope: The scope to use for calculating imputation value ('group' or 'overall').
             Default is 'overall'.
    
    Returns:
    - df_imputed: DataFrame with null values imputed in the specified column.
    """
    
    # Make a copy of the DataFrame to avoid modifying the original one
    df_imputed = df.copy()
    
    # Helper function to apply the imputation strategy
    def apply_strategy(data, func):
        if func == 'median':
            return data.median()
        elif func == 'mean':
            return data.mean()
        elif func == 'mode':
            return data.mode()[0]  # mode returns a Series, take the first mode
        else:
            # Assume the function is a user-defined function that can be called directly
            return func(data)
    
    # Calculate the imputation value for each group or overall
    if scope == 'group' and reference_columns is not None:
        impute_values = df_imputed.groupby(reference_columns)[column_to_impute].transform(lambda x: apply_strategy(x, strategy))
    elif scope == 'overall':
        impute_value = apply_strategy(df_imputed[column_to_impute].dropna(), strategy)
    else:
        raise ValueError("Invalid scope or reference_columns for imputation.")
    
    # Apply the imputation
    if scope == 'group' and reference_columns is not None:
        df_imputed[column_to_impute] = df_imputed[column_to_impute].fillna(impute_values)
    elif scope == 'overall':
        df_imputed[column_to_impute] = df_imputed[column_to_impute].fillna(impute_value)
    
    return df_imputed
