from typing import Tuple, List
import pandas as pd

def handle_missing_values(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove columns that have missing values above the specified threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (float): Maximum percentage of missing values allowed (0-1)
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with columns removed and list of removed column names
    """
    # Calculate percentage of missing values for each column
    missing_percentages = df.isnull().mean()
    
    # Get columns to remove (those above threshold)
    columns_to_remove = missing_percentages[missing_percentages > threshold].index.tolist()
    
    # Remove columns
    df_cleaned = df.drop(columns=columns_to_remove)
    
    return df_cleaned, columns_to_remove
