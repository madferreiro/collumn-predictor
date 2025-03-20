import pytest
import pandas as pd
import numpy as np
from src.preprocessing.missing_values import handle_missing_values

def test_handle_missing_values_no_missing():
    # Create DataFrame with no missing values
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    
    df_cleaned, removed = handle_missing_values(df, threshold=0.5)
    
    assert len(removed) == 0
    assert df_cleaned.equals(df)

def test_handle_missing_values_above_threshold():
    # Create DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan],
        'B': [4, np.nan, np.nan],  # 66.7% missing
        'C': [7, 8, 9]
    })
    
    df_cleaned, removed = handle_missing_values(df, threshold=0.5)
    
    assert 'B' in removed
    assert len(removed) == 1
    assert 'B' not in df_cleaned.columns
    assert 'A' in df_cleaned.columns
    assert 'C' in df_cleaned.columns

def test_handle_missing_values_all_columns():
    # Create DataFrame where all columns have too many missing values
    df = pd.DataFrame({
        'A': [1, np.nan, np.nan],
        'B': [np.nan, np.nan, np.nan],
        'C': [np.nan, np.nan, np.nan]
    })
    
    df_cleaned, removed = handle_missing_values(df, threshold=0.5)
    
    assert len(removed) == 3
    assert len(df_cleaned.columns) == 0 