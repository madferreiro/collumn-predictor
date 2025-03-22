import pytest
import pandas as pd
import numpy as np
from src.preprocessing.normalization import normalize_features

def test_normalize_features_basic():
    # Create DataFrame with numerical columns
    df = pd.DataFrame({
        'age': [20, 30, 40],
        'salary': [50000, 75000, 100000]
    })
    
    df_normalized = normalize_features(df)
    
    # Check values are normalized between 0 and 1
    assert df_normalized['age'].min() == 0
    assert df_normalized['age'].max() == 1
    assert df_normalized['salary'].min() == 0
    assert df_normalized['salary'].max() == 1

def test_normalize_features_with_constant():
    # Create DataFrame with a constant column
    df = pd.DataFrame({
        'age': [30, 30, 30],
        'salary': [50000, 75000, 100000]
    })
    
    df_normalized = normalize_features(df)
    
    # Constant column should be set to 0
    assert (df_normalized['age'] == 0).all()
    # Other column should be normalized
    assert df_normalized['salary'].min() == 0
    assert df_normalized['salary'].max() == 1

def test_normalize_features_mixed_types():
    # Create DataFrame with both numerical and categorical columns
    df = pd.DataFrame({
        'age': [20, 30, 40],
        'name': ['John', 'Jane', 'Bob'],
        'salary': [50000, 75000, 100000]
    })
    
    df_normalized = normalize_features(df)
    
    # Check numerical columns are normalized
    assert df_normalized['age'].min() == 0
    assert df_normalized['age'].max() == 1
    assert df_normalized['salary'].min() == 0
    assert df_normalized['salary'].max() == 1
    # Check categorical column remains unchanged
    assert (df_normalized['name'] == df['name']).all()

def test_normalize_features_disabled():
    # Create DataFrame with numerical columns
    df = pd.DataFrame({
        'age': [20, 30, 40],
        'salary': [50000, 75000, 100000]
    })
    
    df_normalized = normalize_features(df, normalize=False)
    
    # Check values remain unchanged
    assert (df_normalized == df).all().all() 