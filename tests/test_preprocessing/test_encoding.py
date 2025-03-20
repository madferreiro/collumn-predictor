import pytest
import pandas as pd
from src.preprocessing.encoding import apply_one_hot_encoding

def test_apply_one_hot_encoding_basic():
    # Create DataFrame with categorical columns
    df = pd.DataFrame({
        'gender': ['M', 'F', 'M'],
        'education': ['BS', 'MS', 'BS']
    })
    
    df_encoded, categorical_cols = apply_one_hot_encoding(df, max_categories=10)
    
    # Check original categorical columns are stored
    assert set(categorical_cols) == {'gender', 'education'}
    
    # Check one-hot encoded columns are created
    expected_columns = {
        'gender_M', 'gender_F',
        'education_BS', 'education_MS'
    }
    assert set(df_encoded.columns) == expected_columns

def test_apply_one_hot_encoding_with_limit():
    # Create DataFrame with many categories
    df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    })
    
    df_encoded, _ = apply_one_hot_encoding(df, max_categories=5)
    
    # Check that only top 5 categories are encoded
    assert len([col for col in df_encoded.columns if col.startswith('category_')]) == 6  # 5 categories + 'other'
    
    # Check that 'other' category exists
    assert 'category_other' in df_encoded.columns

def test_apply_one_hot_encoding_mixed_data():
    # Create DataFrame with both numerical and categorical columns
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'gender': ['M', 'F', 'M'],
        'education': ['BS', 'MS', 'BS']
    })
    
    df_encoded, categorical_cols = apply_one_hot_encoding(df, max_categories=10)
    
    # Check numerical column remains unchanged
    assert 'age' in df_encoded.columns
    
    # Check categorical columns are encoded
    assert 'gender_M' in df_encoded.columns
    assert 'gender_F' in df_encoded.columns
    assert 'education_BS' in df_encoded.columns
    assert 'education_MS' in df_encoded.columns 