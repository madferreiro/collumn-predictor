import pytest
import pandas as pd
import numpy as np
from src.preprocessing.feature_types import detect_feature_types

def test_detect_feature_types_numerical():
    # Create DataFrame with numerical columns
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000],
        'experience': [5, 10, 15]
    })
    
    names, types = detect_feature_types(df)
    
    assert len(names) == 3
    assert len(types) == 3
    assert all(t == 'numerical' for t in types)

def test_detect_feature_types_categorical():
    # Create DataFrame with categorical columns
    df = pd.DataFrame({
        'gender': ['M', 'F', 'M'],
        'education': ['BS', 'MS', 'BS'],
        'department': ['IT', 'HR', 'IT']
    })
    
    names, types = detect_feature_types(df)
    
    assert len(names) == 3
    assert len(types) == 3
    assert all(t == 'categorical' for t in types)

def test_detect_feature_types_mixed():
    # Create DataFrame with both numerical and categorical columns
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'gender': ['M', 'F', 'M'],
        'salary': [50000, 60000, 70000],
        'education': ['BS', 'MS', 'BS']
    })
    
    names, types = detect_feature_types(df)
    
    assert len(names) == 4
    assert len(types) == 4
    assert types[0] == 'numerical'  # age
    assert types[1] == 'categorical'  # gender
    assert types[2] == 'numerical'  # salary
    assert types[3] == 'categorical'  # education

def test_detect_feature_types_datetime():
    # Create DataFrame with datetime column
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3),
        'age': [25, 30, 35]
    })
    
    names, types = detect_feature_types(df)
    
    assert len(names) == 2
    assert len(types) == 2
    assert types[0] == 'numerical'  # datetime defaults to numerical
    assert types[1] == 'numerical'  # age 