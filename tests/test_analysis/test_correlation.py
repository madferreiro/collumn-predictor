import pytest
import pandas as pd
import numpy as np
from src.analysis.correlation import compute_correlation_scores

def test_compute_correlation_scores_basic():
    # Create DataFrame with perfect positive and negative correlations
    df = pd.DataFrame({
        'perfect_pos': [1, 2, 3, 4],
        'perfect_neg': [4, 3, 2, 1],
        'target': [1, 2, 3, 4]
    })
    
    result = compute_correlation_scores(df, 'target')
    
    assert len(result) == 2  # Two features
    assert set(result.columns) == {'feature', 'importance_score'}
    assert result.iloc[0]['importance_score'] == pytest.approx(1.0)  # Perfect correlation
    assert result.iloc[1]['importance_score'] == pytest.approx(1.0)  # Perfect anti-correlation

def test_compute_correlation_scores_with_categorical():
    # Create DataFrame with both numerical and categorical columns
    df = pd.DataFrame({
        'numerical': [1, 2, 3, 4],
        'categorical': ['A', 'B', 'C', 'D'],
        'target': [1, 2, 3, 4]
    })
    
    result = compute_correlation_scores(df, 'target')
    
    assert len(result) == 1  # Only numerical feature
    assert result.iloc[0]['feature'] == 'numerical'
    assert result.iloc[0]['importance_score'] == pytest.approx(1.0)

def test_compute_correlation_scores_no_correlation():
    # Create DataFrame with uncorrelated data
    df = pd.DataFrame({
        'random': [1, -1, 1, -1],
        'target': [1, 1, -1, -1]
    })
    
    result = compute_correlation_scores(df, 'target')
    
    assert len(result) == 1
    assert result.iloc[0]['importance_score'] == pytest.approx(0.0, abs=0.1)

def test_compute_correlation_scores_non_numeric_target():
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4],
        'target': ['A', 'B', 'C', 'D']
    })
    
    with pytest.raises(ValueError, match="Target column 'target' must be numeric"):
        compute_correlation_scores(df, 'target') 