import pytest
import pandas as pd
from src.analysis.ranking import rank_features_by_correlation

def test_rank_features_basic():
    df = pd.DataFrame({
        'feature': ['A', 'B', 'C'],
        'importance_score': [0.8, 0.6, 0.9]
    })
    
    result = rank_features_by_correlation(df)
    
    assert list(result.columns) == ['feature', 'importance_score', 'rank']
    assert result.iloc[0]['feature'] == 'C'  # Highest importance
    assert result.iloc[0]['rank'] == 1.0
    assert result.iloc[-1]['feature'] == 'B'  # Lowest importance
    assert result.iloc[-1]['rank'] == 3.0

def test_rank_features_with_ties():
    df = pd.DataFrame({
        'feature': ['A', 'B', 'C', 'D'],
        'importance_score': [0.8, 0.8, 0.6, 0.6]
    })
    
    result = rank_features_by_correlation(df)
    
    # A and B should have rank 1, C and D should have rank 3
    assert result[result['feature'].isin(['A', 'B'])]['rank'].unique() == [1.0]
    assert result[result['feature'].isin(['C', 'D'])]['rank'].unique() == [3.0]

def test_rank_features_invalid_input():
    df = pd.DataFrame({
        'wrong_col': ['A', 'B'],
        'importance_score': [0.8, 0.6]
    })
    
    with pytest.raises(ValueError, match="Input DataFrame must have 'feature' and 'importance_score' columns"):
        rank_features_by_correlation(df)

def test_rank_features_empty():
    df = pd.DataFrame(columns=['feature', 'importance_score'])
    result = rank_features_by_correlation(df)
    assert len(result) == 0
    assert list(result.columns) == ['feature', 'importance_score', 'rank'] 