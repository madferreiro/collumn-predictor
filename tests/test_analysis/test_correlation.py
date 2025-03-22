import pytest
import pandas as pd
import numpy as np
from src.analysis.correlation import (
    compute_correlation_scores, 
    calculate_correlation_matrix, 
    identify_highly_correlated_features,
    group_correlated_features
)

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

def test_compute_correlation_scores_empty():
    df = pd.DataFrame(columns=['feature', 'target'])
    
    result = compute_correlation_scores(df, 'target')
    
    assert len(result) == 0
    assert set(result.columns) == {'feature', 'importance_score'}

def test_compute_correlation_scores_single_feature():
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4],
        'target': [1, 2, 3, 4]
    })
    
    result = compute_correlation_scores(df, 'target')
    
    assert len(result) == 1
    assert result.iloc[0]['feature'] == 'feature'
    assert result.iloc[0]['importance_score'] == pytest.approx(1.0)

def test_calculate_correlation_matrix_basic():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [4, 3, 2, 1],  # Perfect negative correlation with A
        'C': [1, 2, 3, 4]   # Perfect positive correlation with A
    })
    
    result = calculate_correlation_matrix(df)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'A', 'B', 'C'}
    assert np.isnan(result.loc['A', 'A'])  # Diagonal should be NaN
    assert result.loc['A', 'B'] == pytest.approx(-1.0)  # Perfect negative correlation
    assert result.loc['A', 'C'] == pytest.approx(1.0)   # Perfect positive correlation

def test_calculate_correlation_matrix_with_exclusion():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [4, 3, 2, 1],
        'C': [1, 2, 3, 4]
    })
    
    result = calculate_correlation_matrix(df, exclude_columns=['B'])
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'A', 'C'}
    assert 'B' not in result.columns

def test_calculate_correlation_matrix_no_numerical():
    df = pd.DataFrame({
        'cat1': ['A', 'B', 'C', 'D'],
        'cat2': ['W', 'X', 'Y', 'Z']
    })
    
    with pytest.raises(ValueError, match="No numerical features found in DataFrame"):
        calculate_correlation_matrix(df)

def test_calculate_correlation_matrix_mixed_types():
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4],
        'cat1': ['A', 'B', 'C', 'D'],
        'num2': [4, 3, 2, 1]
    })
    
    result = calculate_correlation_matrix(df)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'num1', 'num2'}  # Only numerical columns
    assert result.loc['num1', 'num2'] == pytest.approx(-1.0)  # Perfect negative correlation

def test_calculate_correlation_matrix_empty():
    df = pd.DataFrame(columns=['A', 'B'])
    
    with pytest.raises(ValueError, match="No numerical features found in DataFrame"):
        calculate_correlation_matrix(df)

def test_group_correlated_features_basic():
    # Create correlation matrix with two groups of correlated features
    corr_matrix = pd.DataFrame({
        'A': [np.nan, 0.9, 0.2, 0.1],
        'B': [0.9, np.nan, 0.3, 0.2],
        'C': [0.2, 0.3, np.nan, 0.95],
        'D': [0.1, 0.2, 0.95, np.nan]
    }, index=['A', 'B', 'C', 'D'])
    
    groups = group_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(groups) == 2
    assert ['A', 'B'] in groups
    assert ['C', 'D'] in groups

def test_group_correlated_features_no_correlations():
    # Create correlation matrix with no high correlations
    corr_matrix = pd.DataFrame({
        'A': [np.nan, 0.3, 0.2],
        'B': [0.3, np.nan, 0.1],
        'C': [0.2, 0.1, np.nan]
    }, index=['A', 'B', 'C'])
    
    groups = group_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(groups) == 0

def test_group_correlated_features_chain():
    # Create correlation matrix with features forming a chain A-B-C
    corr_matrix = pd.DataFrame({
        'A': [np.nan, 0.9, 0.7],
        'B': [0.9, np.nan, 0.85],
        'C': [0.7, 0.85, np.nan]
    }, index=['A', 'B', 'C'])
    
    groups = group_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(groups) == 1
    assert groups[0] == ['A', 'B', 'C']

def test_group_correlated_features_invalid_threshold():
    corr_matrix = pd.DataFrame({
        'A': [np.nan, 0.5],
        'B': [0.5, np.nan]
    }, index=['A', 'B'])
    
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        group_correlated_features(corr_matrix, threshold=1.5)

def test_group_correlated_features_empty():
    corr_matrix = pd.DataFrame(columns=['A', 'B'], index=['A', 'B'])
    
    groups = group_correlated_features(corr_matrix)
    
    assert len(groups) == 0

def test_group_correlated_features_single_feature():
    corr_matrix = pd.DataFrame({'A': [np.nan]}, index=['A'])
    
    groups = group_correlated_features(corr_matrix)
    
    assert len(groups) == 0

def test_identify_highly_correlated_features_basic():
    # Create correlation matrix with known correlations
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.9, 0.2],
        'B': [0.9, 1.0, 0.3],
        'C': [0.2, 0.3, 1.0]
    }, index=['A', 'B', 'C'])
    
    pairs = identify_highly_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(pairs) == 1
    assert ('A', 'B') in pairs

def test_identify_highly_correlated_features_no_correlations():
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.2, 0.3],
        'B': [0.2, 1.0, 0.1],
        'C': [0.3, 0.1, 1.0]
    }, index=['A', 'B', 'C'])
    
    pairs = identify_highly_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(pairs) == 0

def test_identify_highly_correlated_features_multiple_pairs():
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.9, 0.9],
        'B': [0.9, 1.0, 0.8],
        'C': [0.9, 0.8, 1.0]
    }, index=['A', 'B', 'C'])
    
    pairs = identify_highly_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(pairs) == 3
    assert ('A', 'B') in pairs
    assert ('A', 'C') in pairs
    assert ('B', 'C') in pairs

def test_identify_highly_correlated_features_invalid_threshold():
    corr_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    }, index=['A', 'B'])
    
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        identify_highly_correlated_features(corr_matrix, threshold=1.5)

def test_identify_highly_correlated_features_with_nan():
    corr_matrix = pd.DataFrame({
        'A': [1.0, np.nan, 0.9],
        'B': [np.nan, 1.0, 0.8],
        'C': [0.9, 0.8, 1.0]
    }, index=['A', 'B', 'C'])
    
    pairs = identify_highly_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(pairs) == 2
    assert ('A', 'C') in pairs
    assert ('B', 'C') in pairs

def test_identify_highly_correlated_features_empty():
    corr_matrix = pd.DataFrame(columns=['A', 'B'], index=['A', 'B'])
    
    pairs = identify_highly_correlated_features(corr_matrix, threshold=0.8)
    
    assert len(pairs) == 0

def test_group_correlated_features_overlapping_groups():
    # Create correlation matrix with:
    # 1. Initial groups will be: [A,B,C], [C,D,E]
    # 2. These should merge due to common feature C into [A,B,C,D,E]
    corr_matrix = pd.DataFrame({
        'A': [np.nan, 0.85, 0.85, 0.2, 0.1],
        'B': [0.85, np.nan, 0.85, 0.3, 0.2],
        'C': [0.85, 0.85, np.nan, 0.85, 0.85],  # C is common to both groups
        'D': [0.2, 0.3, 0.85, np.nan, 0.85],
        'E': [0.1, 0.2, 0.85, 0.85, np.nan]
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    groups = group_correlated_features(corr_matrix, threshold=0.8)
    
    # Should merge into one group due to common feature C
    assert len(groups) == 1
    assert groups[0] == ['A', 'B', 'C', 'D', 'E']  # All features merged via common feature C

def test_group_correlated_features_disjoint():
    # Create correlation matrix with two disjoint groups:
    # [A,B,C] and [D,E] with no common features
    corr_matrix = pd.DataFrame({
        'A': [np.nan, 0.85, 0.85, 0.2, 0.1],
        'B': [0.85, np.nan, 0.85, 0.3, 0.2],
        'C': [0.85, 0.85, np.nan, 0.1, 0.3],
        'D': [0.2, 0.3, 0.1, np.nan, 0.85],
        'E': [0.1, 0.2, 0.3, 0.85, np.nan]
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    groups = group_correlated_features(corr_matrix, threshold=0.8)
    
    # Should stay as two separate groups since no common features
    assert len(groups) == 2
    assert ['A', 'B', 'C'] in groups
    assert ['D', 'E'] in groups
