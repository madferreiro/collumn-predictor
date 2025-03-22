import pandas as pd
import numpy as np
from typing import List, Tuple, Set

def compute_correlation_scores(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Calculate correlation scores between features and target variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and target
        target_column (str): Name of the target column
        
    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance_score']
                     where importance_score is absolute correlation with target
    """
    # Handle empty DataFrame
    if df.empty:
        return pd.DataFrame(columns=['feature', 'importance_score'])
    
    # Get numerical columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' must be numeric")
    
    # Calculate correlations with target
    correlations = numeric_df.corr()[target_column]
    
    # Convert to DataFrame with absolute values
    result_df = pd.DataFrame({
        'feature': correlations.index,
        'importance_score': correlations.abs()
    })
    
    # Remove target column from results and sort by importance
    result_df = result_df[result_df['feature'] != target_column]
    result_df = result_df.sort_values('importance_score', ascending=False)
    
    return result_df

def calculate_correlation_matrix(df: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix between numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        exclude_columns (List[str], optional): Columns to exclude from correlation calculation
        
    Returns:
        pd.DataFrame: Correlation matrix for numerical features
    """
    # Get numerical columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    if exclude_columns:
        numeric_df = numeric_df.drop(columns=[col for col in exclude_columns if col in numeric_df.columns])
    
    if numeric_df.empty:
        raise ValueError("No numerical features found in DataFrame")
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Set diagonal to NaN to exclude self-correlations
    np.fill_diagonal(correlation_matrix.values, np.nan)
    
    return correlation_matrix

def identify_highly_correlated_features(correlation_matrix: pd.DataFrame, threshold: float) -> List[Tuple[str, str]]:
    """
    Find pairs of features that have correlation above the threshold.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix from calculate_correlation_matrix
        threshold (float): Correlation threshold to consider features as correlated
        
    Returns:
        List[Tuple[str, str]]: List of feature pairs that are highly correlated
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
            corr = abs(correlation_matrix.iloc[i, j])
            if corr >= threshold:
                corr_pairs.append((col1, col2))
    
    return corr_pairs

def group_correlated_features(correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[List[str]]:
    """
    Group features that are highly correlated with each other.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix from calculate_correlation_matrix
        threshold (float): Correlation threshold to consider features as correlated (default: 0.8)
        
    Returns:
        List[List[str]]: List of feature groups where features within each group are highly correlated
    """
    # Get highly correlated pairs
    corr_pairs = identify_highly_correlated_features(correlation_matrix, threshold)
    
    if not corr_pairs:
        return []
    
    # Group correlated features using sets
    feature_groups: List[Set[str]] = []
    for feat1, feat2 in corr_pairs:
        # Try to add to existing group
        added = False
        for group in feature_groups:
            if feat1 in group or feat2 in group:
                group.add(feat1)
                group.add(feat2)
                added = True
                break
        
        # Create new group if not added to existing
        if not added:
            feature_groups.append({feat1, feat2})
    
    # Merge groups with common features
    i = 0
    while i < len(feature_groups):
        j = i + 1
        while j < len(feature_groups):
            if feature_groups[i] & feature_groups[j]:  # If groups have common features
                feature_groups[i] = feature_groups[i] | feature_groups[j]  # Merge groups
                feature_groups.pop(j)
            else:
                j += 1
        i += 1
    
    # Convert sets to sorted lists
    return [sorted(group) for group in feature_groups]
