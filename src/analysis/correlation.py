import pandas as pd
import numpy as np
from typing import List, Tuple

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
