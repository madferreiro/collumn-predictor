import pandas as pd

def rank_features_by_correlation(correlation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank features based on their absolute correlation values.
    
    Args:
        correlation_df (pd.DataFrame): DataFrame with columns ['feature', 'importance_score']
        
    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance_score', 'rank']
                     where rank is based on importance_score (ties allowed)
    """
    if not all(col in correlation_df.columns for col in ['feature', 'importance_score']):
        raise ValueError("Input DataFrame must have 'feature' and 'importance_score' columns")
        
    # Create copy to avoid modifying input
    result_df = correlation_df.copy()
    
    # Add rank column (1-based, ties allowed)
    result_df['rank'] = result_df['importance_score'].rank(ascending=False, method='min')
    
    # Sort by rank and then feature name for consistent output
    result_df = result_df.sort_values(['rank', 'feature'])
    
    return result_df
