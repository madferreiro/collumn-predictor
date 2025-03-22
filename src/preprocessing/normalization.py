import pandas as pd

def normalize_features(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Normalize numerical features in DataFrame using min-max scaling if normalize is True.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        normalize (bool): Flag to determine whether to perform normalization
        
    Returns:
        pd.DataFrame: DataFrame with normalized numerical columns (if normalize is True), others unchanged
    """
    if not normalize:
        return df
    
    df_normalized = df.copy()
    numeric_cols = df_normalized.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()
        if col_max - col_min == 0:
            # Column is constant, set to 0.0
            df_normalized[col] = 0.0
        else:
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    return df_normalized
