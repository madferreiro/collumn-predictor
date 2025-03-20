from typing import List, Tuple
import pandas as pd

def apply_one_hot_encoding(df: pd.DataFrame, max_categories: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply one-hot encoding to categorical columns while respecting category limits.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        max_categories (int): Maximum number of categories to encode per feature
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with encoded columns and list of original categorical columns
    """
    # Store original categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Create a copy to avoid modifying the original
    df_encoded = df.copy()
    
    for col in categorical_columns:
        # Get unique values and their counts
        value_counts = df[col].value_counts()
        
        # If number of categories exceeds max_categories, keep only the most frequent ones
        if len(value_counts) > max_categories:
            # Keep the most frequent categories
            top_categories = value_counts.head(max_categories).index
            # Replace other categories with 'other'
            df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_categories else 'other')
        
        # Apply one-hot encoding
        dummies = pd.get_dummies(df_encoded[col], prefix=col)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Drop the original categorical column
        df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded, categorical_columns
