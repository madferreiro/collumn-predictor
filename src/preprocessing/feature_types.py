from typing import Tuple, List
import pandas as pd
import numpy as np

def detect_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect which columns are numerical and which are categorical.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Tuple[List[str], List[str]]: Lists of column names and their types ('numerical' or 'categorical')
    """
    column_names = df.columns.tolist()
    column_types = []
    
    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types.append('numerical')
        # Check if column is categorical (object/string type)
        elif pd.api.types.is_object_dtype(df[col]):
            column_types.append('categorical')
        # Default to numerical for other types
        else:
            column_types.append('numerical')
    
    return column_names, column_types
