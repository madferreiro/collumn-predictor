import os
from dataclasses import dataclass
from typing import List, Any
import pandas as pd
from .preprocessing.missing_values import handle_missing_values
from .validations import file_exists, is_valid_csv, has_target_column

@dataclass
class Response:
    success: bool
    result: List[Any]
    error_message: str = ""

def analyze_features(filename: str, target_column: str) -> Response:
    """
    Analyze features in a CSV file to determine which columns best predict a target variable.
    
    Args:
        filename (str): Name of the CSV file in the data directory
        target_column (str): Name of the column to predict
        
    Returns:
        Response: Object containing success status and results
    """
    # Validate file exists
    exists, error = file_exists(filename)
    if not exists:
        return Response(success=False, result=[], error_message=error)
    
    # Validate CSV is valid
    is_valid, error = is_valid_csv(filename)
    if not is_valid:
        return Response(success=False, result=[], error_message=error)
    
    # Validate target column exists
    has_column, error = has_target_column(filename, target_column)
    if not has_column:
        return Response(success=False, result=[], error_message=error)
    
    # Read CSV file (we know it's valid now)
    file_path = os.path.join('data', filename)
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df_cleaned, removed_columns = handle_missing_values(df, threshold=0.5)
    
    # Return all columns except target
    feature_columns = [col for col in df_cleaned.columns if col != target_column]
    return Response(success=True, result=feature_columns)
