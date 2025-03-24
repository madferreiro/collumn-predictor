import os
from dataclasses import dataclass
from typing import List, Any
import pandas as pd
from .preprocessing.missing_values import handle_missing_values
from .validations import file_exists, is_valid_csv, has_target_column
from .analysis.correlation import compute_correlation_scores

@dataclass
class Response:
    success: bool
    result: List[Any]
    error_message: str = ""

def analyze_features(filename: str, target_column: str, debug: bool = False) -> Response:
    """
    Analyze features in a CSV file to determine which columns best predict a target variable.
    
    Args:
        filename (str): Name of the CSV file in the data directory
        target_column (str): Name of the column to predict
        debug (bool): If True, prints debug information including correlation matrix
        
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
    
    if debug:
        # Get numerical columns only for correlation matrix
        numeric_df = df_cleaned.select_dtypes(include=['number'])
        if not numeric_df.empty:
            print("\nCorrelation Matrix:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(numeric_df.corr().round(3))
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')
    
    # Calculate correlation scores
    correlation_df = compute_correlation_scores(df_cleaned, target_column)
    
    # Return features sorted by importance (absolute correlation)
    return Response(success=True, result=correlation_df['feature'].tolist())
