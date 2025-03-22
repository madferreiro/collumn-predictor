import os
from dataclasses import dataclass
from typing import List, Any
import pandas as pd
from src.preprocessing.missing_values import handle_missing_values

@dataclass
class Response:
    success: bool
    result: List[Any]

def analyze_features(filename: str, target_column: str) -> Response:
    """
    Analyze features in a CSV file to determine which columns best predict a target variable.
    
    Args:
        filename (str): Name of the CSV file in the data directory
        target_column (str): Name of the column to predict
        
    Returns:
        Response: Object containing success status and results
    """
    # Check if file exists in data directory
    file_path = os.path.join('data', filename)
    if not os.path.exists(file_path):
        return Response(success=False, result=[])
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return Response(success=False, result=[])
    except Exception as e:
        return Response(success=False, result=[])
    
    # Validate target column exists in DataFrame
    if target_column not in df.columns:
        return Response(success=False, result=[])
    
    # Handle missing values
    df_cleaned, removed_columns = handle_missing_values(df, threshold=0.5)
    
    # Return all columns except target
    feature_columns = [col for col in df_cleaned.columns if col != target_column]
    return Response(success=True, result=feature_columns)
