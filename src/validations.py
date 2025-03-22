import os
import pandas as pd
from typing import Tuple

def file_exists(filename: str) -> Tuple[bool, str]:
    """
    Check if file exists in the data directory.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        Tuple[bool, str]: (exists, error_message)
    """
    file_path = os.path.join('data', filename)
    exists = os.path.exists(file_path)
    error_message = "" if exists else f"File '{filename}' not found in data directory"
    return exists, error_message

def is_valid_csv(filename: str) -> Tuple[bool, str]:
    """
    Check if file is a valid CSV.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    file_path = os.path.join('data', filename)
    try:
        pd.read_csv(file_path)
        return True, ""
    except pd.errors.EmptyDataError:
        return False, f"File '{filename}' is empty"
    except pd.errors.ParserError:
        return False, f"File '{filename}' is not a valid CSV file"
    except Exception as e:
        return False, f"Error reading '{filename}': {str(e)}"

def has_target_column(filename: str, target_column: str) -> Tuple[bool, str]:
    """
    Check if target column exists in the CSV file.
    
    Args:
        filename (str): Name of the file to check
        target_column (str): Name of the column to check
        
    Returns:
        Tuple[bool, str]: (has_column, error_message)
    """
    file_path = os.path.join('data', filename)
    try:
        df = pd.read_csv(file_path)
        has_column = target_column in df.columns
        error_message = "" if has_column else f"Column '{target_column}' not found in '{filename}'"
        return has_column, error_message
    except Exception:
        return False, f"Could not check for column '{target_column}' due to file read error" 