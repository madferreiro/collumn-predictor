import pytest
import pandas as pd
from src.main import analyze_features, Response

def test_analyze_features_nonexistent_file():
    result = analyze_features("nonexistent.csv", target_column="target")
    assert isinstance(result, Response)
    assert result.success is False
    assert result.result == []

def test_analyze_features_empty_result():
    # TODO: Add test with actual CSV file once we implement the analysis
    result = analyze_features("test.csv", target_column="target")
    assert isinstance(result, Response)
    assert result.success is True
    assert isinstance(result.result, list)
    assert "target" not in result.result

def test_analyze_features_invalid_csv():
    # Create an invalid CSV file
    with open('data/invalid.csv', 'w') as f:
        f.write('This is not a valid CSV file')
    
    result = analyze_features("invalid.csv", target_column="target")
    assert isinstance(result, Response)
    assert result.success is False
    assert result.result == []
    
    # Clean up
    import os
    os.remove('data/invalid.csv')

def test_analyze_features_empty_csv():
    # Create an empty CSV file
    with open('data/empty.csv', 'w') as f:
        f.write('')
    
    result = analyze_features("empty.csv", target_column="target")
    assert isinstance(result, Response)
    assert result.success is False
    assert result.result == []
    
    # Clean up
    import os
    os.remove('data/empty.csv')

def test_analyze_features_generic_exception():
    # Create a file that exists but will cause a read error
    with open('data/bad.csv', 'w') as f:
        f.write('column1,column2\n1,2\n"unclosed_quote,3')  # Malformed CSV
    
    result = analyze_features("bad.csv", target_column="target")
    assert isinstance(result, Response)
    assert result.success is False
    assert result.result == []
    
    # Clean up
    import os
    os.remove('data/bad.csv')
