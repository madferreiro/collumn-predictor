import pytest
from src.main import analyze_features, Response

def test_analyze_features_nonexistent_file():
    result = analyze_features("nonexistent.csv", target_column="Salary")
    assert isinstance(result, Response)
    assert result.success is False
    assert result.result == []

def test_analyze_features_empty_result():
    result = analyze_features("minimal.csv", target_column="Salary")
    assert isinstance(result, Response)
    assert result.success is True
    assert isinstance(result.result, list)
    assert "Salary" not in result.result
