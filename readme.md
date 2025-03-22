# Algorithm: Automatic Feature Importance Detection

## Overview
Determine which columns in a dataset best predict a target variable by ranking features based on statistical correlation analysis.

## Inputs
- `data`: DataFrame containing features and target  
- `target_column`: Name of column to predict  
- `missing_threshold`: Maximum percentage of missing values allowed (default: 50%)  
- `normalize`: Whether to normalize numerical features (default: True)  
- `max_one_hot_categories`: Maximum number of one-hot encoded columns per categorical feature (defined in configuration)  
- `correlation_threshold`: Threshold to consider features highly correlated (defined in configuration)  

## Outputs
- `feature_importance`: DataFrame with columns `['feature', 'importance_score', 'rank']`  
  - `importance_score`: Rational number between 0 (inclusive) and 1 (inclusive)  
  - `rank`: Features are ranked based on importance, ties are allowed  

## Implementation Guide  

### 1. Data Preprocessing  
**Functions:**  
- `handle_missing_values(data: DataFrame, threshold: float) -> DataFrame`  
  *Removes columns exceeding the missing value threshold.*  
- `detect_feature_types(data: DataFrame) -> Tuple[List[str], List[str]]`  
  *Identifies numerical and categorical columns.*  
- `apply_one_hot_encoding(data: DataFrame, max_categories: int) -> DataFrame`  
  *Encodes categorical variables using one-hot encoding while respecting category limits.*  
- `normalize_features(data: DataFrame, normalize: bool) -> DataFrame`  
  *Applies normalization to numerical features if requested.*  

### 2. Feature Importance Calculation  
**Functions:**  
- `compute_correlation_scores(data: DataFrame, target_column: str) -> DataFrame`  
  *Calculates correlation coefficients between features and the target variable.*  
- `rank_features_by_correlation(correlation_df: DataFrame) -> DataFrame`  
  *Ranks features based on absolute correlation values.*  

### 3. Handling Correlated Features  
**Functions:**  
- `calculate_correlation_matrix(data: DataFrame) -> DataFrame`  
  *Computes the correlation matrix for numerical features.*  
- `identify_highly_correlated_features(corr_matrix: DataFrame, threshold: float) -> List[Tuple[str, str]]`  
  *Finds feature pairs exceeding the correlation threshold.*  
- `group_correlated_features(correlated_pairs: List[Tuple[str, str]]) -> List[List[str]]`  
  *Groups correlated features into clusters for further analysis.*  

### 4. Implementation Order  
1. **Preprocessing Functions:**  
   - Handle missing values (`handle_missing_values`)  
   - Detect feature types (`detect_feature_types`)  
   - Apply one-hot encoding (`apply_one_hot_encoding`)  
   - Normalize numerical features (`normalize_features`)  

2. **Feature Importance Calculation:**  
   - Compute correlation scores (`compute_correlation_scores`)  
   - Rank features based on correlation (`rank_features_by_correlation`)  

3. **Handling Correlation Among Features:**  
   - Compute correlation matrix (`calculate_correlation_matrix`)  
   - Identify highly correlated features (`identify_highly_correlated_features`)  
   - Group correlated features (`group_correlated_features`)  

4. **Return Final Feature Importance DataFrame**  

This structure ensures a clear and modular approach to implementing automatic feature importance detection using statistical correlation analysis.  


## Usage example 
python -m src.cli minimal.csv Salary

Most relevant features:
- Age