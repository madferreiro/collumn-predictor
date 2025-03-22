#!/usr/bin/env python3

import argparse
from .main import analyze_features

def main():
    parser = argparse.ArgumentParser(
        description='Analyze which columns in a dataset best predict a target variable.'
    )
    parser.add_argument(
        'filename',
        help='Name of the CSV file in the data directory'
    )
    parser.add_argument(
        'target_column',
        help='Name of the column to predict'
    )
    
    args = parser.parse_args()
    
    result = analyze_features(args.filename, args.target_column)
    
    if not result.success:
        print(f"Error: {result.error_message}")
        print("\nPlease check if:")
        print("- The file exists in the data directory")
        print("- The file is a valid CSV")
        print("- The target column exists in the file")
        exit(1)
        
    print("\nMost relevant features:")
    for feature in result.result:
        print(f"- {feature}")

if __name__ == '__main__':
    main() 