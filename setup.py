from setuptools import setup, find_packages

setup(
    name="column_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.3",
        "numpy>=2.2.3",
        "scikit-learn>=1.6.1",
        "pyyaml>=6.0.2",
    ],
    author="Caylent",
    description="Determine which columns in a dataset best predict a target variable",
    python_requires=">=3.11",
) 