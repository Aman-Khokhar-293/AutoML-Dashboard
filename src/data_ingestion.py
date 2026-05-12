"""
Data Ingestion Module
Handles loading and profiling of datasets.
"""

import pandas as pd


def load_data(file):
    """Load a CSV or Excel file into a Pandas DataFrame."""
    if hasattr(file, 'name'):
        name = file.name
    else:
        name = str(file)

    if name.endswith('.csv'):
        df = pd.read_csv(file)
    elif name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported format. Upload CSV or Excel.")
    return df


def profile_data(df):
    """Generate a summary profile of the dataset."""
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)

    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
    }
    return profile
