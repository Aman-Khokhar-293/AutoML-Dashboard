"""
Preprocessing Module
Handles missing values, encoding, scaling, and train/test splitting.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def handle_missing(df, strategy='mean'):
    """Handle missing values using the specified strategy."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        for col in categorical_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    elif strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in categorical_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    elif strategy == 'mode':
        for col in df.columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def encode_categoricals(df, target_col):
    """Encode categorical columns using Label Encoding."""
    df = df.copy()
    le_dict = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    return df, le_dict


def scale_features(X, method='standard'):
    """Scale numeric features using the specified method."""
    if method == 'none':
        return X, None

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        return X, None

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
