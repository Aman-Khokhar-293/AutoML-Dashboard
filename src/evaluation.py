"""
Evaluation Module
Model comparison, metrics, and best-model selection.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix,
)


def evaluate_classification(models_results, X_test, y_test):
    """Evaluate all classification models and return a comparison DataFrame."""
    rows = []
    for name, result in models_results.items():
        y_pred = result['model'].predict(X_test)
        avg = 'weighted' if len(set(y_test)) > 2 else 'binary'
        rows.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            'F1 Score': round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
            'Train Time (s)': result['training_time'],
        })
    df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    return df


def evaluate_regression(models_results, X_test, y_test):
    """Evaluate all regression models and return a comparison DataFrame."""
    rows = []
    for name, result in models_results.items():
        y_pred = result['model'].predict(X_test)
        rows.append({
            'Model': name,
            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'R² Score': round(r2_score(y_test, y_pred), 4),
            'Train Time (s)': result['training_time'],
        })
    df = pd.DataFrame(rows).sort_values('R² Score', ascending=False).reset_index(drop=True)
    return df


def get_confusion_matrix(model, X_test, y_test):
    """Return confusion matrix for a classification model."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_best_model(comparison_df, models_results):
    """Return the name and object of the best-performing model."""
    best_name = comparison_df.iloc[0]['Model']
    best_model = models_results[best_name]['model']
    return best_name, best_model
