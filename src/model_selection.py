"""
Model Selection Module
Provides algorithm registries and task-type auto-detection.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier


def get_classifiers():
    """Return a dict of classifier instances."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }


def get_regressors():
    """Return a dict of regressor instances."""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    }


def detect_task_type(y):
    """Auto-detect whether the task is classification or regression."""
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    unique_ratio = y.nunique() / len(y)
    if y.nunique() <= 20 or unique_ratio < 0.05:
        return 'classification'
    return 'regression'
