"""
Pipeline Module
End-to-end AutoML pipeline orchestrator.
"""

from src.data_ingestion import load_data, profile_data
from src.preprocessing import handle_missing, encode_categoricals, scale_features, split_data
from src.model_selection import get_classifiers, get_regressors, detect_task_type
from src.training import train_models
from src.evaluation import evaluate_classification, evaluate_regression, get_best_model


class AutoMLPipeline:
    """Orchestrates the full automated ML pipeline."""

    def __init__(self):
        self.df = None
        self.profile = None
        self.task_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models_results = None
        self.comparison_df = None
        self.best_model_name = None
        self.best_model = None

    def ingest(self, file):
        """Load data and generate profile."""
        self.df = load_data(file)
        self.profile = profile_data(self.df)
        return self.df, self.profile

    def preprocess(self, target_col, missing_strategy='mean',
                   scale_method='standard', test_size=0.2):
        """Clean, encode, scale, and split the data."""
        df = handle_missing(self.df, strategy=missing_strategy)
        df, _ = encode_categoricals(df, target_col)

        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.task_type = detect_task_type(y)

        X, _ = scale_features(X, method=scale_method)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=test_size
        )
        return self.task_type

    def train(self):
        """Train all models for the detected task type."""
        if self.task_type == 'classification':
            models = get_classifiers()
        else:
            models = get_regressors()
        self.models_results = train_models(models, self.X_train, self.y_train)
        return self.models_results

    def evaluate(self):
        """Evaluate all models and select the best one."""
        if self.task_type == 'classification':
            self.comparison_df = evaluate_classification(
                self.models_results, self.X_test, self.y_test
            )
        else:
            self.comparison_df = evaluate_regression(
                self.models_results, self.X_test, self.y_test
            )
        self.best_model_name, self.best_model = get_best_model(
            self.comparison_df, self.models_results
        )
        return self.comparison_df, self.best_model_name
