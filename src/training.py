"""
Training Module
Handles model training with timing.
"""

import time


def train_models(models, X_train, y_train):
    """Train all models and record training time."""
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - start, 3)
        results[name] = {
            'model': model,
            'training_time': elapsed,
        }
    return results
