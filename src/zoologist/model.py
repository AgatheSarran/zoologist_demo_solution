"""Build the classification pipeline used to identify penguin species."""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from zoologist.preprocess import preprocessor


def create_pipeline():
    """Create a preprocessing and logistic-regression classification pipeline.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted species-classification pipeline.
    """
    model = LogisticRegression()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline
