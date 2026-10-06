"""Train, validate, and persist the penguin classification pipeline."""

from sklearn.model_selection import train_test_split, cross_val_score

from zoologist.data import get_data
from zoologist.model import create_pipeline
from zoologist.registry import save_model

def prepare_training_data():
    """Load penguin data and split its features and labels into train/test sets.

    Returns:
        tuple: Training and test feature matrices followed by their labels.
    """
    penguin_df = get_data()

    X = penguin_df.drop(columns=['species'])
    y = penguin_df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test


def cross_validate(pipeline, X_train, y_train):
    """Return the mean five-fold accuracy of a pipeline on training data."""
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    return cv_score.mean()


def train(pipeline):
    """Evaluate a pipeline with cross-validation, then fit it on training data."""
    X_train, X_test, y_train, y_test = prepare_training_data()

    cv_score = cross_validate(pipeline, X_train, y_train)
    print("Cross-validated accuracy:", cv_score)
    
    return pipeline.fit(X_train, y_train)


def main():
    """Create, train, save, and optionally upload the model pipeline."""
    pipeline = create_pipeline()
    
    trained_pipeline = train(pipeline)
    
    print("Model trained")
    
    save_model(trained_pipeline)
    
    
if __name__ == "__main__":
    main()