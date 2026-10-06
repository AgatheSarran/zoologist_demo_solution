"""
This module loads a trained machine learning model and makes predictions on new data.
"""

# Import necessary libraries
import os
import joblib
import pandas as pd

from sklearn import set_config
set_config(transform_output="pandas")


from zoologist.cloud_utils import download_blob


def load_model():
    """Load a trained model, either from the local drive or from Cloud Storage."""
    model_path = os.environ.get("MODEL_PATH")
    model_source = os.environ.get("MODEL_LOCATION")
    bucket_name = os.environ.get("BUCKET_NAME")

    if model_source == "gcs":
        local_model_path = "models/model_from_gcs.joblib"
        download_blob(bucket_name, model_path, local_model_path)
        model_path = local_model_path

    model = joblib.load(model_path)
    return model


def predict(model, data):
    """Make predictions using the loaded model and input data."""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return predictions, probabilities.max(axis=1)


if __name__ == "__main__":
    # Example usage
    model = load_model()

    # Sample input data
    sample_data = pd.read_csv("data/sample_input.csv")

    # Run predictions
    predictions, probabilities = predict(model, sample_data)
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
