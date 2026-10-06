"""Persist a trained pipeline."""

import os
import time
import joblib

from zoologist.cloud_utils import upload_blob


MODEL_PATH = os.environ.get("MODEL_PATH")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_LOCATION = os.environ.get("MODEL_LOCATION")
MODEL_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "zoo_training_outputs")

timestamp = time.strftime("%Y%m%d-%H%M%S")
# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join( "models", f"{timestamp}.h5")

def save_model(pipeline):
    """Serialize a trained pipeline locally and optionally upload it to GCS.

    Args:
        pipeline: The fitted scikit-learn pipeline to persist.
    """
    with open(MODEL_PATH, 'wb') as f:
        joblib.dump(pipeline, f)

    print("Saved pipeline locally to:", MODEL_PATH)

    if MODEL_LOCATION.lower() == "gcs":
        upload_blob(BUCKET_NAME, MODEL_PATH, MODEL_PATH)

        print(f"Uploaded model to: gs://{BUCKET_NAME}/{MODEL_PATH}")
