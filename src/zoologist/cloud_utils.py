"""
Utility functions to interact with Google Cloud Storage.
"""

from google.cloud import storage


def download_blob(
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str
):
    """Downloads a blob from the bucket.

    Params:
    - bucket_name: str, name of the GCS bucket
    - source_blob_name: str, name of the blob in the bucket
    - destination_file_name: str, local path to save the downloaded file
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_blob(
    bucket_name: str,
    destination_blob_name: str,
    source_file_name: str
):
    """Uploads a blob to the bucket.

    Params:
    - bucket_name: str, name of the GCS bucket
    - destination_blob_name: str, name of the blob in the bucket
    - source_file_name: str, local path of the file to upload
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_file_name)
    blob.upload_from_filename(destination_blob_name)
    