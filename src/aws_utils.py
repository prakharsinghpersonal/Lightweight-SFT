import boto3
import os
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class AWSClient:
    def __init__(self, region_name="us-west-2"):
        self.s3_client = boto3.client("s3", region_name=region_name)

    def upload_file(self, file_name, bucket, object_name=None):
        """Upload a file to an S3 bucket"""
        if object_name is None:
            object_name = file_name

        try:
            self.s3_client.upload_file(file_name, bucket, object_name)
            logger.info(f"Successfully uploaded {file_name} to {bucket}/{object_name}")
        except ClientError as e:
            logger.error(f"Failed to upload {file_name} to S3: {e}")
            return False
        return True

    def download_file(self, bucket, object_name, file_name):
        """Download a file from an S3 bucket"""
        try:
            self.s3_client.download_file(bucket, object_name, file_name)
            logger.info(f"Successfully downloaded {bucket}/{object_name} to {file_name}")
        except ClientError as e:
            logger.error(f"Failed to download {object_name} from S3: {e}")
            return False
        return True

    def sync_checkpoints(self, local_dir, bucket, s3_prefix):
        """Syncs local checkpoints directory to S3"""
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_path = os.path.join(s3_prefix, relative_path)
                self.upload_file(local_path, bucket, s3_path)
