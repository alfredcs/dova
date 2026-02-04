"""S3 Manager for DOVA Lambda deployment artifacts.

Handles S3 bucket creation and Lambda package uploads.
"""

import hashlib
from dataclasses import dataclass

import boto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)


@dataclass
class S3UploadResult:
    """Result of S3 upload operation."""

    success: bool
    bucket: str | None = None
    key: str | None = None
    version_id: str | None = None
    error: str | None = None


class S3Manager:
    """Manages S3 resources for DOVA Lambda deployments."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self.sts = boto3.client("sts", region_name=region)
        self._logger = logger.bind(component="s3_manager")

    def get_account_id(self) -> str:
        """Get the current AWS account ID."""
        return self.sts.get_caller_identity()["Account"]

    def get_deployment_bucket_name(self, stack_name: str) -> str:
        """Generate a unique deployment bucket name.

        Args:
            stack_name: Stack name for the deployment

        Returns:
            S3 bucket name (must be globally unique)
        """
        account_id = self.get_account_id()
        return f"{stack_name}-deploy-{account_id}-{self.region}"

    def ensure_deployment_bucket(self, stack_name: str) -> str | None:
        """Ensure the deployment bucket exists.

        Creates the bucket if it doesn't exist, with versioning enabled.

        Args:
            stack_name: Stack name for the deployment

        Returns:
            Bucket name if successful, None if failed
        """
        bucket_name = self.get_deployment_bucket_name(stack_name)
        self._logger.info("ensuring_deployment_bucket", bucket=bucket_name)

        try:
            # Check if bucket exists
            try:
                self.s3.head_bucket(Bucket=bucket_name)
                self._logger.info("bucket_exists", bucket=bucket_name)
                return bucket_name
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code not in ("404", "NoSuchBucket"):
                    raise

            # Create bucket
            create_params = {"Bucket": bucket_name}

            # LocationConstraint is required for non-us-east-1 regions
            if self.region != "us-east-1":
                create_params["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.region
                }

            self.s3.create_bucket(**create_params)
            self._logger.info("bucket_created", bucket=bucket_name)

            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={"Status": "Enabled"},
            )

            # Add tags
            self.s3.put_bucket_tagging(
                Bucket=bucket_name,
                Tagging={
                    "TagSet": [
                        {"Key": "ManagedBy", "Value": "dova"},
                        {"Key": "Stack", "Value": stack_name},
                    ]
                },
            )

            return bucket_name

        except ClientError as e:
            self._logger.error("bucket_creation_failed", error=str(e))
            return None

    def upload_lambda_package(
        self, bucket: str, zip_path: str, stack_name: str
    ) -> S3UploadResult:
        """Upload Lambda deployment package to S3.

        Args:
            bucket: S3 bucket name
            zip_path: Path to the ZIP file to upload
            stack_name: Stack name for organizing artifacts

        Returns:
            S3UploadResult with upload details
        """
        self._logger.info("uploading_lambda_package", bucket=bucket, path=zip_path)

        try:
            # Calculate content hash for deduplication
            with open(zip_path, "rb") as f:
                content = f.read()
                content_hash = hashlib.sha256(content).hexdigest()[:12]

            # Generate key with hash for versioning
            key = f"{stack_name}/lambda/dova-{content_hash}.zip"

            # Check if this exact version already exists
            try:
                self.s3.head_object(Bucket=bucket, Key=key)
                self._logger.info("package_already_exists", key=key)
                return S3UploadResult(success=True, bucket=bucket, key=key)
            except ClientError:
                pass  # Object doesn't exist, proceed with upload

            # Upload the file
            with open(zip_path, "rb") as f:
                response = self.s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=f,
                    ContentType="application/zip",
                    Metadata={
                        "stack-name": stack_name,
                        "content-hash": content_hash,
                    },
                )

            version_id = response.get("VersionId")
            self._logger.info(
                "package_uploaded",
                bucket=bucket,
                key=key,
                version_id=version_id,
            )

            return S3UploadResult(
                success=True,
                bucket=bucket,
                key=key,
                version_id=version_id,
            )

        except FileNotFoundError:
            return S3UploadResult(
                success=False,
                error=f"Package file not found: {zip_path}",
            )
        except ClientError as e:
            error_msg = e.response["Error"]["Message"]
            self._logger.error("upload_failed", error=error_msg)
            return S3UploadResult(success=False, error=error_msg)

    def delete_deployment_artifacts(self, stack_name: str) -> bool:
        """Delete all deployment artifacts for a stack.

        Args:
            stack_name: Stack name to clean up

        Returns:
            True if successful
        """
        bucket_name = self.get_deployment_bucket_name(stack_name)
        self._logger.info("deleting_deployment_artifacts", bucket=bucket_name)

        try:
            # Check if bucket exists
            try:
                self.s3.head_bucket(Bucket=bucket_name)
            except ClientError:
                self._logger.info("bucket_not_found", bucket=bucket_name)
                return True

            # Delete all objects (including versions)
            paginator = self.s3.get_paginator("list_object_versions")
            for page in paginator.paginate(Bucket=bucket_name):
                objects_to_delete = []

                for version in page.get("Versions", []):
                    objects_to_delete.append({
                        "Key": version["Key"],
                        "VersionId": version["VersionId"],
                    })

                for marker in page.get("DeleteMarkers", []):
                    objects_to_delete.append({
                        "Key": marker["Key"],
                        "VersionId": marker["VersionId"],
                    })

                if objects_to_delete:
                    self.s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={"Objects": objects_to_delete},
                    )

            # Delete the bucket
            self.s3.delete_bucket(Bucket=bucket_name)
            self._logger.info("bucket_deleted", bucket=bucket_name)

            return True

        except ClientError as e:
            self._logger.error("artifact_cleanup_failed", error=str(e))
            return False
