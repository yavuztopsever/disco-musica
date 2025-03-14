"""
Cloud Platform Utilities

This module provides utilities for integrating with cloud platforms for training
and storage (AWS, Azure, etc.).
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union


class CloudStorageClient:
    """
    A class for interacting with cloud storage platforms.
    """

    def __init__(self, platform: str, credentials: Optional[Dict] = None):
        """
        Initialize the CloudStorageClient.

        Args:
            platform: Cloud platform ('aws', 'azure', 'gcp').
            credentials: Platform-specific credentials.
        """
        self.platform = platform.lower()
        self.credentials = credentials
        self.client = None
        
        # Initialize the appropriate client
        if self.platform == "aws":
            self._init_aws_client()
        elif self.platform == "azure":
            self._init_azure_client()
        elif self.platform == "gcp":
            self._init_gcp_client()
        else:
            raise ValueError(f"Unsupported cloud platform: {platform}")
    
    def _init_aws_client(self) -> None:
        """Initialize the AWS S3 client."""
        try:
            import boto3
            
            if self.credentials:
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.credentials.get('access_key_id'),
                    aws_secret_access_key=self.credentials.get('secret_access_key'),
                    region_name=self.credentials.get('region_name', 'us-east-1')
                )
            else:
                # Use environment variables or instance profile
                self.client = boto3.client('s3')
                
            print("Initialized AWS S3 client")
        except ImportError:
            raise ImportError("boto3 is required for AWS integration. Install it with 'pip install boto3'.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AWS S3 client: {e}")
    
    def _init_azure_client(self) -> None:
        """Initialize the Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            if self.credentials:
                connection_string = self.credentials.get('connection_string')
                if connection_string:
                    self.client = BlobServiceClient.from_connection_string(connection_string)
                else:
                    account_name = self.credentials.get('account_name')
                    account_key = self.credentials.get('account_key')
                    if account_name and account_key:
                        self.client = BlobServiceClient(
                            account_url=f"https://{account_name}.blob.core.windows.net",
                            credential=account_key
                        )
                    else:
                        raise ValueError("Either connection_string or both account_name and account_key must be provided")
            else:
                # Use environment variables
                connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
                if connection_string:
                    self.client = BlobServiceClient.from_connection_string(connection_string)
                else:
                    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable must be set")
                
            print("Initialized Azure Blob Storage client")
        except ImportError:
            raise ImportError("azure-storage-blob is required for Azure integration. Install it with 'pip install azure-storage-blob'.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure Blob Storage client: {e}")
    
    def _init_gcp_client(self) -> None:
        """Initialize the Google Cloud Storage client."""
        try:
            from google.cloud import storage
            
            if self.credentials:
                # Create credentials object
                from google.oauth2 import service_account
                
                if 'service_account_info' in self.credentials:
                    # Create credentials from service account info
                    credentials = service_account.Credentials.from_service_account_info(
                        self.credentials['service_account_info']
                    )
                elif 'service_account_file' in self.credentials:
                    # Create credentials from service account file
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials['service_account_file']
                    )
                else:
                    raise ValueError("Either service_account_info or service_account_file must be provided")
                
                self.client = storage.Client(credentials=credentials)
            else:
                # Use environment variables
                self.client = storage.Client()
                
            print("Initialized Google Cloud Storage client")
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCP integration. Install it with 'pip install google-cloud-storage'.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Cloud Storage client: {e}")
    
    def upload_file(
        self, local_path: Union[str, Path], remote_path: str, bucket_name: str
    ) -> str:
        """
        Upload a file to cloud storage.

        Args:
            local_path: Local path of the file to upload.
            remote_path: Remote path to upload the file to.
            bucket_name: Name of the bucket or container.

        Returns:
            URL of the uploaded file.
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        if self.platform == "aws":
            return self._upload_file_aws(local_path, remote_path, bucket_name)
        elif self.platform == "azure":
            return self._upload_file_azure(local_path, remote_path, bucket_name)
        elif self.platform == "gcp":
            return self._upload_file_gcp(local_path, remote_path, bucket_name)
    
    def _upload_file_aws(
        self, local_path: Path, remote_path: str, bucket_name: str
    ) -> str:
        """Upload a file to AWS S3."""
        try:
            # Upload file
            self.client.upload_file(
                Filename=str(local_path),
                Bucket=bucket_name,
                Key=remote_path
            )
            
            # Get URL
            url = f"https://{bucket_name}.s3.amazonaws.com/{remote_path}"
            
            print(f"Uploaded {local_path} to {url}")
            return url
        except Exception as e:
            raise RuntimeError(f"Failed to upload file to AWS S3: {e}")
    
    def _upload_file_azure(
        self, local_path: Path, remote_path: str, container_name: str
    ) -> str:
        """Upload a file to Azure Blob Storage."""
        try:
            # Get container client
            container_client = self.client.get_container_client(container_name)
            
            # Get blob client
            blob_client = container_client.get_blob_client(remote_path)
            
            # Upload file
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Get URL
            url = blob_client.url
            
            print(f"Uploaded {local_path} to {url}")
            return url
        except Exception as e:
            raise RuntimeError(f"Failed to upload file to Azure Blob Storage: {e}")
    
    def _upload_file_gcp(
        self, local_path: Path, remote_path: str, bucket_name: str
    ) -> str:
        """Upload a file to Google Cloud Storage."""
        try:
            # Get bucket
            bucket = self.client.bucket(bucket_name)
            
            # Get blob
            blob = bucket.blob(remote_path)
            
            # Upload file
            blob.upload_from_filename(str(local_path))
            
            # Get URL
            url = f"https://storage.googleapis.com/{bucket_name}/{remote_path}"
            
            print(f"Uploaded {local_path} to {url}")
            return url
        except Exception as e:
            raise RuntimeError(f"Failed to upload file to Google Cloud Storage: {e}")
    
    def download_file(
        self, remote_path: str, local_path: Union[str, Path], bucket_name: str
    ) -> Path:
        """
        Download a file from cloud storage.

        Args:
            remote_path: Remote path of the file to download.
            local_path: Local path to download the file to.
            bucket_name: Name of the bucket or container.

        Returns:
            Path to the downloaded file.
        """
        local_path = Path(local_path)
        
        # Create directory if it doesn't exist
        os.makedirs(local_path.parent, exist_ok=True)
        
        if self.platform == "aws":
            return self._download_file_aws(remote_path, local_path, bucket_name)
        elif self.platform == "azure":
            return self._download_file_azure(remote_path, local_path, bucket_name)
        elif self.platform == "gcp":
            return self._download_file_gcp(remote_path, local_path, bucket_name)
    
    def _download_file_aws(
        self, remote_path: str, local_path: Path, bucket_name: str
    ) -> Path:
        """Download a file from AWS S3."""
        try:
            # Download file
            self.client.download_file(
                Bucket=bucket_name,
                Key=remote_path,
                Filename=str(local_path)
            )
            
            print(f"Downloaded {remote_path} to {local_path}")
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download file from AWS S3: {e}")
    
    def _download_file_azure(
        self, remote_path: str, local_path: Path, container_name: str
    ) -> Path:
        """Download a file from Azure Blob Storage."""
        try:
            # Get container client
            container_client = self.client.get_container_client(container_name)
            
            # Get blob client
            blob_client = container_client.get_blob_client(remote_path)
            
            # Download file
            with open(local_path, "wb") as file:
                blob_data = blob_client.download_blob()
                file.write(blob_data.readall())
            
            print(f"Downloaded {remote_path} to {local_path}")
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download file from Azure Blob Storage: {e}")
    
    def _download_file_gcp(
        self, remote_path: str, local_path: Path, bucket_name: str
    ) -> Path:
        """Download a file from Google Cloud Storage."""
        try:
            # Get bucket
            bucket = self.client.bucket(bucket_name)
            
            # Get blob
            blob = bucket.blob(remote_path)
            
            # Download file
            blob.download_to_filename(str(local_path))
            
            print(f"Downloaded {remote_path} to {local_path}")
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download file from Google Cloud Storage: {e}")
    
    def list_files(
        self, prefix: str, bucket_name: str
    ) -> List[str]:
        """
        List files in cloud storage.

        Args:
            prefix: Prefix to filter files by.
            bucket_name: Name of the bucket or container.

        Returns:
            List of file paths.
        """
        if self.platform == "aws":
            return self._list_files_aws(prefix, bucket_name)
        elif self.platform == "azure":
            return self._list_files_azure(prefix, bucket_name)
        elif self.platform == "gcp":
            return self._list_files_gcp(prefix, bucket_name)
    
    def _list_files_aws(
        self, prefix: str, bucket_name: str
    ) -> List[str]:
        """List files in AWS S3."""
        try:
            # List objects
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            # Extract file paths
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
        except Exception as e:
            raise RuntimeError(f"Failed to list files in AWS S3: {e}")
    
    def _list_files_azure(
        self, prefix: str, container_name: str
    ) -> List[str]:
        """List files in Azure Blob Storage."""
        try:
            # Get container client
            container_client = self.client.get_container_client(container_name)
            
            # List blobs
            return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
        except Exception as e:
            raise RuntimeError(f"Failed to list files in Azure Blob Storage: {e}")
    
    def _list_files_gcp(
        self, prefix: str, bucket_name: str
    ) -> List[str]:
        """List files in Google Cloud Storage."""
        try:
            # Get bucket
            bucket = self.client.bucket(bucket_name)
            
            # List blobs
            return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        except Exception as e:
            raise RuntimeError(f"Failed to list files in Google Cloud Storage: {e}")


class CloudComputeClient:
    """
    A class for interacting with cloud compute platforms.
    """

    def __init__(self, platform: str, credentials: Optional[Dict] = None):
        """
        Initialize the CloudComputeClient.

        Args:
            platform: Cloud platform ('aws', 'azure', 'gcp').
            credentials: Platform-specific credentials.
        """
        self.platform = platform.lower()
        self.credentials = credentials
        self.client = None
        
        # Initialize the appropriate client
        if self.platform == "aws":
            self._init_aws_client()
        elif self.platform == "azure":
            self._init_azure_client()
        elif self.platform == "gcp":
            self._init_gcp_client()
        else:
            raise ValueError(f"Unsupported cloud platform: {platform}")
    
    def _init_aws_client(self) -> None:
        """Initialize the AWS EC2 client."""
        try:
            import boto3
            
            if self.credentials:
                self.client = boto3.client(
                    'ec2',
                    aws_access_key_id=self.credentials.get('access_key_id'),
                    aws_secret_access_key=self.credentials.get('secret_access_key'),
                    region_name=self.credentials.get('region_name', 'us-east-1')
                )
            else:
                # Use environment variables or instance profile
                self.client = boto3.client('ec2')
                
            print("Initialized AWS EC2 client")
        except ImportError:
            raise ImportError("boto3 is required for AWS integration. Install it with 'pip install boto3'.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AWS EC2 client: {e}")
    
    def _init_azure_client(self) -> None:
        """Initialize the Azure Compute client."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            
            if self.credentials:
                subscription_id = self.credentials.get('subscription_id')
                if not subscription_id:
                    raise ValueError("subscription_id must be provided")
                
                # Use provided credentials
                if 'client_id' in self.credentials and 'client_secret' in self.credentials and 'tenant_id' in self.credentials:
                    from azure.identity import ClientSecretCredential
                    credential = ClientSecretCredential(
                        tenant_id=self.credentials['tenant_id'],
                        client_id=self.credentials['client_id'],
                        client_secret=self.credentials['client_secret']
                    )
                else:
                    # Use default credentials
                    credential = DefaultAzureCredential()
                
                self.client = ComputeManagementClient(credential, subscription_id)
            else:
                # Use environment variables
                subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
                if not subscription_id:
                    raise ValueError("AZURE_SUBSCRIPTION_ID environment variable must be set")
                
                credential = DefaultAzureCredential()
                self.client = ComputeManagementClient(credential, subscription_id)
                
            print("Initialized Azure Compute client")
        except ImportError:
            raise ImportError("azure-mgmt-compute and azure-identity are required for Azure integration. Install them with 'pip install azure-mgmt-compute azure-identity'.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure Compute client: {e}")
    
    def _init_gcp_client(self) -> None:
        """Initialize the Google Compute Engine client."""
        try:
            from google.cloud import compute_v1
            
            if self.credentials:
                # Create credentials object
                from google.oauth2 import service_account
                
                if 'service_account_info' in self.credentials:
                    # Create credentials from service account info
                    credentials = service_account.Credentials.from_service_account_info(
                        self.credentials['service_account_info']
                    )
                elif 'service_account_file' in self.credentials:
                    # Create credentials from service account file
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials['service_account_file']
                    )
                else:
                    raise ValueError("Either service_account_info or service_account_file must be provided")
                
                self.client = compute_v1.InstancesClient(credentials=credentials)
            else:
                # Use environment variables
                self.client = compute_v1.InstancesClient()
                
            print("Initialized Google Compute Engine client")
        except ImportError:
            raise ImportError("google-cloud-compute is required for GCP integration. Install it with 'pip install google-cloud-compute'.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Compute Engine client: {e}")
    
    def start_instance(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """
        Start a cloud compute instance.

        Args:
            instance_id: ID of the instance to start.
            **kwargs: Platform-specific arguments.

        Returns:
            Instance information dictionary.
        """
        if self.platform == "aws":
            return self._start_instance_aws(instance_id, **kwargs)
        elif self.platform == "azure":
            return self._start_instance_azure(instance_id, **kwargs)
        elif self.platform == "gcp":
            return self._start_instance_gcp(instance_id, **kwargs)
    
    def _start_instance_aws(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Start an AWS EC2 instance."""
        try:
            response = self.client.start_instances(
                InstanceIds=[instance_id]
            )
            
            print(f"Started AWS EC2 instance: {instance_id}")
            return {
                "platform": "aws",
                "instance_id": instance_id,
                "status": response['StartingInstances'][0]['CurrentState']['Name']
            }
        except Exception as e:
            raise RuntimeError(f"Failed to start AWS EC2 instance: {e}")
    
    def _start_instance_azure(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Start an Azure VM."""
        try:
            # Parse instance ID (resource group and VM name)
            resource_group = kwargs.get('resource_group')
            vm_name = instance_id
            
            if not resource_group:
                raise ValueError("resource_group must be provided")
            
            # Start VM
            self.client.virtual_machines.begin_start(resource_group, vm_name).result()
            
            print(f"Started Azure VM: {vm_name}")
            return {
                "platform": "azure",
                "instance_id": vm_name,
                "resource_group": resource_group,
                "status": "running"
            }
        except Exception as e:
            raise RuntimeError(f"Failed to start Azure VM: {e}")
    
    def _start_instance_gcp(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Start a Google Compute Engine instance."""
        try:
            # Parse instance ID (zone and instance name)
            project = kwargs.get('project')
            zone = kwargs.get('zone')
            instance_name = instance_id
            
            if not project:
                raise ValueError("project must be provided")
            if not zone:
                raise ValueError("zone must be provided")
            
            # Start instance
            operation = self.client.start(
                project=project,
                zone=zone,
                instance=instance_name
            )
            
            # Wait for the operation to complete
            from google.api_core.exceptions import GoogleAPIError
            try:
                operation.result()
                status = "RUNNING"
            except GoogleAPIError:
                status = "ERROR"
            
            print(f"Started Google Compute Engine instance: {instance_name}")
            return {
                "platform": "gcp",
                "instance_id": instance_name,
                "project": project,
                "zone": zone,
                "status": status
            }
        except Exception as e:
            raise RuntimeError(f"Failed to start Google Compute Engine instance: {e}")
    
    def stop_instance(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """
        Stop a cloud compute instance.

        Args:
            instance_id: ID of the instance to stop.
            **kwargs: Platform-specific arguments.

        Returns:
            Instance information dictionary.
        """
        if self.platform == "aws":
            return self._stop_instance_aws(instance_id, **kwargs)
        elif self.platform == "azure":
            return self._stop_instance_azure(instance_id, **kwargs)
        elif self.platform == "gcp":
            return self._stop_instance_gcp(instance_id, **kwargs)
    
    def _stop_instance_aws(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Stop an AWS EC2 instance."""
        try:
            response = self.client.stop_instances(
                InstanceIds=[instance_id]
            )
            
            print(f"Stopped AWS EC2 instance: {instance_id}")
            return {
                "platform": "aws",
                "instance_id": instance_id,
                "status": response['StoppingInstances'][0]['CurrentState']['Name']
            }
        except Exception as e:
            raise RuntimeError(f"Failed to stop AWS EC2 instance: {e}")
    
    def _stop_instance_azure(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Stop an Azure VM."""
        try:
            # Parse instance ID (resource group and VM name)
            resource_group = kwargs.get('resource_group')
            vm_name = instance_id
            
            if not resource_group:
                raise ValueError("resource_group must be provided")
            
            # Stop VM
            self.client.virtual_machines.begin_deallocate(resource_group, vm_name).result()
            
            print(f"Stopped Azure VM: {vm_name}")
            return {
                "platform": "azure",
                "instance_id": vm_name,
                "resource_group": resource_group,
                "status": "stopped"
            }
        except Exception as e:
            raise RuntimeError(f"Failed to stop Azure VM: {e}")
    
    def _stop_instance_gcp(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Stop a Google Compute Engine instance."""
        try:
            # Parse instance ID (zone and instance name)
            project = kwargs.get('project')
            zone = kwargs.get('zone')
            instance_name = instance_id
            
            if not project:
                raise ValueError("project must be provided")
            if not zone:
                raise ValueError("zone must be provided")
            
            # Stop instance
            operation = self.client.stop(
                project=project,
                zone=zone,
                instance=instance_name
            )
            
            # Wait for the operation to complete
            from google.api_core.exceptions import GoogleAPIError
            try:
                operation.result()
                status = "TERMINATED"
            except GoogleAPIError:
                status = "ERROR"
            
            print(f"Stopped Google Compute Engine instance: {instance_name}")
            return {
                "platform": "gcp",
                "instance_id": instance_name,
                "project": project,
                "zone": zone,
                "status": status
            }
        except Exception as e:
            raise RuntimeError(f"Failed to stop Google Compute Engine instance: {e}")
    
    def get_instance_status(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """
        Get the status of a cloud compute instance.

        Args:
            instance_id: ID of the instance to get status for.
            **kwargs: Platform-specific arguments.

        Returns:
            Instance status dictionary.
        """
        if self.platform == "aws":
            return self._get_instance_status_aws(instance_id, **kwargs)
        elif self.platform == "azure":
            return self._get_instance_status_azure(instance_id, **kwargs)
        elif self.platform == "gcp":
            return self._get_instance_status_gcp(instance_id, **kwargs)
    
    def _get_instance_status_aws(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Get the status of an AWS EC2 instance."""
        try:
            response = self.client.describe_instances(
                InstanceIds=[instance_id]
            )
            
            instance = response['Reservations'][0]['Instances'][0]
            
            return {
                "platform": "aws",
                "instance_id": instance_id,
                "status": instance['State']['Name'],
                "instance_type": instance['InstanceType'],
                "public_ip": instance.get('PublicIpAddress'),
                "private_ip": instance.get('PrivateIpAddress')
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get AWS EC2 instance status: {e}")
    
    def _get_instance_status_azure(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Get the status of an Azure VM."""
        try:
            # Parse instance ID (resource group and VM name)
            resource_group = kwargs.get('resource_group')
            vm_name = instance_id
            
            if not resource_group:
                raise ValueError("resource_group must be provided")
            
            # Get VM
            vm = self.client.virtual_machines.get(resource_group, vm_name, expand='instanceView')
            
            # Get status
            status = None
            for status_obj in vm.instance_view.statuses:
                if status_obj.code.startswith('PowerState/'):
                    status = status_obj.code.split('/')[-1]
                    break
            
            # Get network interface
            primary_nic = None
            for nic_ref in vm.network_profile.network_interfaces:
                if nic_ref.primary:
                    primary_nic = nic_ref.id.split('/')[-1]
                    break
            
            return {
                "platform": "azure",
                "instance_id": vm_name,
                "resource_group": resource_group,
                "status": status,
                "vm_size": vm.hardware_profile.vm_size,
                "primary_nic": primary_nic
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get Azure VM status: {e}")
    
    def _get_instance_status_gcp(
        self, instance_id: str, **kwargs
    ) -> Dict:
        """Get the status of a Google Compute Engine instance."""
        try:
            # Parse instance ID (zone and instance name)
            project = kwargs.get('project')
            zone = kwargs.get('zone')
            instance_name = instance_id
            
            if not project:
                raise ValueError("project must be provided")
            if not zone:
                raise ValueError("zone must be provided")
            
            # Get instance
            instance = self.client.get(
                project=project,
                zone=zone,
                instance=instance_name
            )
            
            return {
                "platform": "gcp",
                "instance_id": instance_name,
                "project": project,
                "zone": zone,
                "status": instance.status,
                "machine_type": instance.machine_type.split('/')[-1],
                "public_ip": instance.network_interfaces[0].access_configs[0].nat_ip if instance.network_interfaces and instance.network_interfaces[0].access_configs else None,
                "private_ip": instance.network_interfaces[0].network_ip if instance.network_interfaces else None
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get Google Compute Engine instance status: {e}")


# Example usage
if __name__ == "__main__":
    # Upload a file to S3
    # aws_storage = CloudStorageClient("aws")
    # aws_storage.upload_file("local_file.txt", "remote_file.txt", "my-bucket")
    
    # Start an EC2 instance
    # aws_compute = CloudComputeClient("aws")
    # aws_compute.start_instance("i-1234567890abcdef0")