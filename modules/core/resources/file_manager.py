"""File manager for the system."""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import os
import shutil
from pathlib import Path
import hashlib
import mimetypes

from ..exceptions.base_exceptions import (
    ResourceNotFoundError,
    ValidationError,
    ProcessingError
)


class FileManager:
    """Manager for system files."""
    
    def __init__(self, data_dir: str):
        """Initialize the file manager.
        
        Args:
            data_dir: Directory for storing file data.
        """
        self.data_dir = Path(data_dir)
        self._ensure_data_dir()
        
        # File type mappings
        self._file_types = {
            "audio": ["wav", "mp3", "ogg", "flac"],
            "midi": ["mid", "midi"],
            "text": ["txt", "json"]
        }
        
        # Initialize mime types
        mimetypes.init()
        
    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_resource_dir(self, resource_type: str, resource_id: str) -> Path:
        """Get directory for resource files.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            
        Returns:
            Path to resource directory.
        """
        return self.data_dir / resource_type / resource_id
        
    def _get_file_path(
        self,
        resource_type: str,
        resource_id: str,
        file_name: str
    ) -> Path:
        """Get path for resource file.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            file_name: Name of file.
            
        Returns:
            Path to file.
        """
        return self._get_resource_dir(resource_type, resource_id) / file_name
        
    def _validate_file_type(
        self,
        file_type: str,
        file_format: str
    ) -> None:
        """Validate file type and format.
        
        Args:
            file_type: Type of file (e.g., "audio", "midi").
            file_format: File format (e.g., "wav", "mid").
            
        Raises:
            ValidationError: If file type or format is invalid.
        """
        if file_type not in self._file_types:
            raise ValidationError(f"Invalid file type: {file_type}")
            
        if file_format not in self._file_types[file_type]:
            raise ValidationError(
                f"Invalid format for {file_type}: {file_format}"
            )
            
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash.
        
        Args:
            file_path: Path to file.
            
        Returns:
            File hash.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Dictionary of file information.
        """
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return {
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "mime_type": mime_type,
            "hash": self._get_file_hash(file_path)
        }
        
    def save_file(
        self,
        resource_type: str,
        resource_id: str,
        file_path: Union[str, Path],
        file_type: str,
        file_format: str
    ) -> Dict[str, Any]:
        """Save a file for a resource.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            file_path: Path to source file.
            file_type: Type of file (e.g., "audio", "midi").
            file_format: File format (e.g., "wav", "mid").
            
        Returns:
            Dictionary of file information.
            
        Raises:
            ValidationError: If file type or format is invalid.
            ProcessingError: If file operation fails.
        """
        self._validate_file_type(file_type, file_format)
        
        source_path = Path(file_path)
        if not source_path.exists():
            raise ProcessingError(f"Source file not found: {file_path}")
            
        try:
            # Create resource directory
            resource_dir = self._get_resource_dir(resource_type, resource_id)
            resource_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            target_path = self._get_file_path(
                resource_type,
                resource_id,
                f"file.{file_format}"
            )
            shutil.copy2(source_path, target_path)
            
            # Get file information
            return self._get_file_info(target_path)
            
        except Exception as e:
            raise ProcessingError(f"Error saving file: {e}")
            
    def get_file(
        self,
        resource_type: str,
        resource_id: str,
        file_name: str
    ) -> Path:
        """Get path to resource file.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            file_name: Name of file.
            
        Returns:
            Path to file.
            
        Raises:
            ResourceNotFoundError: If file not found.
        """
        file_path = self._get_file_path(resource_type, resource_id, file_name)
        if not file_path.exists():
            raise ResourceNotFoundError(
                f"File not found: {resource_type}/{resource_id}/{file_name}"
            )
        return file_path
        
    def get_file_info(
        self,
        resource_type: str,
        resource_id: str,
        file_name: str
    ) -> Dict[str, Any]:
        """Get information about a resource file.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            file_name: Name of file.
            
        Returns:
            Dictionary of file information.
            
        Raises:
            ResourceNotFoundError: If file not found.
        """
        file_path = self.get_file(resource_type, resource_id, file_name)
        return self._get_file_info(file_path)
        
    def delete_file(
        self,
        resource_type: str,
        resource_id: str,
        file_name: str
    ) -> None:
        """Delete a resource file.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            file_name: Name of file.
            
        Raises:
            ResourceNotFoundError: If file not found.
            ProcessingError: If deletion fails.
        """
        file_path = self.get_file(resource_type, resource_id, file_name)
        
        try:
            file_path.unlink()
        except Exception as e:
            raise ProcessingError(f"Error deleting file: {e}")
            
    def delete_resource_files(
        self,
        resource_type: str,
        resource_id: str
    ) -> None:
        """Delete all files for a resource.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            
        Raises:
            ProcessingError: If deletion fails.
        """
        resource_dir = self._get_resource_dir(resource_type, resource_id)
        if resource_dir.exists():
            try:
                shutil.rmtree(resource_dir)
            except Exception as e:
                raise ProcessingError(f"Error deleting resource files: {e}")
                
    def list_resource_files(
        self,
        resource_type: str,
        resource_id: str
    ) -> List[Dict[str, Any]]:
        """List all files for a resource.
        
        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            
        Returns:
            List of file information dictionaries.
        """
        resource_dir = self._get_resource_dir(resource_type, resource_id)
        if not resource_dir.exists():
            return []
            
        files = []
        for file_path in resource_dir.iterdir():
            if file_path.is_file():
                files.append(self._get_file_info(file_path))
        return files
        
    def copy_file(
        self,
        source_type: str,
        source_id: str,
        source_name: str,
        target_type: str,
        target_id: str,
        target_name: str
    ) -> Dict[str, Any]:
        """Copy a file between resources.
        
        Args:
            source_type: Type of source resource.
            source_id: Source resource ID.
            source_name: Name of source file.
            target_type: Type of target resource.
            target_id: Target resource ID.
            target_name: Name of target file.
            
        Returns:
            Dictionary of target file information.
            
        Raises:
            ResourceNotFoundError: If source file not found.
            ProcessingError: If copy operation fails.
        """
        source_path = self.get_file(source_type, source_id, source_name)
        
        try:
            # Create target directory
            target_dir = self._get_resource_dir(target_type, target_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            target_path = self._get_file_path(
                target_type,
                target_id,
                target_name
            )
            shutil.copy2(source_path, target_path)
            
            # Get target file information
            return self._get_file_info(target_path)
            
        except Exception as e:
            raise ProcessingError(f"Error copying file: {e}")
            
    def move_file(
        self,
        source_type: str,
        source_id: str,
        source_name: str,
        target_type: str,
        target_id: str,
        target_name: str
    ) -> Dict[str, Any]:
        """Move a file between resources.
        
        Args:
            source_type: Type of source resource.
            source_id: Source resource ID.
            source_name: Name of source file.
            target_type: Type of target resource.
            target_id: Target resource ID.
            target_name: Name of target file.
            
        Returns:
            Dictionary of target file information.
            
        Raises:
            ResourceNotFoundError: If source file not found.
            ProcessingError: If move operation fails.
        """
        source_path = self.get_file(source_type, source_id, source_name)
        
        try:
            # Create target directory
            target_dir = self._get_resource_dir(target_type, target_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Move file
            target_path = self._get_file_path(
                target_type,
                target_id,
                target_name
            )
            shutil.move(source_path, target_path)
            
            # Get target file information
            return self._get_file_info(target_path)
            
        except Exception as e:
            raise ProcessingError(f"Error moving file: {e}") 