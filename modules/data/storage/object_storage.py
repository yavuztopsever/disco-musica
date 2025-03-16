"""Object storage service for handling file storage operations."""

import logging
import os
import shutil
from typing import Dict, Any, List, Optional, Union, BinaryIO
from datetime import datetime
from pathlib import Path
import hashlib
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource

logger = logging.getLogger(__name__)

class ObjectMetadata(BaseModel):
    """Model for object metadata."""
    object_id: str
    filename: str
    content_type: str
    size: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    checksum: str

class ObjectStorage:
    """Service for managing file storage operations."""
    
    def __init__(self, base_path: str):
        """Initialize the object storage.
        
        Args:
            base_path: Base directory for file storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, ObjectMetadata] = {}
        self.logger = logging.getLogger(__name__)
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            SHA-256 checksum.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def _get_object_path(self, object_id: str) -> Path:
        """Get the full path for an object.
        
        Args:
            object_id: ID of the object.
            
        Returns:
            Full path to the object.
        """
        return self.base_path / object_id
        
    async def store_object(
        self,
        file: Union[str, Path, BinaryIO],
        filename: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        object_id: Optional[str] = None
    ) -> str:
        """Store a file in the object storage.
        
        Args:
            file: File to store (path or file-like object).
            filename: Original filename.
            content_type: MIME type of the file.
            metadata: Optional metadata.
            object_id: Optional object ID (generated if not provided).
            
        Returns:
            ID of the stored object.
        """
        # Generate object ID if not provided
        if not object_id:
            object_id = hashlib.sha256(
                f"{filename}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
            
        # Get object path
        object_path = self._get_object_path(object_id)
        
        # Copy file to storage
        if isinstance(file, (str, Path)):
            shutil.copy2(file, object_path)
        else:
            with open(object_path, "wb") as f:
                shutil.copyfileobj(file, f)
                
        # Calculate checksum
        checksum = self._calculate_checksum(object_path)
        
        # Get file size
        size = object_path.stat().st_size
        
        # Create metadata
        now = datetime.utcnow()
        object_metadata = ObjectMetadata(
            object_id=object_id,
            filename=filename,
            content_type=content_type,
            size=size,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            checksum=checksum
        )
        
        self.metadata[object_id] = object_metadata
        return object_id
        
    async def get_object(
        self,
        object_id: str,
        verify_checksum: bool = True
    ) -> Optional[Path]:
        """Get a file from the object storage.
        
        Args:
            object_id: ID of the object.
            verify_checksum: Whether to verify the checksum.
            
        Returns:
            Path to the file if found, None otherwise.
        """
        if object_id not in self.metadata:
            return None
            
        object_path = self._get_object_path(object_id)
        if not object_path.exists():
            return None
            
        # Verify checksum if requested
        if verify_checksum:
            current_checksum = self._calculate_checksum(object_path)
            if current_checksum != self.metadata[object_id].checksum:
                raise DiscoMusicaError(f"Checksum mismatch for object {object_id}")
                
        return object_path
        
    async def delete_object(self, object_id: str) -> bool:
        """Delete a file from the object storage.
        
        Args:
            object_id: ID of the object.
            
        Returns:
            True if deleted, False if not found.
        """
        if object_id not in self.metadata:
            return False
            
        object_path = self._get_object_path(object_id)
        if object_path.exists():
            object_path.unlink()
            
        del self.metadata[object_id]
        return True
        
    async def get_metadata(self, object_id: str) -> Optional[ObjectMetadata]:
        """Get metadata for an object.
        
        Args:
            object_id: ID of the object.
            
        Returns:
            Object metadata if found, None otherwise.
        """
        return self.metadata.get(object_id)
        
    async def update_metadata(
        self,
        object_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for an object.
        
        Args:
            object_id: ID of the object.
            metadata: New metadata.
            
        Returns:
            True if updated, False if not found.
        """
        if object_id not in self.metadata:
            return False
            
        object_metadata = self.metadata[object_id]
        object_metadata.metadata.update(metadata)
        object_metadata.updated_at = datetime.utcnow()
        return True
        
    async def list_objects(
        self,
        prefix: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List objects in the storage.
        
        Args:
            prefix: Optional prefix filter.
            metadata_filter: Optional metadata filter.
            
        Returns:
            List of object summaries.
        """
        objects = []
        for object_id, metadata in self.metadata.items():
            # Apply prefix filter if provided
            if prefix and not object_id.startswith(prefix):
                continue
                
            # Apply metadata filter if provided
            if metadata_filter:
                if not all(metadata.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                    
            objects.append({
                "object_id": object_id,
                "filename": metadata.filename,
                "content_type": metadata.content_type,
                "size": metadata.size,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "metadata": metadata.metadata
            })
            
        return objects 