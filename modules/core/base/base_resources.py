"""Base resources for the system."""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from ..exceptions.base_exceptions import ValidationError


class BaseResource:
    """Base class for all resources."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the resource.
        
        Args:
            resource_type: Type of resource.
            resource_id: Optional resource ID.
            parent_id: Optional parent resource ID.
            metadata: Optional resource metadata.
        """
        self.logger = logging.getLogger(__name__)
        self.resource_type = resource_type
        self.resource_id = resource_id or str(uuid.uuid4())
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary.
        
        Returns:
            Dictionary representation of resource.
        """
        return {
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseResource":
        """Create resource from dictionary.
        
        Args:
            data: Dictionary representation of resource.
            
        Returns:
            Resource instance.
        """
        # Convert ISO format strings to datetime
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
        return cls(**data)
        
    def update(self, data: Dict[str, Any]) -> None:
        """Update resource with new data.
        
        Args:
            data: New resource data.
        """
        # Update fields
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Update timestamp
        self.updated_at = datetime.utcnow()
        
    def validate(self) -> None:
        """Validate resource data.
        
        Raises:
            ValidationError: If validation fails.
        """
        if not self.resource_type:
            raise ValidationError("Resource type cannot be empty")
        if not self.resource_id:
            raise ValidationError("Resource ID cannot be empty")
            
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to resource.
        
        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
        
    def get_metadata(self, key: str) -> Any:
        """Get metadata from resource.
        
        Args:
            key: Metadata key.
            
        Returns:
            Metadata value.
        """
        return self.metadata.get(key)
        
    def remove_metadata(self, key: str) -> None:
        """Remove metadata from resource.
        
        Args:
            key: Metadata key.
        """
        if key in self.metadata:
            del self.metadata[key]
            self.updated_at = datetime.utcnow()
            
    def clear_metadata(self) -> None:
        """Clear all metadata from resource."""
        self.metadata.clear()
        self.updated_at = datetime.utcnow()
        
    def set_parent(self, parent_id: str) -> None:
        """Set parent resource.
        
        Args:
            parent_id: Parent resource ID.
        """
        self.parent_id = parent_id
        self.updated_at = datetime.utcnow()
        
    def remove_parent(self) -> None:
        """Remove parent resource."""
        self.parent_id = None
        self.updated_at = datetime.utcnow()
        
    def is_child_of(self, parent_id: str) -> bool:
        """Check if resource is child of parent.
        
        Args:
            parent_id: Parent resource ID.
            
        Returns:
            True if resource is child of parent.
        """
        return self.parent_id == parent_id
        
    def is_root(self) -> bool:
        """Check if resource is root (no parent).
        
        Returns:
            True if resource is root.
        """
        return self.parent_id is None 