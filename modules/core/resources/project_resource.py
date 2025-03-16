"""Project resource for the system."""
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..base.base_resources import BaseResource
from ..exceptions.base_exceptions import ValidationError


class ProjectResource(BaseResource):
    """Resource for a project."""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the project resource.
        
        Args:
            name: Project name.
            description: Optional project description.
            resource_id: Optional resource ID.
            metadata: Optional resource metadata.
        """
        super().__init__(
            resource_type="project",
            resource_id=resource_id,
            metadata=metadata
        )
        self.name = name
        self.description = description
        self.status = "active"
        self.tracks: List[str] = []
        self.models: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary.
        
        Returns:
            Dictionary representation of project.
        """
        data = super().to_dict()
        data.update({
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "tracks": self.tracks,
            "models": self.models
        })
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectResource":
        """Create project from dictionary.
        
        Args:
            data: Dictionary representation of project.
            
        Returns:
            ProjectResource instance.
        """
        # Extract base resource fields
        resource_fields = {
            "resource_id": data.pop("resource_id", None),
            "metadata": data.pop("metadata", None)
        }
        
        # Create project
        project = cls(
            name=data["name"],
            description=data.get("description"),
            **resource_fields
        )
        
        # Set additional fields
        project.status = data.get("status", "active")
        project.tracks = data.get("tracks", [])
        project.models = data.get("models", [])
        
        # Set timestamps
        if "created_at" in data:
            project.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            project.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return project
        
    def validate(self) -> None:
        """Validate project data.
        
        Raises:
            ValidationError: If validation fails.
        """
        super().validate()
        
        if not self.name:
            raise ValidationError("Project name cannot be empty")
        if self.status not in ["active", "archived", "deleted"]:
            raise ValidationError(
                f"Invalid project status: {self.status}"
            )
            
    def add_track(self, track_id: str) -> None:
        """Add track to project.
        
        Args:
            track_id: Track resource ID.
        """
        if track_id not in self.tracks:
            self.tracks.append(track_id)
            self.updated_at = datetime.utcnow()
            
    def remove_track(self, track_id: str) -> None:
        """Remove track from project.
        
        Args:
            track_id: Track resource ID.
        """
        if track_id in self.tracks:
            self.tracks.remove(track_id)
            self.updated_at = datetime.utcnow()
            
    def add_model(self, model_id: str) -> None:
        """Add model to project.
        
        Args:
            model_id: Model resource ID.
        """
        if model_id not in self.models:
            self.models.append(model_id)
            self.updated_at = datetime.utcnow()
            
    def remove_model(self, model_id: str) -> None:
        """Remove model from project.
        
        Args:
            model_id: Model resource ID.
        """
        if model_id in self.models:
            self.models.remove(model_id)
            self.updated_at = datetime.utcnow()
            
    def archive(self) -> None:
        """Archive the project."""
        self.status = "archived"
        self.updated_at = datetime.utcnow()
        
    def delete(self) -> None:
        """Delete the project."""
        self.status = "deleted"
        self.updated_at = datetime.utcnow()
        
    def is_active(self) -> bool:
        """Check if project is active.
        
        Returns:
            True if project is active.
        """
        return self.status == "active"
        
    def is_archived(self) -> bool:
        """Check if project is archived.
        
        Returns:
            True if project is archived.
        """
        return self.status == "archived"
        
    def is_deleted(self) -> bool:
        """Check if project is deleted.
        
        Returns:
            True if project is deleted.
        """
        return self.status == "deleted" 