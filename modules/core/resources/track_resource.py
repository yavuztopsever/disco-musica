"""Track resource for the system."""
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..base.base_resources import BaseResource
from ..exceptions.base_exceptions import ValidationError


class TrackResource(BaseResource):
    """Resource for a musical track."""
    
    def __init__(
        self,
        name: str,
        project_id: str,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the track resource.
        
        Args:
            name: Track name.
            project_id: Parent project ID.
            resource_id: Optional resource ID.
            metadata: Optional resource metadata.
        """
        super().__init__(
            resource_type="track",
            resource_id=resource_id,
            parent_id=project_id,
            metadata=metadata
        )
        self.name = name
        self.status = "active"
        self.description: Optional[str] = None
        self.duration: float = 0.0
        self.tempo: float = 120.0
        self.key: Optional[str] = None
        self.time_signature: str = "4/4"
        self.file_path: Optional[str] = None
        self.file_format: Optional[str] = None
        self.file_size: int = 0
        self.generations: List[str] = []
        self.tags: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary.
        
        Returns:
            Dictionary representation of track.
        """
        data = super().to_dict()
        data.update({
            "name": self.name,
            "status": self.status,
            "description": self.description,
            "duration": self.duration,
            "tempo": self.tempo,
            "key": self.key,
            "time_signature": self.time_signature,
            "file_path": self.file_path,
            "file_format": self.file_format,
            "file_size": self.file_size,
            "generations": self.generations,
            "tags": self.tags
        })
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackResource":
        """Create track from dictionary.
        
        Args:
            data: Dictionary representation of track.
            
        Returns:
            TrackResource instance.
        """
        # Extract base resource fields
        resource_fields = {
            "resource_id": data.pop("resource_id", None),
            "metadata": data.pop("metadata", None)
        }
        
        # Create track
        track = cls(
            name=data["name"],
            project_id=data["parent_id"],
            **resource_fields
        )
        
        # Set additional fields
        track.status = data.get("status", "active")
        track.description = data.get("description")
        track.duration = data.get("duration", 0.0)
        track.tempo = data.get("tempo", 120.0)
        track.key = data.get("key")
        track.time_signature = data.get("time_signature", "4/4")
        track.file_path = data.get("file_path")
        track.file_format = data.get("file_format")
        track.file_size = data.get("file_size", 0)
        track.generations = data.get("generations", [])
        track.tags = data.get("tags", [])
        
        # Set timestamps
        if "created_at" in data:
            track.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            track.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return track
        
    def validate(self) -> None:
        """Validate track data.
        
        Raises:
            ValidationError: If validation fails.
        """
        super().validate()
        
        if not self.name:
            raise ValidationError("Track name cannot be empty")
        if not self.parent_id:
            raise ValidationError("Track must belong to a project")
        if self.status not in ["active", "archived", "deleted"]:
            raise ValidationError(
                f"Invalid track status: {self.status}"
            )
        if self.duration < 0:
            raise ValidationError("Duration cannot be negative")
        if self.tempo <= 0:
            raise ValidationError("Tempo must be positive")
        if self.key and self.key not in [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
        ]:
            raise ValidationError(f"Invalid key: {self.key}")
        if self.time_signature not in ["4/4", "3/4", "6/8"]:
            raise ValidationError(
                f"Unsupported time signature: {self.time_signature}"
            )
        if self.file_size < 0:
            raise ValidationError("File size cannot be negative")
            
    def set_description(self, description: str) -> None:
        """Set track description.
        
        Args:
            description: Track description.
        """
        self.description = description
        self.updated_at = datetime.utcnow()
        
    def set_metadata(
        self,
        duration: float,
        tempo: float,
        key: Optional[str] = None,
        time_signature: str = "4/4"
    ) -> None:
        """Set track musical metadata.
        
        Args:
            duration: Track duration in seconds.
            tempo: Track tempo in BPM.
            key: Optional musical key.
            time_signature: Musical time signature.
        """
        self.duration = duration
        self.tempo = tempo
        self.key = key
        self.time_signature = time_signature
        self.updated_at = datetime.utcnow()
        
    def set_file(
        self,
        file_path: str,
        file_format: str,
        file_size: int
    ) -> None:
        """Set track file information.
        
        Args:
            file_path: Path to track file.
            file_format: File format (e.g., "wav", "mp3", "midi").
            file_size: File size in bytes.
        """
        self.file_path = file_path
        self.file_format = file_format
        self.file_size = file_size
        self.updated_at = datetime.utcnow()
        
    def add_generation(self, generation_id: str) -> None:
        """Add a generation to the track.
        
        Args:
            generation_id: Generation resource ID.
        """
        if generation_id not in self.generations:
            self.generations.append(generation_id)
            self.updated_at = datetime.utcnow()
            
    def remove_generation(self, generation_id: str) -> None:
        """Remove a generation from the track.
        
        Args:
            generation_id: Generation resource ID.
        """
        if generation_id in self.generations:
            self.generations.remove(generation_id)
            self.updated_at = datetime.utcnow()
            
    def add_tag(self, tag: str) -> None:
        """Add a tag to the track.
        
        Args:
            tag: Tag to add.
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
            
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the track.
        
        Args:
            tag: Tag to remove.
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
            
    def archive(self) -> None:
        """Archive the track."""
        self.status = "archived"
        self.updated_at = datetime.utcnow()
        
    def delete(self) -> None:
        """Delete the track."""
        self.status = "deleted"
        self.updated_at = datetime.utcnow()
        
    def is_active(self) -> bool:
        """Check if track is active.
        
        Returns:
            True if track is active.
        """
        return self.status == "active"
        
    def is_archived(self) -> bool:
        """Check if track is archived.
        
        Returns:
            True if track is archived.
        """
        return self.status == "archived"
        
    def is_deleted(self) -> bool:
        """Check if track is deleted.
        
        Returns:
            True if track is deleted.
        """
        return self.status == "deleted"
        
    def has_file(self) -> bool:
        """Check if track has an associated file.
        
        Returns:
            True if track has an associated file.
        """
        return self.file_path is not None 