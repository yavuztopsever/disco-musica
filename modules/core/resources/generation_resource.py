"""Generation resource for the system."""
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..base.base_resources import BaseResource
from ..exceptions.base_exceptions import ValidationError


class GenerationResource(BaseResource):
    """Resource for a music generation output."""
    
    def __init__(
        self,
        generation_type: str,
        track_id: str,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the generation resource.
        
        Args:
            generation_type: Type of generation (e.g., "text_to_music", "midi_variation").
            track_id: Parent track ID.
            resource_id: Optional resource ID.
            metadata: Optional resource metadata.
        """
        super().__init__(
            resource_type="generation",
            resource_id=resource_id,
            parent_id=track_id,
            metadata=metadata
        )
        self.generation_type = generation_type
        self.status = "active"
        self.model_id: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.duration: float = 0.0
        self.file_path: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert generation to dictionary.
        
        Returns:
            Dictionary representation of generation.
        """
        data = super().to_dict()
        data.update({
            "generation_type": self.generation_type,
            "status": self.status,
            "model_id": self.model_id,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "duration": self.duration,
            "file_path": self.file_path
        })
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResource":
        """Create generation from dictionary.
        
        Args:
            data: Dictionary representation of generation.
            
        Returns:
            GenerationResource instance.
        """
        # Extract base resource fields
        resource_fields = {
            "resource_id": data.pop("resource_id", None),
            "metadata": data.pop("metadata", None)
        }
        
        # Create generation
        generation = cls(
            generation_type=data["generation_type"],
            track_id=data["parent_id"],
            **resource_fields
        )
        
        # Set additional fields
        generation.status = data.get("status", "active")
        generation.model_id = data.get("model_id")
        generation.config = data.get("config")
        generation.inputs = data.get("inputs", {})
        generation.outputs = data.get("outputs", {})
        generation.metrics = data.get("metrics", {})
        generation.duration = data.get("duration", 0.0)
        generation.file_path = data.get("file_path")
        
        # Set timestamps
        if "created_at" in data:
            generation.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            generation.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return generation
        
    def validate(self) -> None:
        """Validate generation data.
        
        Raises:
            ValidationError: If validation fails.
        """
        super().validate()
        
        if not self.generation_type:
            raise ValidationError("Generation type cannot be empty")
        if not self.parent_id:
            raise ValidationError("Generation must belong to a track")
        if self.generation_type not in [
            "text_to_music",
            "midi_variation",
            "audio_variation"
        ]:
            raise ValidationError(
                f"Unsupported generation type: {self.generation_type}"
            )
        if self.status not in ["active", "archived", "deleted"]:
            raise ValidationError(
                f"Invalid generation status: {self.status}"
            )
        if self.duration < 0:
            raise ValidationError("Duration cannot be negative")
            
    def set_model(self, model_id: str) -> None:
        """Set model used for generation.
        
        Args:
            model_id: Model resource ID.
        """
        self.model_id = model_id
        self.updated_at = datetime.utcnow()
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set generation configuration.
        
        Args:
            config: Generation configuration dictionary.
        """
        self.config = config
        self.updated_at = datetime.utcnow()
        
    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """Set generation inputs.
        
        Args:
            inputs: Dictionary of input values.
        """
        self.inputs = inputs
        self.updated_at = datetime.utcnow()
        
    def set_outputs(self, outputs: Dict[str, Any]) -> None:
        """Set generation outputs.
        
        Args:
            outputs: Dictionary of output values.
        """
        self.outputs = outputs
        self.updated_at = datetime.utcnow()
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update generation metrics.
        
        Args:
            metrics: Dictionary of metric names and values.
        """
        self.metrics.update(metrics)
        self.updated_at = datetime.utcnow()
        
    def set_file(
        self,
        file_path: str,
        duration: float
    ) -> None:
        """Set generation file information.
        
        Args:
            file_path: Path to generated file.
            duration: Generation duration in seconds.
        """
        self.file_path = file_path
        self.duration = duration
        self.updated_at = datetime.utcnow()
        
    def archive(self) -> None:
        """Archive the generation."""
        self.status = "archived"
        self.updated_at = datetime.utcnow()
        
    def delete(self) -> None:
        """Delete the generation."""
        self.status = "deleted"
        self.updated_at = datetime.utcnow()
        
    def is_active(self) -> bool:
        """Check if generation is active.
        
        Returns:
            True if generation is active.
        """
        return self.status == "active"
        
    def is_archived(self) -> bool:
        """Check if generation is archived.
        
        Returns:
            True if generation is archived.
        """
        return self.status == "archived"
        
    def is_deleted(self) -> bool:
        """Check if generation is deleted.
        
        Returns:
            True if generation is deleted.
        """
        return self.status == "deleted"
        
    def has_file(self) -> bool:
        """Check if generation has an output file.
        
        Returns:
            True if generation has an output file.
        """
        return self.file_path is not None 