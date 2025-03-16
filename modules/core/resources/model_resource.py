"""Model resource for the system."""
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..base.base_resources import BaseResource
from ..exceptions.base_exceptions import ValidationError


class ModelResource(BaseResource):
    """Resource for an AI model."""
    
    def __init__(
        self,
        name: str,
        model_type: str,
        version: str,
        project_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the model resource.
        
        Args:
            name: Model name.
            model_type: Type of model (e.g., "audio", "midi", "text").
            version: Model version.
            project_id: Optional parent project ID.
            resource_id: Optional resource ID.
            metadata: Optional resource metadata.
        """
        super().__init__(
            resource_type="model",
            resource_id=resource_id,
            parent_id=project_id,
            metadata=metadata
        )
        self.name = name
        self.model_type = model_type
        self.version = version
        self.status = "active"
        self.weights_path: Optional[str] = None
        self.training_config: Optional[Dict[str, Any]] = None
        self.metrics: Dict[str, float] = {}
        self.last_trained: Optional[datetime] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary.
        
        Returns:
            Dictionary representation of model.
        """
        data = super().to_dict()
        data.update({
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "status": self.status,
            "weights_path": self.weights_path,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None
        })
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResource":
        """Create model from dictionary.
        
        Args:
            data: Dictionary representation of model.
            
        Returns:
            ModelResource instance.
        """
        # Extract base resource fields
        resource_fields = {
            "resource_id": data.pop("resource_id", None),
            "metadata": data.pop("metadata", None)
        }
        
        # Create model
        model = cls(
            name=data["name"],
            model_type=data["model_type"],
            version=data["version"],
            project_id=data.get("parent_id"),
            **resource_fields
        )
        
        # Set additional fields
        model.status = data.get("status", "active")
        model.weights_path = data.get("weights_path")
        model.training_config = data.get("training_config")
        model.metrics = data.get("metrics", {})
        if data.get("last_trained"):
            model.last_trained = datetime.fromisoformat(data["last_trained"])
            
        # Set timestamps
        if "created_at" in data:
            model.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            model.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return model
        
    def validate(self) -> None:
        """Validate model data.
        
        Raises:
            ValidationError: If validation fails.
        """
        super().validate()
        
        if not self.name:
            raise ValidationError("Model name cannot be empty")
        if not self.model_type:
            raise ValidationError("Model type cannot be empty")
        if not self.version:
            raise ValidationError("Model version cannot be empty")
        if self.model_type not in ["audio", "midi", "text"]:
            raise ValidationError(
                f"Unsupported model type: {self.model_type}"
            )
        if self.status not in ["active", "archived", "deleted"]:
            raise ValidationError(
                f"Invalid model status: {self.status}"
            )
            
    def set_weights_path(self, path: str) -> None:
        """Set path to model weights.
        
        Args:
            path: Path to model weights file.
        """
        self.weights_path = path
        self.updated_at = datetime.utcnow()
        
    def set_training_config(self, config: Dict[str, Any]) -> None:
        """Set training configuration.
        
        Args:
            config: Training configuration dictionary.
        """
        self.training_config = config
        self.updated_at = datetime.utcnow()
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update model metrics.
        
        Args:
            metrics: Dictionary of metric names and values.
        """
        self.metrics.update(metrics)
        self.updated_at = datetime.utcnow()
        
    def set_last_trained(self, timestamp: datetime) -> None:
        """Set last training timestamp.
        
        Args:
            timestamp: Training completion timestamp.
        """
        self.last_trained = timestamp
        self.updated_at = datetime.utcnow()
        
    def archive(self) -> None:
        """Archive the model."""
        self.status = "archived"
        self.updated_at = datetime.utcnow()
        
    def delete(self) -> None:
        """Delete the model."""
        self.status = "deleted"
        self.updated_at = datetime.utcnow()
        
    def is_active(self) -> bool:
        """Check if model is active.
        
        Returns:
            True if model is active.
        """
        return self.status == "active"
        
    def is_archived(self) -> bool:
        """Check if model is archived.
        
        Returns:
            True if model is archived.
        """
        return self.status == "archived"
        
    def is_deleted(self) -> bool:
        """Check if model is deleted.
        
        Returns:
            True if model is deleted.
        """
        return self.status == "deleted"
        
    def has_weights(self) -> bool:
        """Check if model has weights file.
        
        Returns:
            True if model has weights file.
        """
        return self.weights_path is not None 