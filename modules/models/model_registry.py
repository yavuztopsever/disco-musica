"""Model Registry for managing AI models."""

import os
import json
from typing import Dict, List, Optional, Any, Type, Union
from pathlib import Path
import logging
import importlib
from datetime import datetime

from ..core.config import config
from ..core.models.base_model import BaseModel
from ..core.models.text_to_music_model import TextToMusicModel
from ..core.models.audio_to_music_model import AudioToMusicModel
from ..core.models.midi_to_audio_model import MIDIToAudioModel
from ..exceptions.base_exceptions import (
    ModelNotFoundError,
    ValidationError,
    ProcessingError
)


class ModelRegistry:
    """Registry for managing AI model instances and configurations."""
    
    def __init__(self, models_dir: Union[str, Path]):
        """Initialize the model registry.
        
        Args:
            models_dir: Directory for model files and configurations.
        """
        self.models_dir = Path(models_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Model type mappings
        self.model_types = {
            "text_to_music": TextToMusicModel,
            "audio_to_music": AudioToMusicModel,
            "midi_to_audio": MIDIToAudioModel
        }
        
        # Model instance cache
        self.model_instances: Dict[str, BaseModel] = {}
        
        # Model configurations
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model configurations
        self._load_model_configs()
        
    def _load_model_configs(self) -> None:
        """Load model configurations from disk."""
        config_file = self.models_dir / "models.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    self.model_configs = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading model configurations: {e}")
                
    def _save_model_configs(self) -> None:
        """Save model configurations to disk."""
        try:
            config_file = self.models_dir / "models.json"
            with open(config_file, "w") as f:
                json.dump(self.model_configs, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving model configurations: {e}")
            
    def register_model(
        self,
        model_type: str,
        version: str,
        config: Dict[str, Any]
    ) -> None:
        """Register a new model configuration.
        
        Args:
            model_type: Type of model.
            version: Model version.
            config: Model configuration.
            
        Raises:
            ValidationError: If model type is invalid.
            ProcessingError: If registration fails.
        """
        try:
            # Validate model type
            if model_type not in self.model_types:
                raise ValidationError(f"Invalid model type: {model_type}")
                
            # Create model ID
            model_id = f"{model_type}-{version}"
            
            # Add configuration
            self.model_configs[model_id] = {
                "type": model_type,
                "version": version,
                "config": config,
                "registered_at": datetime.utcnow().isoformat()
            }
            
            # Save configurations
            self._save_model_configs()
            
        except Exception as e:
            raise ProcessingError(f"Error registering model: {e}")
            
    def get_model(
        self,
        model_type: str,
        version: str,
        **kwargs
    ) -> BaseModel:
        """Get or create a model instance.
        
        Args:
            model_type: Type of model.
            version: Model version.
            **kwargs: Additional model parameters.
            
        Returns:
            Model instance.
            
        Raises:
            ModelNotFoundError: If model not found.
            ProcessingError: If model creation fails.
        """
        # Create model ID
        model_id = f"{model_type}-{version}"
        
        # Check if model is registered
        if model_id not in self.model_configs:
            raise ModelNotFoundError(f"Model not found: {model_id}")
            
        try:
            # Return cached instance if available
            if model_id in self.model_instances:
                return self.model_instances[model_id]
                
            # Create new instance
            model_class = self.model_types[model_type]
            model_config = self.model_configs[model_id]["config"]
            
            # Create model instance
            model = model_class(
                version=version,
                config={**model_config, **kwargs}
            )
            
            # Cache instance
            self.model_instances[model_id] = model
            
            return model
            
        except Exception as e:
            raise ProcessingError(f"Error creating model instance: {e}")
            
    def unload_model(self, model_type: str, version: str) -> None:
        """Unload a model instance from memory.
        
        Args:
            model_type: Type of model.
            version: Model version.
            
        Raises:
            ModelNotFoundError: If model not found.
        """
        # Create model ID
        model_id = f"{model_type}-{version}"
        
        # Check if model is loaded
        if model_id not in self.model_instances:
            raise ModelNotFoundError(f"Model not loaded: {model_id}")
            
        try:
            # Get model instance
            model = self.model_instances[model_id]
            
            # Clean up model resources
            model.cleanup()
            
            # Remove from cache
            del self.model_instances[model_id]
            
        except Exception as e:
            self.logger.error(f"Error unloading model {model_id}: {e}")
            
    def unload_all_models(self) -> None:
        """Unload all model instances from memory."""
        for model_id in list(self.model_instances.keys()):
            try:
                model_type, version = model_id.split("-", 1)
                self.unload_model(model_type, version)
            except Exception as e:
                self.logger.error(f"Error unloading model {model_id}: {e}")
                
    def get_model_info(
        self,
        model_type: str,
        version: str
    ) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model_type: Type of model.
            version: Model version.
            
        Returns:
            Model information dictionary.
            
        Raises:
            ModelNotFoundError: If model not found.
        """
        # Create model ID
        model_id = f"{model_type}-{version}"
        
        # Check if model is registered
        if model_id not in self.model_configs:
            raise ModelNotFoundError(f"Model not found: {model_id}")
            
        # Get model configuration
        model_config = self.model_configs[model_id]
        
        # Check if model is loaded
        is_loaded = model_id in self.model_instances
        
        return {
            "id": model_id,
            "type": model_type,
            "version": version,
            "config": model_config["config"],
            "registered_at": model_config["registered_at"],
            "is_loaded": is_loaded
        }
        
    def list_models(
        self,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List registered models.
        
        Args:
            model_type: Optional type filter.
            
        Returns:
            List of model information dictionaries.
        """
        models = []
        
        for model_id, config in self.model_configs.items():
            if model_type and config["type"] != model_type:
                continue
                
            models.append({
                "id": model_id,
                "type": config["type"],
                "version": config["version"],
                "config": config["config"],
                "registered_at": config["registered_at"],
                "is_loaded": model_id in self.model_instances
            })
            
        return models
        
    def get_default_model(self, model_type: str) -> Optional[str]:
        """Get default model version for a type.
        
        Args:
            model_type: Type of model.
            
        Returns:
            Default model version or None if not found.
        """
        # Get models of specified type
        models = self.list_models(model_type)
        
        if not models:
            return None
            
        # Return latest version
        latest = max(models, key=lambda m: m["registered_at"])
        return latest["version"]
        
    def validate_model_exists(
        self,
        model_type: str,
        version: str
    ) -> None:
        """Validate that a model exists.
        
        Args:
            model_type: Type of model.
            version: Model version.
            
        Raises:
            ModelNotFoundError: If model not found.
        """
        model_id = f"{model_type}-{version}"
        if model_id not in self.model_configs:
            raise ModelNotFoundError(f"Model not found: {model_id}")
            
    def get_model_path(
        self,
        model_type: str,
        version: str
    ) -> Path:
        """Get path to model files.
        
        Args:
            model_type: Type of model.
            version: Model version.
            
        Returns:
            Path to model directory.
            
        Raises:
            ModelNotFoundError: If model not found.
        """
        # Validate model exists
        self.validate_model_exists(model_type, version)
        
        return self.models_dir / model_type / version


# Create global model registry instance
model_registry = ModelRegistry(
    models_dir=Path(config.get("paths", "models_dir", "models"))
)


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance.
    
    Returns:
        ModelRegistry instance.
    """
    global model_registry
    return model_registry 