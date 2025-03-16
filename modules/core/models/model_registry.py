"""Model Registry for managing AI model instances."""

import json
from typing import Dict, List, Optional, Union, Any, Type
from pathlib import Path
import logging
from datetime import datetime
import torch
import gc
from collections import OrderedDict

from .base_model import BaseModel
from .text_to_music_model import TextToMusicModel
from .audio_to_music_model import AudioToMusicModel
from .midi_to_audio_model import MIDIToAudioModel
from ..exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)


class ModelRegistry:
    """Registry for managing AI model instances.
    
    This class provides functionality for managing model instances,
    including loading/unloading models, caching, and configuration
    management.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_cached_models: int = 3,
        device: Optional[str] = None
    ):
        """Initialize the model registry.
        
        Args:
            cache_dir: Directory for caching model files.
            max_cached_models: Maximum number of models to keep in memory.
            device: Device to load models on (defaults to GPU if available).
        """
        self.cache_dir = Path(cache_dir)
        self.max_cached_models = max_cached_models
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Model type mappings
        self.model_types = {
            "text_to_music": TextToMusicModel,
            "audio_to_music": AudioToMusicModel,
            "midi_to_audio": MIDIToAudioModel
        }
        
        # Model instance cache (OrderedDict for LRU caching)
        self.model_cache: OrderedDict[str, BaseModel] = OrderedDict()
        
        # Model configurations
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model configurations
        self._load_model_configs()
        
    def _load_model_configs(self) -> None:
        """Load model configurations from cache directory."""
        config_file = self.cache_dir / "model_configs.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    self.model_configs = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading model configs: {e}")
                self.model_configs = {}
                
    def _save_model_configs(self) -> None:
        """Save model configurations to cache directory."""
        config_file = self.cache_dir / "model_configs.json"
        
        try:
            with open(config_file, "w") as f:
                json.dump(self.model_configs, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving model configs: {e}")
            
    def register_model(
        self,
        model_type: str,
        model_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Register a new model configuration.
        
        Args:
            model_type: Type of model.
            model_id: Model ID.
            config: Model configuration.
            
        Raises:
            ValidationError: If model type is invalid.
            ProcessingError: If registration fails.
        """
        try:
            # Validate model type
            if model_type not in self.model_types:
                raise ValidationError(f"Invalid model type: {model_type}")
                
            # Add model configuration
            self.model_configs[model_id] = {
                "type": model_type,
                "config": config,
                "registered_at": datetime.utcnow().isoformat()
            }
            
            # Save configurations
            self._save_model_configs()
            
        except Exception as e:
            raise ProcessingError(f"Error registering model: {e}")
            
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get model configuration.
        
        Args:
            model_id: Model ID.
            
        Returns:
            Model configuration dictionary.
            
        Raises:
            ResourceNotFoundError: If model not found.
        """
        if model_id not in self.model_configs:
            raise ResourceNotFoundError(f"Model {model_id} not found")
            
        return self.model_configs[model_id]
        
    def load_model(
        self,
        model_id: str,
        force_reload: bool = False
    ) -> BaseModel:
        """Load a model instance.
        
        Args:
            model_id: Model ID.
            force_reload: Whether to force reload if already loaded.
            
        Returns:
            Model instance.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If loading fails.
        """
        try:
            # Check if model is already loaded
            if model_id in self.model_cache and not force_reload:
                # Move to end of cache (most recently used)
                self.model_cache.move_to_end(model_id)
                return self.model_cache[model_id]
                
            # Get model configuration
            config = self.get_model_config(model_id)
            
            # Create model instance
            model_class = self.model_types[config["type"]]
            model = model_class(
                model_id=model_id,
                device=self.device,
                **config["config"]
            )
            
            # Load model weights
            weights_path = self.cache_dir / model_id / "weights.pt"
            if weights_path.exists():
                model.load_weights(weights_path)
                
            # Add to cache
            self.model_cache[model_id] = model
            self.model_cache.move_to_end(model_id)
            
            # Remove oldest model if cache is full
            if len(self.model_cache) > self.max_cached_models:
                _, old_model = self.model_cache.popitem(last=False)
                old_model.unload()
                gc.collect()  # Force garbage collection
                
            return model
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error loading model: {e}")
            
    def unload_model(self, model_id: str) -> None:
        """Unload a model instance.
        
        Args:
            model_id: Model ID.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If unloading fails.
        """
        try:
            if model_id not in self.model_cache:
                raise ResourceNotFoundError(f"Model {model_id} not loaded")
                
            # Get model instance
            model = self.model_cache.pop(model_id)
            
            # Unload model
            model.unload()
            gc.collect()  # Force garbage collection
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error unloading model: {e}")
            
    def save_model_weights(
        self,
        model_id: str,
        weights: Any
    ) -> None:
        """Save model weights.
        
        Args:
            model_id: Model ID.
            weights: Model weights.
            
        Raises:
            ResourceNotFoundError: If model not found.
            ProcessingError: If saving fails.
        """
        try:
            # Verify model exists
            if model_id not in self.model_configs:
                raise ResourceNotFoundError(f"Model {model_id} not found")
                
            # Create weights directory
            weights_dir = self.cache_dir / model_id
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            # Save weights
            weights_path = weights_dir / "weights.pt"
            torch.save(weights, weights_path)
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error saving model weights: {e}")
            
    def list_models(
        self,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List registered models.
        
        Args:
            model_type: Optional model type filter.
            
        Returns:
            List of model configurations.
            
        Raises:
            ValidationError: If model type is invalid.
        """
        if model_type and model_type not in self.model_types:
            raise ValidationError(f"Invalid model type: {model_type}")
            
        models = []
        for model_id, config in self.model_configs.items():
            if not model_type or config["type"] == model_type:
                info = {
                    "id": model_id,
                    "type": config["type"],
                    "config": config["config"],
                    "registered_at": config["registered_at"],
                    "is_loaded": model_id in self.model_cache
                }
                models.append(info)
                
        return models
        
    def clear_cache(self) -> None:
        """Clear model cache and unload all models."""
        # Unload all models
        for model in self.model_cache.values():
            model.unload()
            
        # Clear cache
        self.model_cache.clear()
        gc.collect()  # Force garbage collection
        
    def get_memory_usage(self) -> float:
        """Get registry memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        total = 0
        
        # Model configurations
        config_file = self.cache_dir / "model_configs.json"
        if config_file.exists():
            total += config_file.stat().st_size
            
        # Model weights
        for model_id in self.model_configs:
            weights_path = self.cache_dir / model_id / "weights.pt"
            if weights_path.exists():
                total += weights_path.stat().st_size
                
        # Cached models
        if torch.cuda.is_available():
            for model in self.model_cache.values():
                total += sum(
                    p.element_size() * p.nelement()
                    for p in model.parameters()
                )
                
        return total 