"""Base Model class for AI models."""

import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import json
import torch
import numpy as np
from datetime import datetime

from ..config import config
from ..exceptions.base_exceptions import (
    ModelNotFoundError,
    ValidationError,
    ProcessingError
)


class BaseModel(ABC):
    """Base class for all AI models in the system.
    
    This class defines the common interface and functionality that all
    model implementations must provide. It handles model initialization,
    configuration management, resource handling, and basic operations.
    """
    
    def __init__(
        self,
        version: str,
        config: Dict[str, Any],
        device: Optional[str] = None
    ):
        """Initialize the model.
        
        Args:
            version: Model version.
            config: Model configuration.
            device: Optional device to run model on.
        """
        self.version = version
        self.config = config
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{version}")
        
        # Set device
        self.device = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Model state
        self.is_loaded = False
        self.model = None
        self.metadata = {
            "version": version,
            "config": config,
            "device": self.device,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate model configuration.
        
        Raises:
            ValidationError: If configuration is invalid.
        """
        required_fields = [
            "model_type",
            "input_shape",
            "output_shape",
            "sample_rate"
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValidationError(f"Missing required config field: {field}")
                
    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference.
        
        This method must be implemented by subclasses to load their
        specific model architecture and weights.
        
        Raises:
            ModelNotFoundError: If model weights not found.
            ProcessingError: If loading fails.
        """
        pass
        
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run model inference.
        
        This method must be implemented by subclasses to perform their
        specific inference logic.
        
        Args:
            inputs: Model inputs.
            
        Returns:
            Model outputs.
            
        Raises:
            ValidationError: If inputs are invalid.
            ProcessingError: If inference fails.
        """
        pass
        
    def cleanup(self) -> None:
        """Clean up model resources.
        
        This method handles cleanup of model resources, including
        GPU memory if applicable.
        """
        if self.model is not None:
            if isinstance(self.model, torch.nn.Module):
                # Move model to CPU and clear CUDA cache
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.model = None
            
        self.is_loaded = False
        
    def to_device(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Move data to model device.
        
        Args:
            data: Input data.
            
        Returns:
            Data on target device.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device)
        
    def get_weights_path(self) -> Path:
        """Get path to model weights.
        
        Returns:
            Path to weights file.
        """
        weights_dir = Path(config.get("paths", "weights_dir", "weights"))
        return weights_dir / self.config["model_type"] / f"{self.version}.pt"
        
    def save_weights(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        """Save model weights.
        
        Args:
            weights_path: Optional path to save weights to.
            
        Raises:
            ProcessingError: If saving fails.
        """
        if self.model is None:
            raise ProcessingError("No model to save")
            
        try:
            # Get save path
            weights_path = Path(weights_path or self.get_weights_path())
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save weights
            if isinstance(self.model, torch.nn.Module):
                torch.save(self.model.state_dict(), weights_path)
            else:
                raise ProcessingError("Model format not supported for saving")
                
        except Exception as e:
            raise ProcessingError(f"Error saving weights: {e}")
            
    def load_weights(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        """Load model weights.
        
        Args:
            weights_path: Optional path to load weights from.
            
        Raises:
            ModelNotFoundError: If weights not found.
            ProcessingError: If loading fails.
        """
        try:
            # Get weights path
            weights_path = Path(weights_path or self.get_weights_path())
            if not weights_path.exists():
                raise ModelNotFoundError(f"Weights not found: {weights_path}")
                
            # Load weights
            if isinstance(self.model, torch.nn.Module):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                raise ProcessingError("Model format not supported for loading")
                
        except Exception as e:
            raise ProcessingError(f"Error loading weights: {e}")
            
    def get_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary of model information.
        """
        return {
            "version": self.version,
            "config": self.config,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "metadata": self.metadata
        }
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate model inputs.
        
        Args:
            inputs: Input data.
            
        Raises:
            ValidationError: If inputs are invalid.
        """
        required_fields = self.config.get("required_inputs", [])
        for field in required_fields:
            if field not in inputs:
                raise ValidationError(f"Missing required input: {field}")
                
    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess model inputs.
        
        This method can be overridden by subclasses to implement
        specific preprocessing logic.
        
        Args:
            inputs: Raw input data.
            
        Returns:
            Preprocessed inputs.
        """
        return inputs
        
    def postprocess_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess model outputs.
        
        This method can be overridden by subclasses to implement
        specific postprocessing logic.
        
        Args:
            outputs: Raw model outputs.
            
        Returns:
            Processed outputs.
        """
        return outputs
        
    def warm_up(self) -> None:
        """Warm up model with dummy inputs.
        
        This method runs a dummy inference to warm up the model
        and catch any initialization issues.
        """
        try:
            # Create dummy inputs based on input shape
            dummy_inputs = {}
            input_shape = self.config["input_shape"]
            
            for key, shape in input_shape.items():
                dummy_inputs[key] = torch.randn(shape).to(self.device)
                
            # Run dummy inference
            with torch.no_grad():
                _ = self.predict(dummy_inputs)
                
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage.
        
        Returns:
            Dictionary of memory usage statistics.
        """
        stats = {
            "cpu_memory": 0.0,
            "gpu_memory": 0.0
        }
        
        if isinstance(self.model, torch.nn.Module):
            # Get CPU memory
            for param in self.model.parameters():
                stats["cpu_memory"] += param.element_size() * param.nelement()
                
            # Get GPU memory if applicable
            if self.device.startswith("cuda"):
                stats["gpu_memory"] = torch.cuda.max_memory_allocated(self.device)
                
        return stats
        
    def __repr__(self) -> str:
        """Get string representation.
        
        Returns:
            Model string representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"version='{self.version}', "
            f"device='{self.device}', "
            f"loaded={self.is_loaded})"
        ) 