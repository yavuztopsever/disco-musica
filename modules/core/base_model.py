"""
Base model module for Disco Musica.

This module provides a base class for all models in the application,
defining common functionalities and interfaces.
"""

from abc import ABC, abstractmethod
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from modules.core.config import config


class BaseModel(ABC):
    """
    Abstract base class for all models in Disco Musica.
    
    This class defines the common interface for model implementations,
    including loading, saving, and inference operations.
    """
    
    def __init__(self, model_name: str, model_type: str = "pretrained"):
        """
        Initialize the BaseModel.
        
        Args:
            model_name: Name of the model.
            model_type: Type of model ('pretrained' or 'finetuned').
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = self._get_device()
        self.model = None
        self.metadata = {
            "name": model_name,
            "type": model_type,
            "capabilities": [],
            "version": "0.1.0",
            "creation_date": None,
            "last_modified": None,
            "parameters": {},
        }
    
    def _get_device(self) -> torch.device:
        """
        Get the appropriate device for the model.
        
        Returns:
            PyTorch device object.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        else:
            return torch.device("cpu")
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model from disk or download if necessary.
        """
        pass
    
    @abstractmethod
    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the model to disk.
        
        Args:
            output_path: Path to save the model. If None, a default path based on 
                         model name and type will be used.
        
        Returns:
            Path where the model was saved.
        """
        pass
    
    @abstractmethod
    def generate(self, inputs: Any, **kwargs) -> Any:
        """
        Generate output from the model.
        
        Args:
            inputs: Input data for generation.
            **kwargs: Additional generation parameters.
        
        Returns:
            Generated output.
        """
        pass
    
    def to(self, device: Union[str, torch.device]) -> 'BaseModel':
        """
        Move the model to a specific device.
        
        Args:
            device: Device to move the model to ('cuda', 'cpu', 'mps', or torch.device).
        
        Returns:
            Self for chaining.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        if self.model is not None:
            self.model.to(device)
        
        return self
    
    def get_model_path(self) -> Path:
        """
        Get the path where the model is or should be stored.
        
        Returns:
            Path to the model directory.
        """
        base_dir = Path(config.get("models", "model_cache_dir", "models"))
        if self.model_type == "pretrained":
            return base_dir / "pretrained" / self.model_name
        else:
            return base_dir / "finetuned" / self.model_name
    
    def save_metadata(self, metadata_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save model metadata to a JSON file.
        
        Args:
            metadata_path: Path to save the metadata. If None, it will be saved
                           alongside the model.
        
        Returns:
            Path where the metadata was saved.
        """
        if metadata_path is None:
            metadata_path = self.get_model_path() / "metadata.json"
        
        metadata_path = Path(metadata_path)
        os.makedirs(metadata_path.parent, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return metadata_path
    
    def load_metadata(self, metadata_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load model metadata from a JSON file.
        
        Args:
            metadata_path: Path to load the metadata from. If None, it will be loaded
                           from alongside the model.
        
        Returns:
            Dictionary containing the model metadata.
        """
        if metadata_path is None:
            metadata_path = self.get_model_path() / "metadata.json"
        
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            return self.metadata
        
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
            self.metadata.update(loaded_metadata)
        
        return self.metadata
    
    def update_metadata(self, **kwargs) -> None:
        """
        Update model metadata.
        
        Args:
            **kwargs: Key-value pairs to update in the metadata.
        """
        for key, value in kwargs.items():
            if key in self.metadata:
                self.metadata[key] = value
            elif "parameters" in key:
                param_key = key.split(".")[-1]
                self.metadata["parameters"][param_key] = value
    
    def get_model_size(self) -> int:
        """
        Get the size of the model in parameters.
        
        Returns:
            Number of parameters in the model.
        """
        if self.model is None:
            return 0
        
        return sum(p.numel() for p in self.model.parameters())
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if the model is loaded, False otherwise.
        """
        return self.model is not None


class PretrainedModelMixin:
    """
    Mixin for pre-trained models from external sources like Hugging Face.
    """
    
    def download_from_huggingface(self, repo_id: str, local_dir: Union[str, Path], 
                                 use_auth_token: Optional[str] = None) -> Path:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., 'facebook/musicgen-small').
            local_dir: Local directory to save the model.
            use_auth_token: Optional authentication token for private repositories.
        
        Returns:
            Path to the downloaded model.
        """
        from huggingface_hub import snapshot_download
        
        local_dir = Path(local_dir)
        os.makedirs(local_dir.parent, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            use_auth_token=use_auth_token
        )
        
        # Update metadata
        self.update_metadata(
            source="huggingface",
            source_repo=repo_id
        )
        
        return local_dir


class TorchModelMixin:
    """
    Mixin for PyTorch-based models with common functionalities.
    """
    
    def apply_torch_compile(self, **kwargs) -> None:
        """
        Apply torch.compile to the model for potentially faster inference.
        
        Args:
            **kwargs: Arguments to pass to torch.compile.
        """
        if not hasattr(torch, 'compile'):
            print("Warning: torch.compile is not available in this PyTorch version.")
            return
        
        if self.model is not None:
            self.model = torch.compile(self.model, **kwargs)
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Calculate the memory usage of the model.
        
        Returns:
            Tuple of (GPU memory in MB, CPU memory in MB).
        """
        if self.model is None:
            return (0.0, 0.0)
        
        # Calculate parameters memory usage
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        # Calculate buffer memory usage
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        # Total memory in MB
        mem_size = (param_size + buffer_size) / 1024**2
        
        # GPU memory if applicable
        gpu_mem = 0.0
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        return (gpu_mem, mem_size)
    
    def quantize(self, quantization_type: str = "int8") -> None:
        """
        Quantize the model to reduce memory usage and potentially speed up inference.
        
        Args:
            quantization_type: Type of quantization to apply ('int8', 'fp16', etc.).
        """
        if self.model is None:
            raise ValueError("Model must be loaded before quantization.")
        
        if quantization_type == "int8":
            # INT8 quantization
            try:
                from torch.quantization import quantize_dynamic
                self.model = quantize_dynamic(
                    self.model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
            except Exception as e:
                print(f"Failed to apply INT8 quantization: {e}")
        
        elif quantization_type == "fp16":
            # FP16 quantization
            try:
                self.model = self.model.half()
            except Exception as e:
                print(f"Failed to apply FP16 quantization: {e}")
        
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        # Update metadata
        self.update_metadata(quantized=True, quantization_type=quantization_type)