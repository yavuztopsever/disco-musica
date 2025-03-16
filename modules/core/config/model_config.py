"""Configuration models for the system."""
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass


class AudioProcessorConfig(BaseModel):
    """Configuration for audio processor."""
    
    sample_rate: int = Field(44100, description="Sample rate in Hz")
    n_fft: int = Field(2048, description="FFT size")
    hop_length: int = Field(512, description="Hop length in samples")
    normalize: bool = Field(True, description="Whether to normalize audio")


@dataclass
class ModelConfig:
    """Configuration for a model."""
    
    model_type: str
    version: str
    weights_path: str
    device: str = "cuda"
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.model_type:
            raise ValueError("Model type cannot be empty")
        if not self.version:
            raise ValueError("Version cannot be empty")
        if not self.weights_path:
            raise ValueError("Weights path cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be positive")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config.
        """
        return {
            "model_type": self.model_type,
            "version": self.version,
            "weights_path": self.weights_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "tensorboard_dir": self.tensorboard_dir,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_run_name": self.wandb_run_name,
            "wandb_config": self.wandb_config
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary.
        
        Args:
            data: Dictionary representation of config.
            
        Returns:
            ModelConfig instance.
        """
        return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be positive")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config.
        """
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "tensorboard_dir": self.tensorboard_dir,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_run_name": self.wandb_run_name,
            "wandb_config": self.wandb_config
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary.
        
        Args:
            data: Dictionary representation of config.
            
        Returns:
            TrainingConfig instance.
        """
        return cls(**data)


class GenerationConfig(BaseModel):
    """Configuration for music generation."""
    
    temperature: float = Field(0.8, description="Sampling temperature")
    max_duration_seconds: int = Field(30, description="Maximum generation duration")
    top_k: int = Field(50, description="Top-k sampling parameter")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    repetition_penalty: float = Field(1.2, description="Repetition penalty")
    style_weight: float = Field(1.0, description="Style influence weight") 