"""Mastering model for handling audio mastering operations."""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource
from ...core.config import ModelConfig

logger = logging.getLogger(__name__)

class MasteringConfig(BaseModel):
    """Configuration for the mastering model."""
    model_name: str
    model_version: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate: int = 44100
    channels: int = 2
    target_lufs: float = -14.0
    target_peak: float = -1.0
    attack_time: float = 0.01
    release_time: float = 0.1
    ratio: float = 4.0
    threshold: float = -18.0
    knee: float = 6.0
    make_up_gain: float = 0.0
    metadata: Dict[str, Any] = {}

class MasteringModel:
    """Model for handling audio mastering operations."""
    
    def __init__(
        self,
        config: MasteringConfig,
        model: Optional[nn.Module] = None
    ):
        """Initialize the mastering model.
        
        Args:
            config: Model configuration.
            model: Optional pre-trained model.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model()
            
    def _load_model(self) -> nn.Module:
        """Load the pre-trained model.
        
        Returns:
            Loaded model.
        """
        try:
            # Load model from registry
            model = ModelRegistry.get_model(
                self.config.model_name,
                self.config.model_version
            )
            return model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}")
            
    async def master(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Master audio using the model.
        
        Args:
            audio: Input audio tensor.
            sample_rate: Optional sample rate (defaults to config).
            **kwargs: Additional mastering parameters.
            
        Returns:
            Mastered audio and metadata.
        """
        try:
            # Move audio to device
            audio = audio.to(self.device)
            
            # Resample if needed
            if sample_rate is not None and sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.sample_rate
                ).to(self.device)
                audio = resampler(audio)
                
            # Update mastering parameters
            master_kwargs = self.config.dict()
            master_kwargs.update(kwargs)
            
            # Master
            with torch.no_grad():
                outputs = self.model.master(
                    audio=audio,
                    **master_kwargs
                )
                
            # Process outputs
            mastered_audio = outputs.audio
            metadata = outputs.metadata if hasattr(outputs, "metadata") else {}
            
            # Convert to numpy arrays
            mastered_audio = mastered_audio.cpu().numpy()
            
            return {
                "audio": mastered_audio,
                "sample_rate": self.config.sample_rate,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "mastering_params": master_kwargs,
                    "device": str(self.device),
                    **metadata
                }
            }
            
        except Exception as e:
            self.logger.error(f"Mastering failed: {e}")
            raise DiscoMusicaError(f"Mastering failed: {e}")
            
    async def batch_master(
        self,
        audio_list: List[torch.Tensor],
        sample_rates: Optional[List[int]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Master multiple audio files.
        
        Args:
            audio_list: List of input audio tensors.
            sample_rates: Optional list of sample rates.
            **kwargs: Additional mastering parameters.
            
        Returns:
            List of mastered audio and metadata.
        """
        try:
            results = []
            for i, audio in enumerate(audio_list):
                # Get sample rate if provided
                sample_rate = sample_rates[i] if sample_rates is not None else None
                
                # Master audio
                result = await self.master(
                    audio=audio,
                    sample_rate=sample_rate,
                    **kwargs
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch mastering failed: {e}")
            raise DiscoMusicaError(f"Batch mastering failed: {e}")
            
    async def analyze(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze audio characteristics.
        
        Args:
            audio: Input audio tensor.
            sample_rate: Optional sample rate (defaults to config).
            
        Returns:
            Audio analysis results.
        """
        try:
            # Move audio to device
            audio = audio.to(self.device)
            
            # Resample if needed
            if sample_rate is not None and sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.sample_rate
                ).to(self.device)
                audio = resampler(audio)
                
            # Analyze
            with torch.no_grad():
                outputs = self.model.analyze(audio=audio)
                
            # Process outputs
            analysis = outputs.analysis if hasattr(outputs, "analysis") else {}
            
            return {
                "analysis": analysis,
                "sample_rate": self.config.sample_rate,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise DiscoMusicaError(f"Analysis failed: {e}")
            
    async def enhance(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Enhance audio quality.
        
        Args:
            audio: Input audio tensor.
            sample_rate: Optional sample rate (defaults to config).
            **kwargs: Additional enhancement parameters.
            
        Returns:
            Enhanced audio and metadata.
        """
        try:
            # Move audio to device
            audio = audio.to(self.device)
            
            # Resample if needed
            if sample_rate is not None and sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.config.sample_rate
                ).to(self.device)
                audio = resampler(audio)
                
            # Update enhancement parameters
            enhance_kwargs = self.config.dict()
            enhance_kwargs.update(kwargs)
            
            # Enhance
            with torch.no_grad():
                outputs = self.model.enhance(
                    audio=audio,
                    **enhance_kwargs
                )
                
            # Process outputs
            enhanced_audio = outputs.audio
            metadata = outputs.metadata if hasattr(outputs, "metadata") else {}
            
            # Convert to numpy arrays
            enhanced_audio = enhanced_audio.cpu().numpy()
            
            return {
                "audio": enhanced_audio,
                "sample_rate": self.config.sample_rate,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "enhancement_params": enhance_kwargs,
                    "device": str(self.device),
                    **metadata
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhancement failed: {e}")
            raise DiscoMusicaError(f"Enhancement failed: {e}")
            
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model.
        """
        try:
            # Save model state
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config.dict()
                },
                path
            )
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise DiscoMusicaError(f"Failed to save model: {e}")
            
    @classmethod
    def load(cls, path: str) -> "MasteringModel":
        """Load a model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded model.
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(path)
            
            # Create config
            config = MasteringConfig(**checkpoint["config"])
            
            # Create model
            model = cls(config)
            
            # Load state dict
            model.model.load_state_dict(checkpoint["model_state_dict"])
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}") 