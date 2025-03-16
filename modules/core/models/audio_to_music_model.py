"""Audio to Music Model for generating music from audio inputs."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging

from .base_model import BaseModel
from ..processors.audio_processor import AudioProcessor
from ..exceptions.base_exceptions import (
    ModelNotFoundError,
    ValidationError,
    ProcessingError
)


class AudioToMusicModel(BaseModel):
    """Model for generating music from audio inputs.
    
    This model takes audio input and generates variations or continuations
    of the music. It uses a transformer-based architecture with audio
    encoding and decoding components.
    """
    
    def __init__(
        self,
        version: str,
        config: Dict[str, Any],
        device: Optional[str] = None
    ):
        """Initialize the audio to music model.
        
        Args:
            version: Model version.
            config: Model configuration.
            device: Optional device to run model on.
        """
        super().__init__(version, config, device)
        
        # Initialize processor
        self.audio_processor = AudioProcessor()
        
        # Model architecture parameters
        self.hidden_dim = config.get("hidden_dim", 1024)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # Audio parameters
        self.sample_rate = config.get("sample_rate", 44100)
        self.hop_length = config.get("hop_length", 256)
        self.n_mels = config.get("n_mels", 128)
        self.n_fft = config.get("n_fft", 2048)
        
        # Build model
        self._build_model()
        
    def _build_model(self) -> None:
        """Build the model architecture."""
        try:
            # Audio encoder
            self.audio_encoder = nn.ModuleDict({
                "frontend": nn.Sequential(
                    nn.Conv1d(1, self.hidden_dim // 4, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=7, stride=2, padding=3)
                ),
                "transformer": nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.hidden_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.hidden_dim * 4,
                        dropout=self.dropout,
                        batch_first=True
                    ),
                    num_layers=self.num_layers
                )
            })
            
            # Audio decoder
            self.audio_decoder = nn.ModuleDict({
                "transformer": nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=self.hidden_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.hidden_dim * 4,
                        dropout=self.dropout,
                        batch_first=True
                    ),
                    num_layers=self.num_layers
                ),
                "backend": nn.Sequential(
                    nn.ConvTranspose1d(
                        self.hidden_dim,
                        self.hidden_dim // 2,
                        kernel_size=8,
                        stride=2,
                        padding=3
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        self.hidden_dim // 2,
                        self.hidden_dim // 4,
                        kernel_size=8,
                        stride=2,
                        padding=3
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        self.hidden_dim // 4,
                        1,
                        kernel_size=8,
                        stride=2,
                        padding=3
                    ),
                    nn.Tanh()
                )
            })
            
            # Move to device
            self.audio_encoder.to(self.device)
            self.audio_decoder.to(self.device)
            
            # Set model reference
            self.model = nn.ModuleDict({
                "encoder": self.audio_encoder,
                "decoder": self.audio_decoder
            })
            
        except Exception as e:
            raise ProcessingError(f"Error building model: {e}")
            
    def load(self) -> None:
        """Load model weights and prepare for inference."""
        try:
            # Load weights
            self.load_weights()
            
            # Set to eval mode
            self.model.eval()
            
            # Warm up model
            self.warm_up()
            
            self.is_loaded = True
            
        except Exception as e:
            raise ProcessingError(f"Error loading model: {e}")
            
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate music from audio input.
        
        Args:
            inputs: Dictionary containing:
                - audio: Input audio as numpy array
                - sample_rate: Input audio sample rate
                - duration: Optional target duration in seconds
                - temperature: Optional sampling temperature
                
        Returns:
            Dictionary containing:
                - audio: Generated audio as numpy array
                - sample_rate: Audio sample rate
                
        Raises:
            ValidationError: If inputs are invalid.
            ProcessingError: If generation fails.
        """
        try:
            # Validate inputs
            self.validate_inputs(inputs)
            
            # Get parameters
            audio = inputs["audio"]
            input_sr = inputs["sample_rate"]
            duration = inputs.get("duration", 30.0)  # Default 30 seconds
            temperature = inputs.get("temperature", 1.0)
            
            # Resample if needed
            if input_sr != self.sample_rate:
                audio = self.audio_processor.resample(
                    audio,
                    input_sr,
                    self.sample_rate
                )
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(1)  # [B, C, T]
            audio_tensor = self.to_device(audio_tensor)
            
            # Generate
            with torch.no_grad():
                # Encode audio
                encoded = self.audio_encoder.frontend(audio_tensor)
                memory = self.audio_encoder.transformer(encoded.transpose(1, 2))
                
                # Calculate target length
                target_length = int(duration * self.sample_rate)
                
                # Initialize decoder input
                decoder_input = torch.zeros(
                    (1, target_length // 8, self.hidden_dim),
                    device=self.device
                )
                
                # Decode
                decoded = self.audio_decoder.transformer(
                    decoder_input,
                    memory
                )
                audio = self.audio_decoder.backend(decoded.transpose(1, 2))
                
                # Adjust length
                audio = self._adjust_length(audio, target_length)
                
                # Apply temperature
                if temperature != 1.0:
                    audio = audio / temperature
                
                # Convert to numpy
                audio = audio.cpu().numpy().squeeze()
                
            return {
                "audio": audio,
                "sample_rate": self.sample_rate
            }
            
        except Exception as e:
            raise ProcessingError(f"Error in music generation: {e}")
            
    def _adjust_length(
        self,
        audio: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """Adjust audio length to target length.
        
        Args:
            audio: Audio tensor [B, 1, T].
            target_length: Target length in samples.
            
        Returns:
            Adjusted audio tensor.
        """
        current_length = audio.size(-1)
        
        if current_length > target_length:
            # Trim
            audio = audio[..., :target_length]
        elif current_length < target_length:
            # Pad
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
            
        return audio
        
    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess model inputs.
        
        Args:
            inputs: Raw input dictionary.
            
        Returns:
            Preprocessed inputs.
        """
        # Validate audio
        if "audio" in inputs:
            audio = inputs["audio"]
            if not isinstance(audio, np.ndarray):
                raise ValidationError("Audio must be a numpy array")
            if audio.ndim != 1:
                raise ValidationError("Audio must be mono (1D array)")
            inputs["audio"] = audio.astype(np.float32)
            
        # Validate sample rate
        if "sample_rate" in inputs:
            sr = int(inputs["sample_rate"])
            if sr <= 0:
                raise ValidationError("Sample rate must be positive")
            inputs["sample_rate"] = sr
            
        # Validate duration
        if "duration" in inputs:
            duration = float(inputs["duration"])
            if duration <= 0:
                raise ValidationError("Duration must be positive")
            if duration > self.config.get("max_duration", 300):
                raise ValidationError("Duration exceeds maximum allowed")
            inputs["duration"] = duration
            
        # Validate temperature
        if "temperature" in inputs:
            temperature = float(inputs["temperature"])
            if temperature <= 0:
                raise ValidationError("Temperature must be positive")
            inputs["temperature"] = temperature
            
        return inputs
        
    def postprocess_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess model outputs.
        
        Args:
            outputs: Raw output dictionary.
            
        Returns:
            Processed outputs.
        """
        if "audio" in outputs:
            # Normalize audio
            audio = outputs["audio"]
            audio = audio / np.abs(audio).max()
            outputs["audio"] = audio
            
        return outputs
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate model inputs.
        
        Args:
            inputs: Input dictionary.
            
        Raises:
            ValidationError: If inputs are invalid.
        """
        super().validate_inputs(inputs)
        
        # Validate required fields
        required = ["audio", "sample_rate"]
        for field in required:
            if field not in inputs:
                raise ValidationError(f"Missing required input: {field}")
                
        # Validate audio length
        max_length = self.config.get("max_audio_length", 300) * self.sample_rate
        if len(inputs["audio"]) > max_length:
            raise ValidationError(f"Audio exceeds maximum length of {max_length} samples")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage.
        
        Returns:
            Dictionary of memory usage statistics.
        """
        stats = super().get_memory_usage()
        
        # Add audio processor memory
        audio_proc_mem = self.audio_processor.get_memory_usage()
        stats["audio_processor_memory"] = audio_proc_mem
        
        return stats 