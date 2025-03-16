"""MIDI to Audio Model for generating audio from MIDI inputs."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging
import pretty_midi

from .base_model import BaseModel
from ..processors.midi_processor import MIDIProcessor
from ..processors.audio_processor import AudioProcessor
from ..exceptions.base_exceptions import (
    ModelNotFoundError,
    ValidationError,
    ProcessingError
)


class MIDIToAudioModel(BaseModel):
    """Model for generating audio from MIDI inputs.
    
    This model takes MIDI input and generates corresponding audio output.
    It uses a transformer-based architecture to convert MIDI events into
    audio waveforms, with support for different instruments and styles.
    """
    
    def __init__(
        self,
        version: str,
        config: Dict[str, Any],
        device: Optional[str] = None
    ):
        """Initialize the MIDI to audio model.
        
        Args:
            version: Model version.
            config: Model configuration.
            device: Optional device to run model on.
        """
        super().__init__(version, config, device)
        
        # Initialize processors
        self.midi_processor = MIDIProcessor()
        self.audio_processor = AudioProcessor()
        
        # Model architecture parameters
        self.hidden_dim = config.get("hidden_dim", 1024)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        
        # MIDI parameters
        self.max_events = config.get("max_events", 2048)
        self.event_dim = config.get("event_dim", 256)
        
        # Audio parameters
        self.sample_rate = config.get("sample_rate", 44100)
        self.hop_length = config.get("hop_length", 256)
        self.n_mels = config.get("n_mels", 128)
        
        # Build model
        self._build_model()
        
    def _build_model(self) -> None:
        """Build the model architecture."""
        try:
            # MIDI encoder
            self.midi_encoder = nn.ModuleDict({
                "embedding": nn.Embedding(
                    self.event_dim,
                    self.hidden_dim
                ),
                "position_encoding": nn.Parameter(
                    torch.randn(1, self.max_events, self.hidden_dim)
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
                "upsampler": nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8)
                ),
                "conv_stack": nn.Sequential(
                    nn.ConvTranspose1d(
                        self.hidden_dim * 8,
                        self.hidden_dim * 4,
                        kernel_size=8,
                        stride=4,
                        padding=2
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        self.hidden_dim * 4,
                        self.hidden_dim * 2,
                        kernel_size=8,
                        stride=4,
                        padding=2
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        self.hidden_dim * 2,
                        1,
                        kernel_size=8,
                        stride=4,
                        padding=2
                    ),
                    nn.Tanh()
                )
            })
            
            # Move to device
            self.midi_encoder.to(self.device)
            self.audio_decoder.to(self.device)
            
            # Set model reference
            self.model = nn.ModuleDict({
                "encoder": self.midi_encoder,
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
        """Generate audio from MIDI input.
        
        Args:
            inputs: Dictionary containing:
                - midi: MIDI data (PrettyMIDI object or path)
                - instrument: Optional instrument name
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
            midi_data = inputs["midi"]
            instrument = inputs.get("instrument", "piano")
            duration = inputs.get("duration")
            temperature = inputs.get("temperature", 1.0)
            
            # Load MIDI if path provided
            if isinstance(midi_data, (str, Path)):
                midi_data = pretty_midi.PrettyMIDI(str(midi_data))
            
            # Process MIDI
            midi_events = self.midi_processor.encode_midi(
                midi_data,
                max_events=self.max_events
            )
            midi_events = self.to_device(midi_events)
            
            # Get duration from MIDI if not specified
            if duration is None:
                duration = midi_data.get_end_time()
            
            # Generate
            with torch.no_grad():
                # Encode MIDI
                event_embeddings = self.midi_encoder.embedding(midi_events)
                event_embeddings = event_embeddings + self.midi_encoder.position_encoding[:, :midi_events.size(1)]
                memory = self.midi_encoder.transformer(event_embeddings)
                
                # Calculate target length
                target_length = int(duration * self.sample_rate)
                
                # Initialize decoder input
                decoder_input = torch.zeros(
                    (1, target_length // 64, self.hidden_dim),
                    device=self.device
                )
                
                # Decode
                decoded = self.audio_decoder.transformer(
                    decoder_input,
                    memory
                )
                decoded = self.audio_decoder.upsampler(decoded)
                decoded = decoded.transpose(1, 2)  # [B, C, T]
                audio = self.audio_decoder.conv_stack(decoded)
                
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
            raise ProcessingError(f"Error in audio generation: {e}")
            
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
        # Validate MIDI
        if "midi" in inputs:
            midi = inputs["midi"]
            if isinstance(midi, (str, Path)):
                try:
                    midi = pretty_midi.PrettyMIDI(str(midi))
                except Exception as e:
                    raise ValidationError(f"Invalid MIDI file: {e}")
            elif not isinstance(midi, pretty_midi.PrettyMIDI):
                raise ValidationError("MIDI must be a PrettyMIDI object or file path")
            inputs["midi"] = midi
            
        # Validate instrument
        if "instrument" in inputs:
            instrument = inputs["instrument"]
            if not isinstance(instrument, str):
                raise ValidationError("Instrument must be a string")
            if instrument not in self.config.get("supported_instruments", ["piano"]):
                raise ValidationError(f"Unsupported instrument: {instrument}")
            
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
        if "midi" not in inputs:
            raise ValidationError("MIDI input is required")
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage.
        
        Returns:
            Dictionary of memory usage statistics.
        """
        stats = super().get_memory_usage()
        
        # Add processor memory
        midi_proc_mem = self.midi_processor.get_memory_usage()
        audio_proc_mem = self.audio_processor.get_memory_usage()
        stats.update({
            "midi_processor_memory": midi_proc_mem,
            "audio_processor_memory": audio_proc_mem
        })
        
        return stats 