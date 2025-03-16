"""Audio Processor for handling audio data processing."""

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from scipy import signal

from ..exceptions.base_exceptions import (
    ProcessingError,
    ValidationError
)


class AudioProcessor:
    """Processor for handling audio data.
    
    This class provides functionality for processing audio data, including
    loading, saving, feature extraction, and various audio manipulations.
    """
    
    def __init__(self):
        """Initialize the audio processor."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Default parameters
        self.default_sr = 44100  # Default sample rate
        self.default_hop_length = 512  # Default hop length for STFT
        self.default_n_fft = 2048  # Default FFT size
        self.default_n_mels = 128  # Default number of mel bands
        
        # Supported formats
        self.supported_formats = {
            "wav": "PCM_16",
            "mp3": None,  # Uses torchaudio backend
            "ogg": "VORBIS",
            "flac": "PCM_16"
        }
        
    def load_audio(
        self,
        audio_path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Load audio file.
        
        Args:
            audio_path: Path to audio file.
            sr: Optional target sample rate.
            mono: Whether to convert to mono.
            
        Returns:
            Tuple of (audio array, sample rate).
            
        Raises:
            ProcessingError: If loading fails.
        """
        try:
            # Load audio
            audio_path = str(audio_path)
            if audio_path.endswith(".mp3"):
                # Use torchaudio for MP3
                waveform, file_sr = torchaudio.load(audio_path)
                audio = waveform.numpy()
            else:
                # Use soundfile for other formats
                audio, file_sr = sf.read(audio_path)
            
            # Convert to mono if needed
            if mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr is not None and sr != file_sr:
                audio = self.resample(audio, file_sr, sr)
                file_sr = sr
            
            return audio, file_sr
            
        except Exception as e:
            raise ProcessingError(f"Error loading audio file: {e}")
            
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sr: int,
        format: Optional[str] = None
    ) -> None:
        """Save audio file.
        
        Args:
            audio: Audio array.
            output_path: Path to save audio file.
            sr: Sample rate.
            format: Optional output format.
            
        Raises:
            ProcessingError: If saving fails.
        """
        try:
            output_path = str(output_path)
            
            # Determine format
            if format is None:
                format = output_path.split(".")[-1].lower()
            
            if format not in self.supported_formats:
                raise ValidationError(f"Unsupported format: {format}")
            
            # Save based on format
            if format == "mp3":
                # Use torchaudio for MP3
                tensor = torch.from_numpy(audio)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                torchaudio.save(output_path, tensor, sr)
            else:
                # Use soundfile for other formats
                sf.write(
                    output_path,
                    audio,
                    sr,
                    format=self.supported_formats[format]
                )
                
        except Exception as e:
            raise ProcessingError(f"Error saving audio file: {e}")
            
    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.
        
        Args:
            audio: Audio array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.
            
        Returns:
            Resampled audio array.
            
        Raises:
            ProcessingError: If resampling fails.
        """
        try:
            return librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
        except Exception as e:
            raise ProcessingError(f"Error resampling audio: {e}")
            
    def extract_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """Extract audio features.
        
        Args:
            audio: Audio array.
            sr: Sample rate.
            
        Returns:
            Dictionary of features.
            
        Raises:
            ProcessingError: If feature extraction fails.
        """
        try:
            features = {}
            
            # Basic features
            features["duration"] = len(audio) / sr
            features["rms"] = np.sqrt(np.mean(audio**2))
            features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(audio)[0].mean()
            
            # Spectral features
            stft = librosa.stft(
                audio,
                n_fft=self.default_n_fft,
                hop_length=self.default_hop_length
            )
            mag_spec = np.abs(stft)
            mel_spec = librosa.feature.melspectrogram(
                S=mag_spec,
                sr=sr,
                n_mels=self.default_n_mels
            )
            
            features.update({
                "spectral_centroid": librosa.feature.spectral_centroid(S=mag_spec)[0].mean(),
                "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=mag_spec)[0].mean(),
                "spectral_rolloff": librosa.feature.spectral_rolloff(S=mag_spec)[0].mean(),
                "mel_mean": np.mean(mel_spec),
                "mel_std": np.std(mel_spec)
            })
            
            # Rhythm features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features["tempo"] = tempo
            
            # Harmonic features
            harmonic, percussive = librosa.effects.hpss(audio)
            features.update({
                "harmonic_ratio": np.sum(harmonic**2) / np.sum(audio**2),
                "percussive_ratio": np.sum(percussive**2) / np.sum(audio**2)
            })
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Error extracting features: {e}")
            
    def normalize(
        self,
        audio: np.ndarray,
        method: str = "peak"
    ) -> np.ndarray:
        """Normalize audio.
        
        Args:
            audio: Audio array.
            method: Normalization method ("peak" or "rms").
            
        Returns:
            Normalized audio array.
            
        Raises:
            ValidationError: If method is invalid.
        """
        if method not in ["peak", "rms"]:
            raise ValidationError(f"Invalid normalization method: {method}")
            
        if method == "peak":
            # Peak normalization
            return audio / np.max(np.abs(audio))
        else:
            # RMS normalization
            target_rms = 0.2
            current_rms = np.sqrt(np.mean(audio**2))
            return audio * (target_rms / current_rms)
            
    def apply_effects(
        self,
        audio: np.ndarray,
        sr: int,
        effects: Dict[str, Any]
    ) -> np.ndarray:
        """Apply audio effects.
        
        Args:
            audio: Audio array.
            sr: Sample rate.
            effects: Dictionary of effects and parameters.
            
        Returns:
            Processed audio array.
            
        Raises:
            ValidationError: If effect parameters are invalid.
            ProcessingError: If effect application fails.
        """
        try:
            processed = audio.copy()
            
            for effect, params in effects.items():
                if effect == "reverb":
                    # Simple convolution reverb
                    room_size = params.get("room_size", 0.5)
                    decay = params.get("decay", 0.5)
                    
                    # Create impulse response
                    ir_length = int(sr * room_size)
                    ir = np.exp(-decay * np.arange(ir_length) / sr)
                    
                    # Apply convolution
                    processed = signal.convolve(processed, ir, mode="full")
                    processed = processed[:len(audio)]  # Truncate to original length
                    
                elif effect == "delay":
                    # Simple delay effect
                    delay_time = params.get("delay_time", 0.3)
                    feedback = params.get("feedback", 0.3)
                    
                    delay_samples = int(sr * delay_time)
                    num_delays = int(1 / feedback)
                    
                    delay_signal = np.zeros_like(audio)
                    for i in range(num_delays):
                        shift = i * delay_samples
                        if shift >= len(audio):
                            break
                        delay_signal[shift:] += audio[:len(audio)-shift] * (feedback ** i)
                    
                    processed = processed + delay_signal
                    
                elif effect == "pitch_shift":
                    # Pitch shifting
                    n_steps = params.get("n_steps", 0)
                    processed = librosa.effects.pitch_shift(
                        processed,
                        sr=sr,
                        n_steps=n_steps
                    )
                    
                elif effect == "time_stretch":
                    # Time stretching
                    rate = params.get("rate", 1.0)
                    processed = librosa.effects.time_stretch(
                        processed,
                        rate=rate
                    )
                    
            return self.normalize(processed)
            
        except Exception as e:
            raise ProcessingError(f"Error applying effects: {e}")
            
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -60.0,
        min_silence_duration: float = 0.1
    ) -> np.ndarray:
        """Trim silence from audio.
        
        Args:
            audio: Audio array.
            threshold_db: Silence threshold in dB.
            min_silence_duration: Minimum silence duration in seconds.
            
        Returns:
            Trimmed audio array.
        """
        try:
            return librosa.effects.trim(
                audio,
                top_db=-threshold_db,
                frame_length=2048,
                hop_length=512
            )[0]
        except Exception as e:
            raise ProcessingError(f"Error trimming silence: {e}")
            
    def split_into_segments(
        self,
        audio: np.ndarray,
        sr: int,
        segment_duration: float,
        overlap: float = 0.0
    ) -> List[np.ndarray]:
        """Split audio into segments.
        
        Args:
            audio: Audio array.
            sr: Sample rate.
            segment_duration: Segment duration in seconds.
            overlap: Overlap between segments in seconds.
            
        Returns:
            List of audio segments.
            
        Raises:
            ValidationError: If parameters are invalid.
        """
        if segment_duration <= 0:
            raise ValidationError("Segment duration must be positive")
        if overlap < 0 or overlap >= segment_duration:
            raise ValidationError("Invalid overlap duration")
            
        # Calculate sizes in samples
        segment_size = int(segment_duration * sr)
        overlap_size = int(overlap * sr)
        hop_size = segment_size - overlap_size
        
        # Create segments
        segments = []
        start = 0
        while start + segment_size <= len(audio):
            segment = audio[start:start + segment_size]
            segments.append(segment)
            start += hop_size
            
        return segments
        
    def get_memory_usage(self) -> float:
        """Get processor memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        # Estimate memory usage of internal data structures
        memory = 0
        
        # Format dictionary
        memory += sum(
            len(k) + len(str(v)) for k, v in self.supported_formats.items()
        )
        
        return memory 