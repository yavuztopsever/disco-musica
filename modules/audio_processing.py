"""
Audio Processing Module

This module provides functionalities for audio analysis, manipulation, and synthesis.
It leverages libraries like Librosa and Pydub for audio processing tasks.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment


class AudioProcessor:
    """
    A class for audio processing operations.
    """

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the AudioProcessor.

        Args:
            sample_rate: Sample rate for audio processing.
            n_fft: FFT window size for spectrogram computation.
            hop_length: Hop length for spectrogram computation.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, audio_path: Union[str, Path], mono: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.

        Args:
            audio_path: Path to the audio file.
            mono: Whether to convert to mono.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        audio_path = Path(audio_path)
        
        # Check file format
        if audio_path.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg']:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=mono)
            return y, sr
        else:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

    def save_audio(
        self, audio_data: np.ndarray, output_path: Union[str, Path], sample_rate: Optional[int] = None
    ) -> Path:
        """
        Save audio data to a file.

        Args:
            audio_data: Audio data to save.
            output_path: Path to save the audio file.
            sample_rate: Sample rate of the audio data.

        Returns:
            Path to the saved audio file.
        """
        output_path = Path(output_path)
        
        # Use default sample rate if not specified
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Save audio file
        sf.write(output_path, audio_data, sample_rate)
        
        print(f"Saved audio to: {output_path}")
        return output_path

    def convert_format(
        self, audio_path: Union[str, Path], output_format: str = 'wav', output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Convert audio file format.

        Args:
            audio_path: Path to the audio file.
            output_format: Output audio format.
            output_path: Path to save the converted audio file.

        Returns:
            Path to the converted audio file.
        """
        audio_path = Path(audio_path)
        
        # Determine output path
        if output_path is None:
            output_path = audio_path.with_suffix(f".{output_format}")
        else:
            output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Convert audio format
        audio = AudioSegment.from_file(audio_path)
        audio.export(output_path, format=output_format)
        
        print(f"Converted audio to {output_format}: {output_path}")
        return output_path

    def change_sample_rate(
        self, audio_path: Union[str, Path], target_sr: int, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Change the sample rate of an audio file.

        Args:
            audio_path: Path to the audio file.
            target_sr: Target sample rate.
            output_path: Path to save the resampled audio file.

        Returns:
            Path to the resampled audio file.
        """
        audio_path = Path(audio_path)
        
        # Determine output path
        if output_path is None:
            output_path = audio_path.with_stem(f"{audio_path.stem}_{target_sr}hz")
        else:
            output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Resample audio
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Save resampled audio
        sf.write(output_path, y_resampled, target_sr)
        
        print(f"Changed sample rate to {target_sr} Hz: {output_path}")
        return output_path

    def compute_spectrum(
        self, audio_data: np.ndarray, spectrum_type: str = 'mel'
    ) -> np.ndarray:
        """
        Compute the spectrum of audio data.

        Args:
            audio_data: Audio data.
            spectrum_type: Type of spectrum to compute ('mel', 'stft', etc.).

        Returns:
            Computed spectrum.
        """
        if spectrum_type == 'mel':
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_data, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            # Convert to dB scale
            S_db = librosa.power_to_db(S, ref=np.max)
            return S_db
        elif spectrum_type == 'stft':
            # Compute short-time Fourier transform
            D = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            # Convert to magnitude
            S = np.abs(D)
            # Convert to dB scale
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            return S_db
        else:
            raise ValueError(f"Unsupported spectrum type: {spectrum_type}")

    def extract_features(
        self, audio_data: np.ndarray, feature_type: str = 'mfcc', **kwargs
    ) -> np.ndarray:
        """
        Extract features from audio data.

        Args:
            audio_data: Audio data.
            feature_type: Type of feature to extract ('mfcc', 'chroma', etc.).
            **kwargs: Additional arguments for feature extraction.

        Returns:
            Extracted features.
        """
        if feature_type == 'mfcc':
            # Extract MFCC features
            n_mfcc = kwargs.get('n_mfcc', 13)
            return librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
            )
        elif feature_type == 'chroma':
            # Extract chroma features
            return librosa.feature.chroma_stft(
                y=audio_data, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
        elif feature_type == 'spectral_contrast':
            # Extract spectral contrast features
            n_bands = kwargs.get('n_bands', 6)
            return librosa.feature.spectral_contrast(
                y=audio_data, sr=self.sample_rate, n_bands=n_bands, n_fft=self.n_fft, hop_length=self.hop_length
            )
        elif feature_type == 'tonnetz':
            # Extract tonnetz features
            return librosa.feature.tonnetz(
                y=audio_data, sr=self.sample_rate
            )
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

    def synthesize_audio(
        self, spectrum: np.ndarray, spectrum_type: str = 'mel', **kwargs
    ) -> np.ndarray:
        """
        Synthesize audio from a spectrum.

        Args:
            spectrum: Spectrum to synthesize audio from.
            spectrum_type: Type of spectrum ('mel', 'stft', etc.).
            **kwargs: Additional arguments for synthesis.

        Returns:
            Synthesized audio data.
        """
        if spectrum_type == 'stft':
            # Convert from dB scale
            S = librosa.db_to_amplitude(spectrum)
            # Griffin-Lim algorithm to recover phase
            n_iter = kwargs.get('n_iter', 32)
            y = librosa.griffinlim(S, n_iter=n_iter, hop_length=self.hop_length)
            return y
        else:
            raise ValueError(f"Unsupported spectrum type for synthesis: {spectrum_type}")

    def apply_effects(
        self, audio_data: np.ndarray, effects: List[Dict], **kwargs
    ) -> np.ndarray:
        """
        Apply audio effects to audio data.

        Args:
            audio_data: Audio data.
            effects: List of effects to apply.
            **kwargs: Additional arguments for effects.

        Returns:
            Processed audio data.
        """
        processed_audio = audio_data.copy()
        
        for effect in effects:
            effect_type = effect.get('type')
            
            if effect_type == 'reverb':
                # Apply reverb effect (placeholder)
                # In a real implementation, this would apply reverb
                pass
            elif effect_type == 'delay':
                # Apply delay effect (placeholder)
                # In a real implementation, this would apply delay
                pass
            elif effect_type == 'pitch_shift':
                # Apply pitch shift effect
                n_steps = effect.get('n_steps', 0)
                processed_audio = librosa.effects.pitch_shift(processed_audio, sr=self.sample_rate, n_steps=n_steps)
            elif effect_type == 'time_stretch':
                # Apply time stretch effect
                rate = effect.get('rate', 1.0)
                processed_audio = librosa.effects.time_stretch(processed_audio, rate=rate)
            else:
                print(f"Unsupported effect type: {effect_type}")
        
        return processed_audio

    def mix_audio(
        self, audio_data_1: np.ndarray, audio_data_2: np.ndarray, mix_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Mix two audio tracks.

        Args:
            audio_data_1: First audio track.
            audio_data_2: Second audio track.
            mix_ratio: Mixing ratio (0.0 to 1.0).

        Returns:
            Mixed audio data.
        """
        # Ensure both audio tracks have the same length
        min_length = min(len(audio_data_1), len(audio_data_2))
        audio_data_1 = audio_data_1[:min_length]
        audio_data_2 = audio_data_2[:min_length]
        
        # Mix audio tracks
        mixed_audio = (1 - mix_ratio) * audio_data_1 + mix_ratio * audio_data_2
        
        return mixed_audio

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data.

        Args:
            audio_data: Audio data to normalize.

        Returns:
            Normalized audio data.
        """
        return librosa.util.normalize(audio_data)

    def trim_silence(
        self, audio_data: np.ndarray, threshold_db: float = 60, pad_ms: int = 100
    ) -> np.ndarray:
        """
        Trim silence from audio data.

        Args:
            audio_data: Audio data to trim.
            threshold_db: Threshold for silence detection in dB.
            pad_ms: Padding to add after trimming in milliseconds.

        Returns:
            Trimmed audio data.
        """
        # Trim silence
        trimmed, _ = librosa.effects.trim(audio_data, top_db=threshold_db)
        
        # Add padding
        pad_samples = int(pad_ms * self.sample_rate / 1000)
        padded = np.pad(trimmed, (pad_samples, pad_samples), mode='constant')
        
        return padded


# Example usage
if __name__ == "__main__":
    audio_processor = AudioProcessor()
    # Example: y, sr = audio_processor.load_audio("path/to/audio.wav")