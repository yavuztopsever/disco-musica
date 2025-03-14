"""
Audio processing module for Disco Musica.

This module provides utilities for processing audio data, including loading,
saving, feature extraction, and various transformations.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment

from modules.core.config import config


class AudioProcessor:
    """
    Class for audio processing operations.
    
    This class provides methods for loading, saving, and processing audio data,
    including feature extraction, normalization, and other transformations.
    """
    
    def __init__(
        self, 
        sample_rate: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None
    ):
        """
        Initialize the AudioProcessor.
        
        Args:
            sample_rate: Sample rate for audio processing. If None, uses the value from config.
            n_fft: FFT window size for spectrogram computation. If None, uses the value from config.
            hop_length: Hop length for spectrogram computation. If None, uses the value from config.
        """
        self.sample_rate = sample_rate or config.get("audio", "sample_rate", 44100)
        self.n_fft = n_fft or 2048
        self.hop_length = hop_length or 512
        
        # Load other config values
        self.bit_depth = config.get("audio", "bit_depth", 16)
        self.channels = config.get("audio", "channels", 2)
        self.format = config.get("audio", "format", "wav")
        self.normalization_level = config.get("audio", "normalization_level", -14.0)
        self.segment_length = config.get("audio", "segment_length", 30.0)
        self.overlap = config.get("audio", "overlap", 5.0)
    
    def load_audio(self, 
                  audio_path: Union[str, Path], 
                  sr: Optional[int] = None, 
                  mono: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.
        
        Args:
            audio_path: Path to the audio file.
            sr: Sample rate to load the audio at. If None, uses the instance sample rate.
            mono: Whether to convert to mono.
        
        Returns:
            Tuple of (audio_data, sample_rate).
        """
        sr = sr or self.sample_rate
        
        try:
            y, sr_orig = librosa.load(audio_path, sr=sr, mono=mono)
            return y, sr
        except Exception as e:
            raise IOError(f"Failed to load audio file {audio_path}: {e}")
    
    def save_audio(self, 
                  audio_data: np.ndarray, 
                  output_path: Union[str, Path],
                  sr: Optional[int] = None,
                  format: Optional[str] = None) -> Path:
        """
        Save audio data to a file.
        
        Args:
            audio_data: Audio data to save.
            output_path: Path to save the audio file.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            format: Format to save the audio in. If None, uses the instance format.
        
        Returns:
            Path to the saved audio file.
        """
        sr = sr or self.sample_rate
        format = format or self.format
        
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            sf.write(output_path, audio_data, sr, format=format)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save audio file {output_path}: {e}")
    
    def convert_format(self, 
                      audio_path: Union[str, Path], 
                      output_format: str = "wav",
                      output_path: Optional[Union[str, Path]] = None) -> Path:
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
        
        if output_path is None:
            output_path = audio_path.with_suffix(f".{output_format}")
        else:
            output_path = Path(output_path)
            
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            audio = AudioSegment.from_file(audio_path)
            audio.export(output_path, format=output_format)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to convert audio file {audio_path} to {output_format}: {e}")
    
    def change_sample_rate(self, 
                          audio_path: Union[str, Path], 
                          target_sr: int,
                          output_path: Optional[Union[str, Path]] = None) -> Path:
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
        
        if output_path is None:
            output_path = audio_path.with_stem(f"{audio_path.stem}_{target_sr}Hz")
        else:
            output_path = Path(output_path)
            
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Resample
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            
            # Save
            sf.write(output_path, y_resampled, target_sr)
            
            return output_path
        except Exception as e:
            raise IOError(f"Failed to resample audio file {audio_path} to {target_sr} Hz: {e}")
    
    def compute_melspectrogram(self, 
                              audio_data: np.ndarray,
                              sr: Optional[int] = None,
                              n_fft: Optional[int] = None,
                              hop_length: Optional[int] = None,
                              n_mels: int = 128,
                              fmin: int = 20,
                              fmax: Optional[int] = None) -> np.ndarray:
        """
        Compute a mel spectrogram from audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            n_fft: FFT window size. If None, uses the instance n_fft.
            hop_length: Hop length. If None, uses the instance hop_length.
            n_mels: Number of mel bands.
            fmin: Minimum frequency.
            fmax: Maximum frequency. If None, uses sr/2.
            
        Returns:
            Mel spectrogram as a numpy array.
        """
        sr = sr or self.sample_rate
        n_fft = n_fft or self.n_fft
        hop_length = hop_length or self.hop_length
        
        try:
            S = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=sr, 
                n_fft=n_fft, 
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax
            )
            
            # Convert to dB scale
            S_db = librosa.power_to_db(S, ref=np.max)
            
            return S_db
        except Exception as e:
            raise RuntimeError(f"Failed to compute mel spectrogram: {e}")
    
    def compute_stft(self, 
                    audio_data: np.ndarray,
                    sr: Optional[int] = None,
                    n_fft: Optional[int] = None,
                    hop_length: Optional[int] = None) -> np.ndarray:
        """
        Compute the Short-Time Fourier Transform (STFT) of audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            n_fft: FFT window size. If None, uses the instance n_fft.
            hop_length: Hop length. If None, uses the instance hop_length.
            
        Returns:
            STFT as a complex-valued numpy array.
        """
        sr = sr or self.sample_rate
        n_fft = n_fft or self.n_fft
        hop_length = hop_length or self.hop_length
        
        try:
            D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
            return D
        except Exception as e:
            raise RuntimeError(f"Failed to compute STFT: {e}")
    
    def extract_features(self, 
                        audio_data: np.ndarray,
                        sr: Optional[int] = None,
                        feature_types: List[str] = ["mfcc", "chroma", "spectral_contrast", "tonnetz"]) -> Dict[str, np.ndarray]:
        """
        Extract various features from audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            feature_types: List of feature types to extract.
            
        Returns:
            Dictionary mapping feature names to feature arrays.
        """
        sr = sr or self.sample_rate
        features = {}
        
        try:
            for feature_type in feature_types:
                if feature_type == "mfcc":
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
                    features["mfcc"] = mfccs
                    
                elif feature_type == "chroma":
                    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
                    features["chroma"] = chroma
                    
                elif feature_type == "spectral_contrast":
                    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
                    features["spectral_contrast"] = contrast
                    
                elif feature_type == "tonnetz":
                    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
                    features["tonnetz"] = tonnetz
                    
                elif feature_type == "tempogram":
                    oenv = librosa.onset.onset_strength(y=audio_data, sr=sr)
                    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
                    features["tempogram"] = tempogram
                    
                else:
                    print(f"Warning: Unsupported feature type '{feature_type}'")
                    
            return features
        except Exception as e:
            raise RuntimeError(f"Failed to extract features: {e}")
    
    def normalize_audio(self, 
                       audio_data: np.ndarray,
                       target_lufs: Optional[float] = None) -> np.ndarray:
        """
        Normalize audio data to a target loudness level (LUFS).
        
        Args:
            audio_data: Audio data.
            target_lufs: Target loudness level in LUFS. If None, uses the instance normalization_level.
            
        Returns:
            Normalized audio data.
        """
        target_lufs = target_lufs or self.normalization_level
        
        try:
            # Simple peak normalization as a fallback if pyloudnorm is not available
            return librosa.util.normalize(audio_data)
            
            # For more accurate LUFS normalization, install pyloudnorm:
            # pip install pyloudnorm
            # Then use the following code:
            """
            import pyloudnorm as pyln
            
            # Create meter
            meter = pyln.Meter(self.sample_rate)
            
            # Measure loudness
            loudness = meter.integrated_loudness(audio_data)
            
            # Calculate gain needed to reach target
            gain_db = target_lufs - loudness
            
            # Apply gain
            normalized_audio = pyln.normalize.loudness(audio_data, loudness, target_lufs)
            
            return normalized_audio
            """
        except Exception as e:
            print(f"Warning: Using simple peak normalization due to error: {e}")
            return librosa.util.normalize(audio_data)
    
    def trim_silence(self, 
                    audio_data: np.ndarray,
                    top_db: float = 60.0,
                    frame_length: int = 2048,
                    hop_length: Optional[int] = None) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio data.
        
        Args:
            audio_data: Audio data.
            top_db: Threshold (in decibels) below reference to consider as silence.
            frame_length: Frame length for silence detection.
            hop_length: Hop length for silence detection. If None, uses the instance hop_length.
            
        Returns:
            Trimmed audio data.
        """
        hop_length = hop_length or self.hop_length
        
        try:
            trimmed_audio, _ = librosa.effects.trim(
                audio_data,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            return trimmed_audio
        except Exception as e:
            raise RuntimeError(f"Failed to trim silence: {e}")
    
    def segment_audio(self, 
                     audio_data: np.ndarray,
                     sr: Optional[int] = None,
                     segment_length: Optional[float] = None,
                     overlap: Optional[float] = None) -> List[np.ndarray]:
        """
        Segment audio data into smaller chunks.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            segment_length: Length of each segment in seconds. If None, uses the instance segment_length.
            overlap: Overlap between segments in seconds. If None, uses the instance overlap.
            
        Returns:
            List of audio segments as numpy arrays.
        """
        sr = sr or self.sample_rate
        segment_length = segment_length or self.segment_length
        overlap = overlap or self.overlap
        
        # Convert to samples
        segment_samples = int(segment_length * sr)
        hop_samples = int((segment_length - overlap) * sr)
        
        # Generate segments
        segments = []
        for i in range(0, len(audio_data) - segment_samples + 1, hop_samples):
            segment = audio_data[i:i + segment_samples]
            segments.append(segment)
        
        # Handle the last segment if needed
        if len(segments) == 0 and len(audio_data) > 0:
            # If audio is shorter than segment_length, pad with zeros
            segment = np.zeros(segment_samples)
            segment[:len(audio_data)] = audio_data
            segments.append(segment)
        elif len(audio_data) > (i + hop_samples) and len(audio_data) < (i + segment_samples):
            # If there's a partial segment at the end
            segment = np.zeros(segment_samples)
            segment[:len(audio_data) - (i + hop_samples)] = audio_data[i + hop_samples:]
            segments.append(segment)
        
        return segments
    
    def detect_tempo(self, 
                    audio_data: np.ndarray,
                    sr: Optional[int] = None) -> Tuple[float, float]:
        """
        Detect the tempo of audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            
        Returns:
            Tuple of (estimated_tempo, confidence).
        """
        sr = sr or self.sample_rate
        
        try:
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            
            # Calculate confidence based on onset strength variation
            onset_std = np.std(onset_env)
            confidence = min(1.0, onset_std / 0.1)  # Normalize, with 0.1 as a reference value
            
            return tempo[0], confidence
        except Exception as e:
            raise RuntimeError(f"Failed to detect tempo: {e}")
    
    def detect_key(self, 
                  audio_data: np.ndarray,
                  sr: Optional[int] = None) -> Tuple[str, float]:
        """
        Detect the musical key of audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            
        Returns:
            Tuple of (key_name, confidence).
        """
        sr = sr or self.sample_rate
        
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sr)
            
            # Compute key from chromagram
            chroma_sum = np.sum(chroma, axis=1)
            key_index = np.argmax(chroma_sum)
            
            # Map index to key name (C, C#, D, etc.)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_name = key_names[key_index]
            
            # Calculate confidence based on prominence of the detected key
            max_val = chroma_sum[key_index]
            mean_val = np.mean(chroma_sum)
            confidence = min(1.0, (max_val - mean_val) / mean_val)
            
            return key_name, confidence
        except Exception as e:
            raise RuntimeError(f"Failed to detect key: {e}")
    
    def apply_effects(self, 
                     audio_data: np.ndarray,
                     sr: Optional[int] = None,
                     effects: List[Dict] = []) -> np.ndarray:
        """
        Apply audio effects to audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            effects: List of effect dictionaries, each with a 'type' key and effect-specific parameters.
            
        Returns:
            Processed audio data.
        """
        sr = sr or self.sample_rate
        processed_audio = audio_data.copy()
        
        try:
            for effect in effects:
                effect_type = effect.get('type', '')
                
                if effect_type == 'pitch_shift':
                    n_steps = effect.get('n_steps', 0)
                    processed_audio = librosa.effects.pitch_shift(processed_audio, sr=sr, n_steps=n_steps)
                
                elif effect_type == 'time_stretch':
                    rate = effect.get('rate', 1.0)
                    processed_audio = librosa.effects.time_stretch(processed_audio, rate=rate)
                
                elif effect_type == 'reverb':
                    # Simple convolution reverb using a short impulse response
                    # For better reverb, consider using external libraries like pedalboard
                    reverb_time = effect.get('reverb_time', 1.0)
                    decay = np.exp(-np.linspace(0, 5, int(sr * reverb_time)))
                    impulse_response = np.random.randn(int(sr * reverb_time)) * decay
                    processed_audio = np.convolve(processed_audio, impulse_response, mode='full')[:len(audio_data)]
                
                elif effect_type == 'eq':
                    # Simple EQ using FFT
                    # For better EQ, consider using external libraries like pedalboard
                    low_gain = effect.get('low_gain', 1.0)
                    mid_gain = effect.get('mid_gain', 1.0)
                    high_gain = effect.get('high_gain', 1.0)
                    
                    D = librosa.stft(processed_audio)
                    frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
                    
                    # Apply EQ
                    for i, freq in enumerate(frequencies):
                        if freq < 250:
                            D[i, :] *= low_gain
                        elif freq < 4000:
                            D[i, :] *= mid_gain
                        else:
                            D[i, :] *= high_gain
                    
                    processed_audio = librosa.istft(D)
                
                else:
                    print(f"Warning: Unknown effect type '{effect_type}'")
            
            return processed_audio
        except Exception as e:
            raise RuntimeError(f"Failed to apply effects: {e}")
    
    def mix_audio(self, 
                 audio_data_1: np.ndarray,
                 audio_data_2: np.ndarray,
                 mix_ratio: float = 0.5) -> np.ndarray:
        """
        Mix two audio tracks.
        
        Args:
            audio_data_1: First audio track.
            audio_data_2: Second audio track.
            mix_ratio: Mixing ratio (0.0 to 1.0), representing the weight of the second track.
            
        Returns:
            Mixed audio data.
        """
        try:
            # Ensure both audio tracks have the same length
            min_length = min(len(audio_data_1), len(audio_data_2))
            audio_data_1 = audio_data_1[:min_length]
            audio_data_2 = audio_data_2[:min_length]
            
            # Mix audio tracks
            mixed_audio = (1 - mix_ratio) * audio_data_1 + mix_ratio * audio_data_2
            
            return mixed_audio
        except Exception as e:
            raise RuntimeError(f"Failed to mix audio: {e}")
    
    def fade(self, 
            audio_data: np.ndarray,
            sr: Optional[int] = None,
            fade_in_time: float = 0.01,
            fade_out_time: float = 0.01) -> np.ndarray:
        """
        Apply fade in/out to audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            fade_in_time: Fade in time in seconds.
            fade_out_time: Fade out time in seconds.
            
        Returns:
            Audio data with fades applied.
        """
        sr = sr or self.sample_rate
        
        try:
            # Calculate fade lengths in samples
            fade_in_len = int(fade_in_time * sr)
            fade_out_len = int(fade_out_time * sr)
            
            # Create fade curves
            fade_in_curve = np.linspace(0, 1, fade_in_len)
            fade_out_curve = np.linspace(1, 0, fade_out_len)
            
            # Apply fades
            result = audio_data.copy()
            
            if fade_in_len > 0 and fade_in_len < len(result):
                result[:fade_in_len] *= fade_in_curve
                
            if fade_out_len > 0 and fade_out_len < len(result):
                result[-fade_out_len:] *= fade_out_curve
            
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to apply fade: {e}")
    
    def analyze_audio(self, 
                     audio_data: np.ndarray,
                     sr: Optional[int] = None) -> Dict[str, any]:
        """
        Perform comprehensive analysis of audio data.
        
        Args:
            audio_data: Audio data.
            sr: Sample rate of the audio data. If None, uses the instance sample rate.
            
        Returns:
            Dictionary of analysis results.
        """
        sr = sr or self.sample_rate
        
        try:
            results = {
                "duration": len(audio_data) / sr,
                "sample_rate": sr,
                "num_samples": len(audio_data),
            }
            
            # Basic statistics
            results["peak_amplitude"] = np.max(np.abs(audio_data))
            results["rms_amplitude"] = np.sqrt(np.mean(audio_data**2))
            
            # Spectral features
            spectral = {}
            
            # Spectral centroid (brightness)
            cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral["centroid_mean"] = np.mean(cent)
            spectral["centroid_std"] = np.std(cent)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            spectral["bandwidth_mean"] = np.mean(bandwidth)
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
            spectral["flatness_mean"] = np.mean(flatness)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            spectral["rolloff_mean"] = np.mean(rolloff)
            
            results["spectral"] = spectral
            
            # Rhythmic features
            rhythmic = {}
            
            # Tempo
            tempo, confidence = self.detect_tempo(audio_data, sr)
            rhythmic["tempo"] = tempo
            rhythmic["tempo_confidence"] = confidence
            
            # Onset strength
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
            rhythmic["onset_strength_mean"] = np.mean(onset_env)
            rhythmic["onset_strength_std"] = np.std(onset_env)
            
            results["rhythmic"] = rhythmic
            
            # Harmonic features
            harmonic = {}
            
            # Key detection
            key, key_confidence = self.detect_key(audio_data, sr)
            harmonic["key"] = key
            harmonic["key_confidence"] = key_confidence
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            harmonic["chroma_mean"] = np.mean(chroma, axis=1).tolist()
            
            results["harmonic"] = harmonic
            
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to analyze audio: {e}")