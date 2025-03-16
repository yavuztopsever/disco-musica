"""
Audio Analysis Service Module

This module provides functionality for analyzing audio files using various techniques including
spectrogram analysis, feature extraction, and genre classification using transformer models.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    mfcc: np.ndarray
    spectral_centroid: np.ndarray
    chroma: np.ndarray
    tempo: float
    key: str
    time_signature: str
    duration: float
    rms: np.ndarray
    zero_crossing_rate: np.ndarray

@dataclass
class GenreClassification:
    """Container for genre classification results."""
    genre: str
    confidence: float
    secondary_genres: List[Tuple[str, float]]

class AudioAnalyzer:
    """Service for analyzing audio files and extracting musical features."""
    
    def __init__(self, model_name: str = "facebook/ast-base-patch-16-224"):
        """Initialize the audio analyzer with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use for genre classification
        """
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Load the pre-trained model and feature extractor."""
        try:
            # Load model and feature extractor
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                num_labels=527,  # AudioSet has 527 classes
                ignore_mismatched_sizes=True
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                sampling_rate=16000,  # AST expects 16kHz audio
                do_normalize=True
            )
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            logger.info(f"Successfully loaded model: {self.model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def load_audio(self, file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Load and preprocess an audio file.
        
        Args:
            file_path: Path to the audio file
            sr: Target sampling rate
            
        Returns:
            Tuple of (audio data, sampling rate)
        """
        try:
            # Load audio with librosa
            audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
            
            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            audio = librosa.util.normalize(audio)
            
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
            
    def extract_features(self, audio: np.ndarray, sr: int) -> AudioFeatures:
        """Extract various audio features from the audio data.
        
        Args:
            audio: Audio data array
            sr: Sampling rate
            
        Returns:
            AudioFeatures object containing extracted features
        """
        try:
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Extract key
            key = self._detect_key(audio, sr)
            
            # Extract time signature
            time_signature = self._detect_time_signature(audio, sr)
            
            # Extract duration
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # Extract RMS energy
            rms = librosa.feature.rms(y=audio)
            
            # Extract zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
            
            return AudioFeatures(
                mfcc=mfcc,
                spectral_centroid=spectral_centroid,
                chroma=chroma,
                tempo=tempo,
                key=key,
                time_signature=time_signature,
                duration=duration,
                rms=rms,
                zero_crossing_rate=zero_crossing_rate
            )
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            raise
            
    def classify_genre(self, audio: np.ndarray, sr: int) -> GenreClassification:
        """Classify the genre of the audio using the pre-trained model.
        
        Args:
            audio: Audio data array
            sr: Sampling rate
            
        Returns:
            GenreClassification object containing genre and confidence
        """
        try:
            # Prepare input for the model
            inputs = self.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16000 * 30  # 30 seconds max length
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities for all classes
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Get top 3 genres and their probabilities
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
            # Get genre labels
            primary_genre = self.model.config.id2label[top3_indices[0][0].item()]
            secondary_genres = [
                (self.model.config.id2label[idx.item()], prob.item())
                for idx, prob in zip(top3_indices[0][1:], top3_prob[0][1:])
            ]
            
            return GenreClassification(
                genre=primary_genre,
                confidence=top3_prob[0][0].item(),
                secondary_genres=secondary_genres
            )
        except Exception as e:
            logger.error(f"Failed to classify genre: {e}")
            raise
            
    def _detect_key(self, audio: np.ndarray, sr: int) -> str:
        """Detect the musical key of the audio.
        
        Args:
            audio: Audio data array
            sr: Sampling rate
            
        Returns:
            Detected key as a string
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            
            # Get the key profile
            key_profile = np.mean(chroma, axis=1)
            
            # Define key profiles for major and minor keys
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            # Compare with key profiles
            major_correlation = np.correlate(key_profile, major_profile)
            minor_correlation = np.correlate(key_profile, minor_profile)
            
            # Determine if major or minor
            is_major = major_correlation > minor_correlation
            
            # Get the key index
            key_index = np.argmax(key_profile)
            
            # Map key index to note name
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = notes[key_index]
            
            return f"{key} {'major' if is_major else 'minor'}"
        except Exception as e:
            logger.error(f"Failed to detect key: {e}")
            return "Unknown"
            
    def _detect_time_signature(self, audio: np.ndarray, sr: int) -> str:
        """Detect the time signature of the audio.
        
        Args:
            audio: Audio data array
            sr: Sampling rate
            
        Returns:
            Detected time signature as a string
        """
        try:
            # Get onset strength
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            
            # Get tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Get beat intervals
            beat_intervals = np.diff(beat_frames)
            
            # Count beats per bar (assuming 4/4 time)
            beats_per_bar = 4
            
            # Calculate average beat interval
            avg_beat_interval = np.mean(beat_intervals)
            
            # Calculate bar duration in frames
            bar_duration = beats_per_bar * avg_beat_interval
            
            # Count bars
            num_bars = int(len(audio) / (bar_duration * sr))
            
            return f"{beats_per_bar}/4"
        except Exception as e:
            logger.error(f"Failed to detect time signature: {e}")
            return "Unknown"
            
    def analyze_file(self, file_path: str) -> Dict:
        """Perform complete analysis of an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            
            # Extract features
            features = self.extract_features(audio, sr)
            
            # Classify genre
            genre_classification = self.classify_genre(audio, sr)
            
            # Combine results
            analysis_results = {
                "features": {
                    "mfcc": features.mfcc.tolist(),
                    "spectral_centroid": features.spectral_centroid.tolist(),
                    "chroma": features.chroma.tolist(),
                    "tempo": features.tempo,
                    "key": features.key,
                    "time_signature": features.time_signature,
                    "duration": features.duration,
                    "rms": features.rms.tolist(),
                    "zero_crossing_rate": features.zero_crossing_rate.tolist()
                },
                "genre": {
                    "primary": genre_classification.genre,
                    "confidence": genre_classification.confidence,
                    "secondary": genre_classification.secondary_genres
                }
            }
            
            return analysis_results
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            raise 