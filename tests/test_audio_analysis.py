"""Tests for the audio analysis module."""

import os
import pytest
import numpy as np
from modules.services.audio_analysis import AudioAnalyzer, AudioFeatures, GenreClassification

@pytest.fixture
def audio_analyzer():
    """Create an AudioAnalyzer instance for testing."""
    return AudioAnalyzer()

@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    # Create a simple sine wave as test audio
    duration = 1.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save as WAV file
    file_path = tmp_path / "test_audio.wav"
    import soundfile as sf
    sf.write(file_path, audio, sample_rate)
    
    return str(file_path)

def test_audio_analyzer_initialization(audio_analyzer):
    """Test that the AudioAnalyzer initializes correctly."""
    assert audio_analyzer.model is not None
    assert audio_analyzer.feature_extractor is not None
    assert audio_analyzer.model_name == "facebook/ast-base-patch-16-224"

def test_load_audio(audio_analyzer, sample_audio_file):
    """Test loading an audio file."""
    audio, sr = audio_analyzer.load_audio(sample_audio_file)
    assert isinstance(audio, np.ndarray)
    assert sr == 16000
    assert len(audio) > 0

def test_extract_features(audio_analyzer, sample_audio_file):
    """Test feature extraction from audio."""
    audio, sr = audio_analyzer.load_audio(sample_audio_file)
    features = audio_analyzer.extract_features(audio, sr)
    
    assert isinstance(features, AudioFeatures)
    assert isinstance(features.mfcc, np.ndarray)
    assert isinstance(features.spectral_centroid, np.ndarray)
    assert isinstance(features.chroma, np.ndarray)
    assert isinstance(features.tempo, float)
    assert isinstance(features.key, str)
    assert isinstance(features.time_signature, str)
    assert isinstance(features.duration, float)
    assert isinstance(features.rms, np.ndarray)
    assert isinstance(features.zero_crossing_rate, np.ndarray)

def test_classify_genre(audio_analyzer, sample_audio_file):
    """Test genre classification."""
    audio, sr = audio_analyzer.load_audio(sample_audio_file)
    classification = audio_analyzer.classify_genre(audio, sr)
    
    assert isinstance(classification, GenreClassification)
    assert isinstance(classification.genre, str)
    assert isinstance(classification.confidence, float)
    assert isinstance(classification.secondary_genres, list)
    assert len(classification.secondary_genres) == 2  # Should return top 3 genres

def test_analyze_file(audio_analyzer, sample_audio_file):
    """Test complete file analysis."""
    results = audio_analyzer.analyze_file(sample_audio_file)
    
    assert isinstance(results, dict)
    assert "features" in results
    assert "genre" in results
    
    features = results["features"]
    assert "mfcc" in features
    assert "spectral_centroid" in features
    assert "chroma" in features
    assert "tempo" in features
    assert "key" in features
    assert "time_signature" in features
    assert "duration" in features
    assert "rms" in features
    assert "zero_crossing_rate" in features
    
    genre = results["genre"]
    assert "primary" in genre
    assert "confidence" in genre
    assert "secondary" in genre

def test_invalid_file_path(audio_analyzer):
    """Test handling of invalid file path."""
    with pytest.raises(Exception):
        audio_analyzer.analyze_file("nonexistent_file.wav")

def test_detect_key(audio_analyzer, sample_audio_file):
    """Test key detection."""
    audio, sr = audio_analyzer.load_audio(sample_audio_file)
    key = audio_analyzer._detect_key(audio, sr)
    assert isinstance(key, str)
    assert "major" in key or "minor" in key

def test_detect_time_signature(audio_analyzer, sample_audio_file):
    """Test time signature detection."""
    audio, sr = audio_analyzer.load_audio(sample_audio_file)
    time_signature = audio_analyzer._detect_time_signature(audio, sr)
    assert isinstance(time_signature, str)
    assert "/" in time_signature 