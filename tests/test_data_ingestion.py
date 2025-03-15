"""Tests for the data ingestion module."""

import os
import pytest
from pathlib import Path
from disco_musica.modules.data_ingestion import DataIngestionModule

@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path / "test_data"

@pytest.fixture
def ingestion_module(data_dir):
    """Create a DataIngestionModule instance for testing."""
    return DataIngestionModule(data_dir=data_dir)

def test_initialization(ingestion_module, data_dir):
    """Test that the module initializes correctly."""
    assert ingestion_module.data_dir == data_dir
    assert data_dir.exists()

def test_audio_file_ingestion(ingestion_module, data_dir):
    """Test ingesting an audio file."""
    # Create a dummy audio file
    audio_file = data_dir / "test.wav"
    audio_file.touch()
    
    # Test ingestion
    result = ingestion_module.ingest_audio(audio_file)
    assert result is not None
    assert result.exists()

def test_midi_file_ingestion(ingestion_module, data_dir):
    """Test ingesting a MIDI file."""
    # Create a dummy MIDI file
    midi_file = data_dir / "test.mid"
    midi_file.touch()
    
    # Test ingestion
    result = ingestion_module.ingest_midi(midi_file)
    assert result is not None
    assert result.exists()

def test_invalid_file_type(ingestion_module, data_dir):
    """Test handling of invalid file types."""
    # Create a file with invalid extension
    invalid_file = data_dir / "test.txt"
    invalid_file.touch()
    
    # Test that invalid file type raises ValueError
    with pytest.raises(ValueError):
        ingestion_module.ingest_audio(invalid_file)

def test_nonexistent_file(ingestion_module):
    """Test handling of nonexistent files."""
    nonexistent_file = Path("nonexistent.wav")
    
    # Test that nonexistent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ingestion_module.ingest_audio(nonexistent_file)

def test_directory_creation(ingestion_module, data_dir):
    """Test that necessary directories are created."""
    # Check that subdirectories exist
    assert (data_dir / "audio").exists()
    assert (data_dir / "midi").exists()
    assert (data_dir / "processed").exists()

def test_file_validation(ingestion_module, data_dir):
    """Test file validation functionality."""
    # Create a file with invalid content
    invalid_audio = data_dir / "invalid.wav"
    invalid_audio.write_bytes(b"invalid content")
    
    # Test that invalid file content raises ValueError
    with pytest.raises(ValueError):
        ingestion_module.validate_audio_file(invalid_audio)

def test_batch_processing(ingestion_module, data_dir):
    """Test batch processing of multiple files."""
    # Create multiple test files
    files = [
        data_dir / f"test_{i}.wav" for i in range(3)
    ]
    for file in files:
        file.touch()
    
    # Test batch processing
    results = ingestion_module.process_batch(files)
    assert len(results) == len(files)
    assert all(result.exists() for result in results)

def test_error_handling(ingestion_module, data_dir):
    """Test error handling during ingestion."""
    # Create a file with invalid permissions
    restricted_file = data_dir / "restricted.wav"
    restricted_file.touch()
    os.chmod(restricted_file, 0o000)
    
    # Test that permission error is handled
    with pytest.raises(PermissionError):
        ingestion_module.ingest_audio(restricted_file)
    
    # Restore permissions for cleanup
    os.chmod(restricted_file, 0o644)