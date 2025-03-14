"""
Tests for the Data Ingestion Module.
"""

import os
import pytest
from pathlib import Path
import tempfile
import shutil

from modules.data_ingestion import DataIngestionModule


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def data_ingestion(temp_dir):
    """Create a DataIngestionModule instance for testing."""
    return DataIngestionModule(data_dir=temp_dir)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    from pydub import AudioSegment
    
    # Create a sample audio file
    audio_path = Path(temp_dir) / "sample.wav"
    sample_rate = 44100
    duration_ms = 1000  # 1 second
    
    # Generate a silent audio segment
    audio = AudioSegment.silent(duration=duration_ms)
    
    # Export the audio file
    audio.export(audio_path, format="wav")
    
    return audio_path


def test_init(data_ingestion, temp_dir):
    """Test initialization of DataIngestionModule."""
    assert data_ingestion.data_dir == Path(temp_dir)
    assert data_ingestion.raw_dir == Path(temp_dir) / "raw"
    assert data_ingestion.processed_dir == Path(temp_dir) / "processed"
    assert data_ingestion.datasets_dir == Path(temp_dir) / "datasets"
    
    # Check if directories were created
    assert os.path.exists(data_ingestion.raw_dir)
    assert os.path.exists(data_ingestion.processed_dir)
    assert os.path.exists(data_ingestion.datasets_dir)


def test_ingest_audio(data_ingestion, sample_audio_file):
    """Test ingesting an audio file."""
    # Ingest the sample audio file
    target_path = data_ingestion.ingest_audio(sample_audio_file)
    
    # Check if the file was ingested
    assert os.path.exists(target_path)
    assert target_path.parent == data_ingestion.raw_dir / "audio"


def test_ingest_audio_with_target_dir(data_ingestion, sample_audio_file, temp_dir):
    """Test ingesting an audio file with a specific target directory."""
    # Create a target directory
    target_dir = Path(temp_dir) / "custom_target"
    os.makedirs(target_dir, exist_ok=True)
    
    # Ingest the sample audio file
    target_path = data_ingestion.ingest_audio(sample_audio_file, target_dir=str(target_dir))
    
    # Check if the file was ingested
    assert os.path.exists(target_path)
    assert target_path.parent == target_dir