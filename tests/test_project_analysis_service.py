"""
Test module for ProjectAnalysisService.

This module contains test cases for the ProjectAnalysisService class,
which provides functionality for analyzing both song and DAW files.
"""

import os
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from modules.services.project_analysis_service import ProjectAnalysisService
from modules.core.resources import ProjectResource, TrackResource
from modules.core.exceptions import ResourceNotFoundError, ProcessingError

@pytest.fixture
def mock_project_service():
    """Create a mock project service."""
    service = Mock()
    service.get_project = AsyncMock()
    service.get_track = AsyncMock()
    service.update_project = AsyncMock()
    return service

@pytest.fixture
def mock_audio_analyzer():
    """Create a mock audio analyzer."""
    with patch('modules.services.project_analysis_service.AudioAnalyzer') as mock:
        analyzer = Mock()
        analyzer.load_audio = Mock(return_value=(np.zeros(44100), 44100))
        analyzer.extract_features = Mock()
        analyzer.classify_genre = Mock()
        mock.return_value = analyzer
        yield analyzer

@pytest.fixture
def sample_project():
    """Create a sample project resource."""
    project = ProjectResource(
        resource_id="test_project",
        basic_info={
            "name": "Test Project",
            "description": "A test project",
            "daw_type": "Logic Pro",
            "project_directory": "/tmp/test_project"
        },
        tracks=["track1", "track2"],
        musical_properties={
            "key": "C",
            "time_signature": "4/4",
            "bpm": 120,
            "genre": ["Pop"],
            "emotion": ["Happy"],
            "style_tags": ["Upbeat"]
        },
        analysis_results=[]
    )
    return project

@pytest.fixture
def sample_audio_track():
    """Create a sample audio track resource."""
    track = TrackResource(
        resource_id="track1",
        basic_info={
            "name": "Audio Track 1",
            "type": "audio",
            "role": "Lead",
            "file_path": "/tmp/test_project/audio1.wav"
        },
        audio_properties={
            "duration_seconds": 180.0,
            "sample_rate": 44100,
            "channels": 2
        },
        midi_properties={}
    )
    return track

@pytest.fixture
def sample_midi_track():
    """Create a sample MIDI track resource."""
    track = TrackResource(
        resource_id="track2",
        basic_info={
            "name": "MIDI Track 1",
            "type": "midi",
            "role": "Bass",
            "file_path": "/tmp/test_project/midi1.mid"
        },
        audio_properties={
            "duration_seconds": 180.0
        },
        midi_properties={
            "note_count": 1000,
            "velocity_range": [0, 127],
            "pitch_range": [36, 84]
        }
    )
    return track

@pytest.mark.asyncio
async def test_project_analysis_service_initialization(mock_project_service):
    """Test ProjectAnalysisService initialization."""
    service = ProjectAnalysisService(mock_project_service)
    assert service.project_service == mock_project_service
    assert service.audio_analyzer is not None

@pytest.mark.asyncio
async def test_analyze_project_success(
    mock_project_service,
    mock_audio_analyzer,
    sample_project,
    sample_audio_track,
    sample_midi_track
):
    """Test successful project analysis."""
    # Setup mocks
    mock_project_service.get_project.return_value = sample_project
    mock_project_service.get_track.side_effect = [sample_audio_track, sample_midi_track]
    
    # Setup audio analyzer mocks
    mock_audio_analyzer.extract_features.return_value = Mock(
        tempo=120,
        key="C",
        time_signature="4/4",
        rms=np.array([0.5]),
        zero_crossing_rate=np.array([0.1])
    )
    mock_audio_analyzer.classify_genre.return_value = Mock(
        genre="Pop",
        confidence=0.95,
        secondary_genres=["Rock", "Electronic"]
    )
    
    # Create service and analyze project
    service = ProjectAnalysisService(mock_project_service)
    results = await service.analyze_project("test_project")
    
    # Verify results structure
    assert "project_id" in results
    assert "timestamp" in results
    assert "structure_analysis" in results
    assert "audio_analysis" in results
    assert "midi_analysis" in results
    assert "musical_analysis" in results
    
    # Verify structure analysis
    structure = results["structure_analysis"]
    assert structure["total_tracks"] == 2
    assert len(structure["audio_tracks"]) == 1
    assert len(structure["midi_tracks"]) == 1
    
    # Verify audio analysis
    audio = results["audio_analysis"]
    assert len(audio["tracks"]) == 1
    assert "overall_metrics" in audio
    
    # Verify MIDI analysis
    midi = results["midi_analysis"]
    assert len(midi["tracks"]) == 1
    assert "overall_metrics" in midi
    
    # Verify musical analysis
    musical = results["musical_analysis"]
    assert "harmony" in musical
    assert "style" in musical
    assert "arrangement" in musical
    
    # Verify project update was called
    mock_project_service.update_project.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_project_not_found(mock_project_service):
    """Test project analysis with non-existent project."""
    mock_project_service.get_project.return_value = None
    
    service = ProjectAnalysisService(mock_project_service)
    with pytest.raises(ResourceNotFoundError):
        await service.analyze_project("non_existent_project")

@pytest.mark.asyncio
async def test_analyze_project_processing_error(
    mock_project_service,
    sample_project,
    sample_audio_track
):
    """Test project analysis with processing error."""
    mock_project_service.get_project.return_value = sample_project
    mock_project_service.get_track.return_value = sample_audio_track
    
    service = ProjectAnalysisService(mock_project_service)
    with patch('modules.services.project_analysis_service.AudioAnalyzer') as mock:
        mock.return_value.load_audio.side_effect = Exception("Audio loading failed")
        
        with pytest.raises(ProcessingError):
            await service.analyze_project("test_project")

@pytest.mark.asyncio
async def test_analyze_project_structure(
    mock_project_service,
    sample_project,
    sample_audio_track,
    sample_midi_track
):
    """Test project structure analysis."""
    mock_project_service.get_track.side_effect = [sample_audio_track, sample_midi_track]
    
    service = ProjectAnalysisService(mock_project_service)
    structure = await service._analyze_project_structure(sample_project)
    
    assert structure["total_tracks"] == 2
    assert len(structure["audio_tracks"]) == 1
    assert len(structure["midi_tracks"]) == 1
    assert "project_metadata" in structure

@pytest.mark.asyncio
async def test_analyze_audio_tracks(
    mock_project_service,
    mock_audio_analyzer,
    sample_project,
    sample_audio_track
):
    """Test audio tracks analysis."""
    mock_project_service.get_track.return_value = sample_audio_track
    mock_audio_analyzer.extract_features.return_value = Mock(
        tempo=120,
        key="C",
        time_signature="4/4",
        rms=np.array([0.5]),
        zero_crossing_rate=np.array([0.1])
    )
    mock_audio_analyzer.classify_genre.return_value = Mock(
        genre="Pop",
        confidence=0.95,
        secondary_genres=["Rock", "Electronic"]
    )
    
    service = ProjectAnalysisService(mock_project_service)
    audio_analysis = await service._analyze_audio_tracks(sample_project)
    
    assert len(audio_analysis["tracks"]) == 1
    assert "overall_metrics" in audio_analysis
    assert "features" in audio_analysis["tracks"][0]
    assert "genre" in audio_analysis["tracks"][0]

@pytest.mark.asyncio
async def test_analyze_midi_tracks(
    mock_project_service,
    sample_project,
    sample_midi_track
):
    """Test MIDI tracks analysis."""
    mock_project_service.get_track.return_value = sample_midi_track
    
    service = ProjectAnalysisService(mock_project_service)
    midi_analysis = await service._analyze_midi_tracks(sample_project)
    
    assert len(midi_analysis["tracks"]) == 1
    assert "overall_metrics" in midi_analysis
    assert "metrics" in midi_analysis["tracks"][0]

@pytest.mark.asyncio
async def test_analyze_musical_content(
    mock_project_service,
    sample_project,
    sample_audio_track,
    sample_midi_track
):
    """Test musical content analysis."""
    mock_project_service.get_track.side_effect = [sample_audio_track, sample_midi_track]
    
    service = ProjectAnalysisService(mock_project_service)
    musical_analysis = await service._analyze_musical_content(sample_project)
    
    assert "harmony" in musical_analysis
    assert "style" in musical_analysis
    assert "arrangement" in musical_analysis
    assert len(musical_analysis["arrangement"]["instrumentation"]) == 2

@pytest.mark.asyncio
async def test_update_project_with_analysis(
    mock_project_service,
    sample_project
):
    """Test updating project with analysis results."""
    analysis_results = {
        "musical_analysis": {
            "harmony": {
                "key": "D",
                "time_signature": "3/4",
                "bpm": 140
            },
            "style": {
                "genre": ["Rock"],
                "emotion": ["Energetic"],
                "style_tags": ["Heavy"]
            }
        }
    }
    
    service = ProjectAnalysisService(mock_project_service)
    await service._update_project_with_analysis(sample_project, analysis_results)
    
    # Verify project update was called
    mock_project_service.update_project.assert_called_once()
    
    # Verify musical properties were updated
    updated_project = mock_project_service.update_project.call_args[0][1]
    assert updated_project.musical_properties["key"] == "D"
    assert updated_project.musical_properties["time_signature"] == "3/4"
    assert updated_project.musical_properties["bpm"] == 140
    assert updated_project.musical_properties["genre"] == ["Rock"]
    assert updated_project.musical_properties["emotion"] == ["Energetic"]
    assert updated_project.musical_properties["style_tags"] == ["Heavy"] 