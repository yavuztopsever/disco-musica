"""
Project Analysis Service Module

This module provides functionality for analyzing both song and DAW files to create detailed profiles.
It combines audio analysis with project management to provide comprehensive project insights.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import json

from ..core.resources import ProjectResource, TrackResource
from ..core.exceptions import ResourceNotFoundError, ProcessingError
from .audio_analysis import AudioAnalyzer, AudioFeatures, GenreClassification

logger = logging.getLogger(__name__)

class ProjectAnalysisService:
    """Service for analyzing projects and creating detailed profiles."""
    
    def __init__(self, project_service: 'ProjectService'):
        """Initialize the project analysis service.
        
        Args:
            project_service: Project service instance for project management.
        """
        self.project_service = project_service
        self.audio_analyzer = AudioAnalyzer()
        
    async def analyze_project(self, project_id: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a project.
        
        Args:
            project_id: Project ID.
            
        Returns:
            Dictionary containing analysis results.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If analysis fails.
        """
        try:
            # Get project
            project = await self.project_service.get_project(project_id)
            if not project:
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            # Analyze project structure
            structure_analysis = await self._analyze_project_structure(project)
            
            # Analyze audio tracks
            audio_analysis = await self._analyze_audio_tracks(project)
            
            # Analyze MIDI tracks
            midi_analysis = await self._analyze_midi_tracks(project)
            
            # Perform musical analysis
            musical_analysis = await self._analyze_musical_content(project)
            
            # Combine all analyses
            analysis_results = {
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat(),
                "structure_analysis": structure_analysis,
                "audio_analysis": audio_analysis,
                "midi_analysis": midi_analysis,
                "musical_analysis": musical_analysis
            }
            
            # Update project with analysis results
            await self._update_project_with_analysis(project, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze project {project_id}: {e}")
            raise ProcessingError(f"Project analysis failed: {str(e)}")
            
    async def _analyze_project_structure(self, project: ProjectResource) -> Dict[str, Any]:
        """Analyze project structure and organization.
        
        Args:
            project: Project resource.
            
        Returns:
            Dictionary containing structure analysis results.
        """
        try:
            # Get project directory
            project_dir = project.basic_info.get("project_directory")
            if not project_dir or not os.path.exists(project_dir):
                raise ProcessingError(f"Project directory not found: {project_dir}")
                
            # Analyze directory structure
            structure = {
                "total_tracks": len(project.tracks),
                "audio_tracks": [],
                "midi_tracks": [],
                "project_metadata": {
                    "name": project.basic_info.get("name"),
                    "description": project.basic_info.get("description"),
                    "daw_type": project.basic_info.get("daw_type"),
                    "created_at": project.creation_timestamp.isoformat(),
                    "modified_at": project.modification_timestamp.isoformat()
                }
            }
            
            # Categorize tracks
            for track_id in project.tracks:
                track = await self.project_service.get_track(track_id)
                if not track:
                    continue
                    
                track_type = track.basic_info.get("type")
                if track_type == "audio":
                    structure["audio_tracks"].append({
                        "id": track_id,
                        "name": track.basic_info.get("name"),
                        "duration": track.audio_properties.get("duration_seconds"),
                        "sample_rate": track.audio_properties.get("sample_rate"),
                        "channels": track.audio_properties.get("channels")
                    })
                elif track_type == "midi":
                    structure["midi_tracks"].append({
                        "id": track_id,
                        "name": track.basic_info.get("name"),
                        "note_count": track.midi_properties.get("note_count"),
                        "pitch_range": track.midi_properties.get("pitch_range")
                    })
                    
            return structure
            
        except Exception as e:
            logger.error(f"Failed to analyze project structure: {e}")
            raise ProcessingError(f"Structure analysis failed: {str(e)}")
            
    async def _analyze_audio_tracks(self, project: ProjectResource) -> Dict[str, Any]:
        """Analyze audio tracks in the project.
        
        Args:
            project: Project resource.
            
        Returns:
            Dictionary containing audio analysis results.
        """
        try:
            audio_analysis = {
                "tracks": [],
                "overall_metrics": {
                    "total_duration": 0.0,
                    "average_loudness": 0.0,
                    "dynamic_range": 0.0
                }
            }
            
            total_loudness = 0.0
            total_dynamic_range = 0.0
            track_count = 0
            
            for track_id in project.tracks:
                track = await self.project_service.get_track(track_id)
                if not track or track.basic_info.get("type") != "audio":
                    continue
                    
                # Get audio file path
                file_path = track.basic_info.get("file_path")
                if not file_path or not os.path.exists(file_path):
                    continue
                    
                # Load and analyze audio
                audio, sr = self.audio_analyzer.load_audio(file_path)
                features = self.audio_analyzer.extract_features(audio, sr)
                genre = self.audio_analyzer.classify_genre(audio, sr)
                
                # Calculate track metrics
                duration = librosa.get_duration(y=audio, sr=sr)
                rms = np.mean(features.rms)
                dynamic_range = np.max(features.rms) - np.min(features.rms)
                
                # Update overall metrics
                total_loudness += rms
                total_dynamic_range += dynamic_range
                track_count += 1
                
                # Store track analysis
                audio_analysis["tracks"].append({
                    "id": track_id,
                    "name": track.basic_info.get("name"),
                    "duration": duration,
                    "features": {
                        "tempo": features.tempo,
                        "key": features.key,
                        "time_signature": features.time_signature,
                        "rms": float(rms),
                        "dynamic_range": float(dynamic_range),
                        "zero_crossing_rate": float(np.mean(features.zero_crossing_rate))
                    },
                    "genre": {
                        "primary": genre.genre,
                        "confidence": genre.confidence,
                        "secondary": genre.secondary_genres
                    }
                })
                
            # Calculate overall metrics
            if track_count > 0:
                audio_analysis["overall_metrics"].update({
                    "average_loudness": float(total_loudness / track_count),
                    "dynamic_range": float(total_dynamic_range / track_count)
                })
                
            return audio_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze audio tracks: {e}")
            raise ProcessingError(f"Audio analysis failed: {str(e)}")
            
    async def _analyze_midi_tracks(self, project: ProjectResource) -> Dict[str, Any]:
        """Analyze MIDI tracks in the project.
        
        Args:
            project: Project resource.
            
        Returns:
            Dictionary containing MIDI analysis results.
        """
        try:
            midi_analysis = {
                "tracks": [],
                "overall_metrics": {
                    "total_notes": 0,
                    "average_velocity": 0.0,
                    "note_density": 0.0
                }
            }
            
            total_notes = 0
            total_velocity = 0.0
            total_duration = 0.0
            
            for track_id in project.tracks:
                track = await self.project_service.get_track(track_id)
                if not track or track.basic_info.get("type") != "midi":
                    continue
                    
                # Get MIDI properties
                midi_props = track.midi_properties
                note_count = midi_props.get("note_count", 0)
                velocity_range = midi_props.get("velocity_range", [0, 0])
                pitch_range = midi_props.get("pitch_range", [0, 0])
                
                # Calculate track metrics
                avg_velocity = sum(velocity_range) / 2
                note_density = note_count / (track.audio_properties.get("duration_seconds", 1))
                
                # Update overall metrics
                total_notes += note_count
                total_velocity += avg_velocity
                total_duration += track.audio_properties.get("duration_seconds", 0)
                
                # Store track analysis
                midi_analysis["tracks"].append({
                    "id": track_id,
                    "name": track.basic_info.get("name"),
                    "metrics": {
                        "note_count": note_count,
                        "average_velocity": float(avg_velocity),
                        "note_density": float(note_density),
                        "pitch_range": pitch_range
                    }
                })
                
            # Calculate overall metrics
            if total_duration > 0:
                midi_analysis["overall_metrics"].update({
                    "total_notes": total_notes,
                    "average_velocity": float(total_velocity / len(midi_analysis["tracks"])),
                    "note_density": float(total_notes / total_duration)
                })
                
            return midi_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze MIDI tracks: {e}")
            raise ProcessingError(f"MIDI analysis failed: {str(e)}")
            
    async def _analyze_musical_content(self, project: ProjectResource) -> Dict[str, Any]:
        """Analyze musical content across all tracks.
        
        Args:
            project: Project resource.
            
        Returns:
            Dictionary containing musical analysis results.
        """
        try:
            musical_analysis = {
                "harmony": {
                    "key": project.musical_properties.get("key"),
                    "time_signature": project.musical_properties.get("time_signature"),
                    "bpm": project.musical_properties.get("bpm")
                },
                "style": {
                    "genre": project.musical_properties.get("genre", []),
                    "emotion": project.musical_properties.get("emotion", []),
                    "style_tags": project.musical_properties.get("style_tags", [])
                },
                "arrangement": {
                    "track_count": len(project.tracks),
                    "instrumentation": [],
                    "section_analysis": {}
                }
            }
            
            # Analyze instrumentation
            for track_id in project.tracks:
                track = await self.project_service.get_track(track_id)
                if not track:
                    continue
                    
                musical_analysis["arrangement"]["instrumentation"].append({
                    "id": track_id,
                    "name": track.basic_info.get("name"),
                    "type": track.basic_info.get("type"),
                    "role": track.basic_info.get("role")
                })
                
            return musical_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze musical content: {e}")
            raise ProcessingError(f"Musical analysis failed: {str(e)}")
            
    async def _update_project_with_analysis(
        self,
        project: ProjectResource,
        analysis_results: Dict[str, Any]
    ) -> None:
        """Update project with analysis results.
        
        Args:
            project: Project resource.
            analysis_results: Analysis results dictionary.
        """
        try:
            # Update project musical properties
            project.musical_properties.update({
                "key": analysis_results["musical_analysis"]["harmony"]["key"],
                "time_signature": analysis_results["musical_analysis"]["harmony"]["time_signature"],
                "bpm": analysis_results["musical_analysis"]["harmony"]["bpm"],
                "genre": analysis_results["musical_analysis"]["style"]["genre"],
                "emotion": analysis_results["musical_analysis"]["style"]["emotion"],
                "style_tags": analysis_results["musical_analysis"]["style"]["style_tags"]
            })
            
            # Add analysis results
            analysis_id = f"analysis_{project.resource_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            project.analysis_results.append(analysis_id)
            
            # Save analysis results
            analysis_path = os.path.join(
                project.basic_info.get("project_directory", ""),
                "analysis",
                f"{analysis_id}.json"
            )
            os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
            
            with open(analysis_path, "w") as f:
                json.dump(analysis_results, f, indent=2)
                
            # Update project
            await self.project_service.update_project(project.resource_id, project)
            
        except Exception as e:
            logger.error(f"Failed to update project with analysis: {e}")
            raise ProcessingError(f"Project update failed: {str(e)}") 