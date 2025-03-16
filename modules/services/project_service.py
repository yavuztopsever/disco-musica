"""Project Service for managing music generation projects."""

import os
import json
import shutil
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import uuid
import subprocess
import hashlib

from ..core.resources.base_resources import BaseResource
from ..core.resources.model_resource import ModelResource
from ..core.resources.track_resource import TrackResource
from ..core.resources.generation_resource import GenerationResource
from ..core.exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)
from ..services.project_analysis_service import ProjectAnalysisService


class ProjectService:
    """Service for managing music generation projects.
    
    This class provides functionality for managing projects, including
    resource management, configuration handling, and output organization.
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        resource_manager: Any,  # ResourceManager instance
        model_service: Any,  # ModelService instance
        generation_service: Any,  # GenerationService instance
        output_service: Any,  # OutputService instance
        project_repository: 'ProjectRepository',
        track_repository: 'TrackRepository',
        audio_processor: 'AudioProcessor',
        midi_processor: 'MidiProcessor'
    ):
        """Initialize the project service.
        
        Args:
            base_path: Base path for project storage.
            resource_manager: ResourceManager instance.
            model_service: ModelService instance.
            generation_service: GenerationService instance.
            output_service: OutputService instance.
            project_repository: ProjectRepository instance.
            track_repository: TrackRepository instance.
            audio_processor: AudioProcessor instance.
            midi_processor: MidiProcessor instance.
        """
        self.base_path = Path(base_path)
        self.resource_manager = resource_manager
        self.model_service = model_service
        self.generation_service = generation_service
        self.output_service = output_service
        self.project_repository = project_repository
        self.track_repository = track_repository
        self.audio_processor = audio_processor
        self.midi_processor = midi_processor
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure project directories exist
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """Ensure required project directories exist."""
        directories = [
            self.base_path,
            self.base_path / "projects",
            self.base_path / "shared_models",
            self.base_path / "shared_resources"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new project.
        
        Args:
            name: Project name.
            description: Optional project description.
            metadata: Optional project metadata.
            
        Returns:
            Project information dictionary.
            
        Raises:
            ValidationError: If project parameters are invalid.
            ProcessingError: If project creation fails.
        """
        try:
            # Validate name
            if not name or not name.strip():
                raise ValidationError("Project name cannot be empty")
                
            # Generate project ID and path
            project_id = str(uuid.uuid4())
            project_path = self.base_path / "projects" / project_id
            
            # Create project structure
            project_dirs = [
                project_path,
                project_path / "models",
                project_path / "tracks",
                project_path / "generations",
                project_path / "outputs",
                project_path / "config"
            ]
            
            for directory in project_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                
            # Create project config
            config = {
                "id": project_id,
                "name": name,
                "description": description or "",
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            # Save config
            config_path = project_path / "config" / "project.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            return config
            
        except Exception as e:
            raise ProcessingError(f"Error creating project: {e}")
            
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project information.
        
        Args:
            project_id: Project ID.
            
        Returns:
            Project information dictionary.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If loading project fails.
        """
        try:
            project_path = self.base_path / "projects" / project_id
            config_path = project_path / "config" / "project.json"
            
            if not config_path.exists():
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Add additional info
            config.update({
                "path": str(project_path),
                "size": self._get_directory_size(project_path),
                "resource_counts": self._get_resource_counts(project_id)
            })
            
            return config
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error loading project: {e}")
            
    def list_projects(
        self,
        status: Optional[str] = None,
        sort_by: str = "updated_at",
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """List available projects.
        
        Args:
            status: Optional status filter.
            sort_by: Field to sort by.
            reverse: Whether to reverse sort order.
            
        Returns:
            List of project information dictionaries.
        """
        projects = []
        projects_dir = self.base_path / "projects"
        
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                try:
                    project = self.get_project(project_dir.name)
                    if not status or project["status"] == status:
                        projects.append(project)
                except Exception as e:
                    self.logger.warning(f"Error loading project {project_dir.name}: {e}")
                    
        # Sort projects
        if sort_by in projects[0] if projects else False:
            projects.sort(key=lambda x: x[sort_by], reverse=reverse)
            
        return projects
        
    def update_project(
        self,
        project_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update project information.
        
        Args:
            project_id: Project ID.
            updates: Dictionary of updates.
            
        Returns:
            Updated project information.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ValidationError: If updates are invalid.
            ProcessingError: If update fails.
        """
        try:
            # Get current config
            project = self.get_project(project_id)
            
            # Validate updates
            invalid_keys = set(updates.keys()) - {
                "name", "description", "metadata", "status"
            }
            if invalid_keys:
                raise ValidationError(f"Invalid update keys: {invalid_keys}")
                
            # Update config
            project.update(updates)
            project["updated_at"] = datetime.utcnow().isoformat()
            
            # Save config
            config_path = (
                self.base_path / "projects" / project_id / 
                "config" / "project.json"
            )
            with open(config_path, "w") as f:
                # Only save persistent fields
                config = {
                    k: v for k, v in project.items()
                    if k not in ["path", "size", "resource_counts"]
                }
                json.dump(config, f, indent=2)
                
            return project
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error updating project: {e}")
            
    def delete_project(
        self,
        project_id: str,
        delete_resources: bool = False
    ) -> None:
        """Delete a project.
        
        Args:
            project_id: Project ID.
            delete_resources: Whether to delete project resources.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If deletion fails.
        """
        try:
            project_path = self.base_path / "projects" / project_id
            
            if not project_path.exists():
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            if delete_resources:
                # Delete all project resources
                resource_types = ["models", "tracks", "generations"]
                for resource_type in resource_types:
                    resources = self.list_project_resources(
                        project_id,
                        resource_type
                    )
                    for resource in resources:
                        self.resource_manager.delete_resource(resource["id"])
                        
            # Delete project directory
            shutil.rmtree(project_path)
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error deleting project: {e}")
            
    def list_project_resources(
        self,
        project_id: str,
        resource_type: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List project resources.
        
        Args:
            project_id: Project ID.
            resource_type: Type of resource ("models", "tracks", "generations").
            status: Optional status filter.
            
        Returns:
            List of resource information dictionaries.
            
        Raises:
            ValidationError: If resource type is invalid.
            ResourceNotFoundError: If project not found.
        """
        if resource_type not in ["models", "tracks", "generations"]:
            raise ValidationError(f"Invalid resource type: {resource_type}")
            
        # Verify project exists
        if not (self.base_path / "projects" / project_id).exists():
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Get resources from resource manager
        resources = self.resource_manager.list_resources(
            resource_type=resource_type,
            filters={"project_id": project_id}
        )
        
        # Filter by status if specified
        if status:
            resources = [r for r in resources if r["status"] == status]
            
        return resources
        
    def get_project_resource(
        self,
        project_id: str,
        resource_id: str
    ) -> Dict[str, Any]:
        """Get project resource information.
        
        Args:
            project_id: Project ID.
            resource_id: Resource ID.
            
        Returns:
            Resource information dictionary.
            
        Raises:
            ResourceNotFoundError: If project or resource not found.
        """
        # Verify project exists
        if not (self.base_path / "projects" / project_id).exists():
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Get resource from resource manager
        resource = self.resource_manager.get_resource(resource_id)
        
        # Verify resource belongs to project
        if resource["project_id"] != project_id:
            raise ResourceNotFoundError(
                f"Resource {resource_id} not found in project {project_id}"
            )
            
        return resource
        
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total
        
    def _get_resource_counts(self, project_id: str) -> Dict[str, int]:
        """Get counts of different resource types in project."""
        counts = {}
        resource_types = ["models", "tracks", "generations"]
        
        for resource_type in resource_types:
            try:
                resources = self.list_project_resources(
                    project_id,
                    resource_type
                )
                counts[resource_type] = len(resources)
            except Exception:
                counts[resource_type] = 0
                
        return counts
        
    def get_memory_usage(self) -> float:
        """Get service memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        # Get size of all project directories
        total = 0
        projects_dir = self.base_path / "projects"
        
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    total += self._get_directory_size(project_dir)
                    
        return total
        
    def analyze_project(self, project_id: str) -> Dict[str, Any]:
        """Perform comprehensive project analysis.
        
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
            project = self.get_project(project_id)
            if not project:
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            # Get project resources
            resources = self.list_project_resources(project_id)
            
            # Initialize analysis results
            analysis = {
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_version": "1.0",
                "results": {}
            }
            
            # Perform audio analysis
            audio_tracks = [r for r in resources if r["type"] == "audio"]
            if audio_tracks:
                analysis["results"]["audio_analysis"] = self._analyze_audio_tracks(
                    project_id,
                    audio_tracks
                )
                
            # Perform MIDI analysis
            midi_tracks = [r for r in resources if r["type"] == "midi"]
            if midi_tracks:
                analysis["results"]["midi_analysis"] = self._analyze_midi_tracks(
                    project_id,
                    midi_tracks
                )
                
            # Perform harmonic analysis
            analysis["results"]["harmonic_analysis"] = self._analyze_harmony(
                project_id,
                resources
            )
            
            # Perform genre/style classification
            analysis["results"]["style_analysis"] = self._analyze_style(
                project_id,
                resources
            )
            
            # Perform emotion analysis
            analysis["results"]["emotion_analysis"] = self._analyze_emotion(
                project_id,
                resources
            )
            
            # Save analysis results
            analysis_path = (
                self.base_path / "projects" / project_id / 
                "analysis" / "project_analysis.json"
            )
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
                
            return analysis
            
        except Exception as e:
            raise ProcessingError(f"Error analyzing project: {e}")
            
    def _analyze_audio_tracks(
        self,
        project_id: str,
        audio_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze audio tracks.
        
        Args:
            project_id: Project ID.
            audio_tracks: List of audio track resources.
            
        Returns:
            Dictionary containing audio analysis results.
        """
        results = {
            "tracks": [],
            "overall": {
                "average_loudness": 0.0,
                "dynamic_range": 0.0,
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "zero_crossing_rate": 0.0
            }
        }
        
        for track in audio_tracks:
            track_analysis = {
                "track_id": track["id"],
                "name": track["name"],
                "duration": track["duration"],
                "sample_rate": track["sample_rate"],
                "channels": track["channels"],
                "loudness": track.get("loudness", 0.0),
                "dynamic_range": track.get("dynamic_range", 0.0),
                "spectral_features": track.get("spectral_features", {}),
                "onset_times": track.get("onset_times", []),
                "tempo": track.get("tempo", 0.0)
            }
            results["tracks"].append(track_analysis)
            
            # Update overall statistics
            results["overall"]["average_loudness"] += track_analysis["loudness"]
            results["overall"]["dynamic_range"] += track_analysis["dynamic_range"]
            
        # Calculate averages
        num_tracks = len(audio_tracks)
        if num_tracks > 0:
            results["overall"]["average_loudness"] /= num_tracks
            results["overall"]["dynamic_range"] /= num_tracks
            
        return results
        
    def _analyze_midi_tracks(
        self,
        project_id: str,
        midi_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze MIDI tracks.
        
        Args:
            project_id: Project ID.
            midi_tracks: List of MIDI track resources.
            
        Returns:
            Dictionary containing MIDI analysis results.
        """
        results = {
            "tracks": [],
            "overall": {
                "note_count": 0,
                "average_velocity": 0.0,
                "pitch_range": [0, 0],
                "polyphony": 0
            }
        }
        
        for track in midi_tracks:
            track_analysis = {
                "track_id": track["id"],
                "name": track["name"],
                "instrument": track.get("instrument", "unknown"),
                "note_count": track.get("note_count", 0),
                "average_velocity": track.get("average_velocity", 0.0),
                "pitch_range": track.get("pitch_range", [0, 0]),
                "polyphony": track.get("polyphony", 0),
                "note_density": track.get("note_density", 0.0)
            }
            results["tracks"].append(track_analysis)
            
            # Update overall statistics
            results["overall"]["note_count"] += track_analysis["note_count"]
            results["overall"]["average_velocity"] += track_analysis["average_velocity"]
            results["overall"]["polyphony"] = max(
                results["overall"]["polyphony"],
                track_analysis["polyphony"]
            )
            
            # Update pitch range
            if track_analysis["pitch_range"][0] < results["overall"]["pitch_range"][0]:
                results["overall"]["pitch_range"][0] = track_analysis["pitch_range"][0]
            if track_analysis["pitch_range"][1] > results["overall"]["pitch_range"][1]:
                results["overall"]["pitch_range"][1] = track_analysis["pitch_range"][1]
                
        # Calculate averages
        num_tracks = len(midi_tracks)
        if num_tracks > 0:
            results["overall"]["average_velocity"] /= num_tracks
            
        return results
        
    def _analyze_harmony(
        self,
        project_id: str,
        resources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze harmonic content.
        
        Args:
            project_id: Project ID.
            resources: List of project resources.
            
        Returns:
            Dictionary containing harmonic analysis results.
        """
        results = {
            "key": None,
            "mode": None,
            "chord_progression": [],
            "harmonic_rhythm": 0.0,
            "modulations": [],
            "cadences": []
        }
        
        # Get MIDI tracks for harmonic analysis
        midi_tracks = [r for r in resources if r["type"] == "midi"]
        
        if midi_tracks:
            # Analyze key and mode
            key_analysis = self._analyze_key_and_mode(midi_tracks)
            results["key"] = key_analysis["key"]
            results["mode"] = key_analysis["mode"]
            
            # Analyze chord progression
            chord_analysis = self._analyze_chord_progression(midi_tracks)
            results["chord_progression"] = chord_analysis["progression"]
            results["harmonic_rhythm"] = chord_analysis["rhythm"]
            
            # Analyze modulations
            results["modulations"] = self._analyze_modulations(midi_tracks)
            
            # Analyze cadences
            results["cadences"] = self._analyze_cadences(midi_tracks)
            
        return results
        
    def _analyze_style(
        self,
        project_id: str,
        resources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze musical style.
        
        Args:
            project_id: Project ID.
            resources: List of project resources.
            
        Returns:
            Dictionary containing style analysis results.
        """
        results = {
            "genre": [],
            "style_tags": [],
            "production_techniques": [],
            "instrumentation": [],
            "arrangement_complexity": 0.0
        }
        
        # Analyze audio tracks for genre and style
        audio_tracks = [r for r in resources if r["type"] == "audio"]
        if audio_tracks:
            # Genre classification
            results["genre"] = self._classify_genre(audio_tracks)
            
            # Style analysis
            style_analysis = self._analyze_style_characteristics(audio_tracks)
            results["style_tags"] = style_analysis["tags"]
            results["production_techniques"] = style_analysis["techniques"]
            
            # Instrumentation analysis
            results["instrumentation"] = self._analyze_instrumentation(audio_tracks)
            
            # Arrangement complexity
            results["arrangement_complexity"] = self._calculate_arrangement_complexity(
                resources
            )
            
        return results
        
    def _analyze_emotion(
        self,
        project_id: str,
        resources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze emotional content.
        
        Args:
            project_id: Project ID.
            resources: List of project resources.
            
        Returns:
            Dictionary containing emotion analysis results.
        """
        results = {
            "primary_emotion": None,
            "emotion_tags": [],
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "emotion_timeline": []
        }
        
        # Analyze audio tracks for emotional content
        audio_tracks = [r for r in resources if r["type"] == "audio"]
        if audio_tracks:
            # Emotion classification
            emotion_analysis = self._classify_emotion(audio_tracks)
            results["primary_emotion"] = emotion_analysis["primary"]
            results["emotion_tags"] = emotion_analysis["tags"]
            
            # Valence-Arousal-Dominance analysis
            vad_analysis = self._analyze_vad(audio_tracks)
            results["valence"] = vad_analysis["valence"]
            results["arousal"] = vad_analysis["arousal"]
            results["dominance"] = vad_analysis["dominance"]
            
            # Emotion timeline
            results["emotion_timeline"] = self._generate_emotion_timeline(audio_tracks)
            
        return results
        
    def _analyze_key_and_mode(
        self,
        midi_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze key and mode of MIDI tracks.
        
        Args:
            midi_tracks: List of MIDI track resources.
            
        Returns:
            Dictionary containing key and mode analysis.
        """
        # Collect all notes from all tracks
        all_notes = []
        for track in midi_tracks:
            if "notes" in track:
                all_notes.extend(track["notes"])
                
        if not all_notes:
            return {"key": None, "mode": None}
            
        # Convert MIDI note numbers to pitch classes
        pitch_classes = [note % 12 for note in all_notes]
        
        # Count pitch class occurrences
        pitch_counts = [0] * 12
        for pc in pitch_classes:
            pitch_counts[pc] += 1
            
        # Find the most common pitch class (key)
        key = pitch_counts.index(max(pitch_counts))
        
        # Determine mode based on pitch distribution
        # Compare major and minor scale patterns
        major_pattern = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        minor_pattern = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
        
        major_score = sum(
            pitch_counts[i] * major_pattern[i]
            for i in range(12)
        )
        minor_score = sum(
            pitch_counts[i] * minor_pattern[i]
            for i in range(12)
        )
        
        mode = "major" if major_score >= minor_score else "minor"
        
        return {
            "key": key,
            "mode": mode,
            "confidence": max(major_score, minor_score) / sum(pitch_counts)
        }
        
    def _analyze_chord_progression(
        self,
        midi_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze chord progression in MIDI tracks.
        
        Args:
            midi_tracks: List of MIDI track resources.
            
        Returns:
            Dictionary containing chord progression analysis.
        """
        # Collect all notes with their timing
        notes_with_time = []
        for track in midi_tracks:
            if "notes" in track and "timing" in track:
                for note, time in zip(track["notes"], track["timing"]):
                    notes_with_time.append((note, time))
                    
        if not notes_with_time:
            return {"progression": [], "rhythm": 0.0}
            
        # Sort notes by time
        notes_with_time.sort(key=lambda x: x[1])
        
        # Group notes into chords based on timing
        chords = []
        current_chord = []
        current_time = notes_with_time[0][1]
        
        for note, time in notes_with_time:
            if abs(time - current_time) < 0.1:  # 100ms tolerance
                current_chord.append(note)
            else:
                if current_chord:
                    chords.append(current_chord)
                current_chord = [note]
                current_time = time
                
        if current_chord:
            chords.append(current_chord)
            
        # Analyze chord progression
        progression = []
        for chord in chords:
            # Convert MIDI notes to pitch classes
            pitch_classes = sorted(set(note % 12 for note in chord))
            
            # Identify chord type
            if len(pitch_classes) >= 3:
                root = pitch_classes[0]
                third = pitch_classes[1]
                fifth = pitch_classes[2]
                
                # Determine chord quality
                if (third - root) % 12 == 4:  # Major third
                    quality = "major"
                elif (third - root) % 12 == 3:  # Minor third
                    quality = "minor"
                else:
                    quality = "unknown"
                    
                progression.append({
                    "root": root,
                    "quality": quality,
                    "pitch_classes": pitch_classes
                })
                
        # Calculate harmonic rhythm (average time between chord changes)
        if len(progression) > 1:
            times = [notes_with_time[i][1] for i in range(len(notes_with_time))]
            rhythm = (times[-1] - times[0]) / (len(progression) - 1)
        else:
            rhythm = 0.0
            
        return {
            "progression": progression,
            "rhythm": rhythm
        }
        
    def _analyze_modulations(
        self,
        midi_tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze key modulations in MIDI tracks.
        
        Args:
            midi_tracks: List of MIDI track resources.
            
        Returns:
            List of modulation events.
        """
        modulations = []
        
        # Analyze key changes over time
        time_windows = []
        window_size = 2.0  # 2 seconds per window
        
        for track in midi_tracks:
            if "notes" in track and "timing" in track:
                notes = track["notes"]
                times = track["timing"]
                
                # Create time windows
                current_window = []
                current_time = times[0]
                
                for note, time in zip(notes, times):
                    if time - current_time <= window_size:
                        current_window.append(note)
                    else:
                        if current_window:
                            time_windows.append({
                                "time": current_time,
                                "notes": current_window
                            })
                        current_window = [note]
                        current_time = time
                        
                if current_window:
                    time_windows.append({
                        "time": current_time,
                        "notes": current_window
                    })
                    
        # Analyze key changes between windows
        for i in range(len(time_windows) - 1):
            window1 = time_windows[i]
            window2 = time_windows[i + 1]
            
            # Analyze key in each window
            key1 = self._analyze_key_and_mode([{"notes": window1["notes"]}])
            key2 = self._analyze_key_and_mode([{"notes": window2["notes"]}])
            
            # Check for modulation
            if key1["key"] != key2["key"]:
                modulations.append({
                    "time": window2["time"],
                    "from_key": key1["key"],
                    "from_mode": key1["mode"],
                    "to_key": key2["key"],
                    "to_mode": key2["mode"],
                    "confidence": min(key1["confidence"], key2["confidence"])
                })
                
        return modulations
        
    def _analyze_cadences(
        self,
        midi_tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze cadences in MIDI tracks.
        
        Args:
            midi_tracks: List of MIDI track resources.
            
        Returns:
            List of cadence events.
        """
        cadences = []
        
        # Get chord progression
        chord_analysis = self._analyze_chord_progression(midi_tracks)
        progression = chord_analysis["progression"]
        
        if not progression:
            return cadences
            
        # Analyze cadences
        for i in range(len(progression) - 1):
            current_chord = progression[i]
            next_chord = progression[i + 1]
            
            # Check for perfect cadence (V-I)
            if (current_chord["root"] == 7 and  # V
                next_chord["root"] == 0 and     # I
                current_chord["quality"] == "major" and
                next_chord["quality"] == "major"):
                cadences.append({
                    "type": "perfect",
                    "time": i,
                    "chords": [current_chord, next_chord]
                })
                
            # Check for plagal cadence (IV-I)
            elif (current_chord["root"] == 5 and  # IV
                  next_chord["root"] == 0 and     # I
                  current_chord["quality"] == "major" and
                  next_chord["quality"] == "major"):
                cadences.append({
                    "type": "plagal",
                    "time": i,
                    "chords": [current_chord, next_chord]
                })
                
            # Check for imperfect cadence (any chord to V)
            elif next_chord["root"] == 7:  # V
                cadences.append({
                    "type": "imperfect",
                    "time": i,
                    "chords": [current_chord, next_chord]
                })
                
            # Check for interrupted cadence (V-VI)
            elif (current_chord["root"] == 7 and  # V
                  next_chord["root"] == 9 and     # VI
                  current_chord["quality"] == "major" and
                  next_chord["quality"] == "minor"):
                cadences.append({
                    "type": "interrupted",
                    "time": i,
                    "chords": [current_chord, next_chord]
                })
                
        return cadences
        
    def _classify_genre(
        self,
        audio_tracks: List[Dict[str, Any]]
    ) -> List[str]:
        """Classify genre of audio tracks.
        
        Args:
            audio_tracks: List of audio track resources.
            
        Returns:
            List of genre labels.
        """
        genres = []
        
        for track in audio_tracks:
            if "features" in track:
                features = track["features"]
                
                # Extract relevant features for genre classification
                tempo = features.get("tempo", 0)
                spectral_centroid = features.get("spectral_centroid", 0)
                spectral_rolloff = features.get("spectral_rolloff", 0)
                zero_crossing_rate = features.get("zero_crossing_rate", 0)
                mfcc = features.get("mfcc", [])
                
                # Simple rule-based genre classification
                if tempo > 120 and spectral_centroid > 3000:
                    genres.append("electronic")
                elif tempo > 100 and spectral_rolloff > 4000:
                    genres.append("rock")
                elif tempo < 100 and spectral_centroid < 2000:
                    genres.append("jazz")
                elif zero_crossing_rate > 0.1:
                    genres.append("classical")
                else:
                    genres.append("pop")
                    
        # Remove duplicates and sort
        return sorted(list(set(genres)))
        
    def _analyze_style_characteristics(
        self,
        audio_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze style characteristics of audio tracks.
        
        Args:
            audio_tracks: List of audio track resources.
            
        Returns:
            Dictionary containing style analysis.
        """
        results = {
            "tags": [],
            "techniques": []
        }
        
        for track in audio_tracks:
            if "features" in track:
                features = track["features"]
                
                # Analyze production techniques
                if features.get("reverb", 0) > 0.5:
                    results["techniques"].append("reverb")
                if features.get("delay", 0) > 0.5:
                    results["techniques"].append("delay")
                if features.get("compression", 0) > 0.5:
                    results["techniques"].append("compression")
                if features.get("distortion", 0) > 0.5:
                    results["techniques"].append("distortion")
                    
                # Analyze style tags
                if features.get("brightness", 0) > 0.7:
                    results["tags"].append("bright")
                if features.get("warmth", 0) > 0.7:
                    results["tags"].append("warm")
                if features.get("clarity", 0) > 0.7:
                    results["tags"].append("clear")
                if features.get("punch", 0) > 0.7:
                    results["tags"].append("punchy")
                    
        # Remove duplicates
        results["tags"] = list(set(results["tags"]))
        results["techniques"] = list(set(results["techniques"]))
        
        return results
        
    def _analyze_instrumentation(
        self,
        audio_tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze instrumentation in audio tracks.
        
        Args:
            audio_tracks: List of audio track resources.
            
        Returns:
            List of instrument analysis results.
        """
        instruments = []
        
        for track in audio_tracks:
            if "features" in track:
                features = track["features"]
                
                # Extract relevant features for instrument classification
                spectral_centroid = features.get("spectral_centroid", 0)
                spectral_rolloff = features.get("spectral_rolloff", 0)
                spectral_flatness = features.get("spectral_flatness", 0)
                mfcc = features.get("mfcc", [])
                
                # Simple rule-based instrument classification
                if spectral_flatness > 0.5:
                    instruments.append({
                        "name": "drums",
                        "confidence": spectral_flatness
                    })
                elif spectral_centroid > 3000:
                    instruments.append({
                        "name": "guitar",
                        "confidence": spectral_centroid / 5000
                    })
                elif spectral_rolloff < 2000:
                    instruments.append({
                        "name": "bass",
                        "confidence": 1 - (spectral_rolloff / 2000)
                    })
                else:
                    instruments.append({
                        "name": "synth",
                        "confidence": 0.5
                    })
                    
        return instruments
        
    def _calculate_arrangement_complexity(
        self,
        resources: List[Dict[str, Any]]
    ) -> float:
        """Calculate arrangement complexity score.
        
        Args:
            resources: List of project resources.
            
        Returns:
            Complexity score between 0 and 1.
        """
        # Count number of tracks
        num_tracks = len(resources)
        
        # Count number of different instrument types
        instrument_types = set()
        for resource in resources:
            if "instrument" in resource:
                instrument_types.add(resource["instrument"])
                
        # Calculate average track duration
        durations = []
        for resource in resources:
            if "duration" in resource:
                durations.append(resource["duration"])
                
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate complexity score
        track_complexity = min(num_tracks / 10, 1.0)  # Normalize to 0-1
        instrument_complexity = min(len(instrument_types) / 5, 1.0)  # Normalize to 0-1
        duration_complexity = min(avg_duration / 300, 1.0)  # Normalize to 0-1
        
        # Weighted average of complexity factors
        complexity = (
            0.4 * track_complexity +
            0.3 * instrument_complexity +
            0.3 * duration_complexity
        )
        
        return complexity
        
    def _classify_emotion(
        self,
        audio_tracks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify emotion in audio tracks.
        
        Args:
            audio_tracks: List of audio track resources.
            
        Returns:
            Dictionary containing emotion classification.
        """
        results = {
            "primary": None,
            "tags": []
        }
        
        for track in audio_tracks:
            if "features" in track:
                features = track["features"]
                
                # Extract relevant features for emotion classification
                energy = features.get("energy", 0)
                valence = features.get("valence", 0)
                arousal = features.get("arousal", 0)
                
                # Simple rule-based emotion classification
                if energy > 0.7 and arousal > 0.7:
                    results["tags"].append("energetic")
                    results["primary"] = "energetic"
                elif energy < 0.3 and arousal < 0.3:
                    results["tags"].append("calm")
                    results["primary"] = "calm"
                elif valence > 0.7:
                    results["tags"].append("happy")
                    results["primary"] = "happy"
                elif valence < 0.3:
                    results["tags"].append("sad")
                    results["primary"] = "sad"
                    
        # Remove duplicates
        results["tags"] = list(set(results["tags"]))
        
        return results
        
    def _analyze_vad(
        self,
        audio_tracks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze Valence-Arousal-Dominance in audio tracks.
        
        Args:
            audio_tracks: List of audio track resources.
            
        Returns:
            Dictionary containing VAD values.
        """
        total_valence = 0
        total_arousal = 0
        total_dominance = 0
        count = 0
        
        for track in audio_tracks:
            if "features" in track:
                features = track["features"]
                
                # Extract VAD features
                valence = features.get("valence", 0)
                arousal = features.get("arousal", 0)
                dominance = features.get("dominance", 0)
                
                total_valence += valence
                total_arousal += arousal
                total_dominance += dominance
                count += 1
                
        if count > 0:
            return {
                "valence": total_valence / count,
                "arousal": total_arousal / count,
                "dominance": total_dominance / count
            }
        else:
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0
            }
            
    def _generate_emotion_timeline(
        self,
        audio_tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate emotion timeline from audio tracks.
        
        Args:
            audio_tracks: List of audio track resources.
            
        Returns:
            List of emotion events over time.
        """
        timeline = []
        
        for track in audio_tracks:
            if "features" in track and "timing" in track:
                features = track["features"]
                times = track["timing"]
                
                # Extract emotion features over time
                for i, time in enumerate(times):
                    if i < len(features):
                        emotion = {
                            "time": time,
                            "valence": features.get("valence", 0),
                            "arousal": features.get("arousal", 0),
                            "dominance": features.get("dominance", 0)
                        }
                        timeline.append(emotion)
                        
        # Sort by time
        timeline.sort(key=lambda x: x["time"])
        
        return timeline
        
    async def update_project_profile(self, project_id: str) -> ProjectResource:
        """Update project profile with latest analysis.
        
        Args:
            project_id: Project ID.
            
        Returns:
            Updated project resource.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If project file not found or processing fails.
        """
        # Get project
        project = await self.project_repository.find_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(f"Project {project_id} not found")
            
        # Get project file path
        project_path = project.basic_info.get("original_file_path")
        if not project_path or not os.path.exists(project_path):
            raise ProcessingError(f"Project file not found: {project_path}")
            
        # Export and organize project files
        exported_files = await self.export_project_files(project_path)
        organized_files = await self.organize_project_files(project_id, exported_files)
        
        # Process tracks
        track_resources = []
        for track_name, file_path in organized_files.items():
            track_data = {
                "name": track_name,
                "type": "midi" if file_path.endswith('.mid') else "audio",
                "file_path": file_path
            }
            track_resource = await self._process_track(track_data, project_id)
            track_resources.append(track_resource.resource_id)
        
        # Update project tracks
        project.tracks = track_resources
        
        # Perform analysis using ProjectAnalysisService
        analysis_service = ProjectAnalysisService(self)
        analysis_results = await analysis_service.analyze_project(project, track_resources)
        
        # Update project with analysis results
        await analysis_service._update_project_with_analysis(project, analysis_results)
        
        return project
        
    async def _extract_project_data(self, project_path: str) -> Dict[str, Any]:
        """Extract data from Logic project file.
        
        Args:
            project_path: Path to Logic project file.
            
        Returns:
            Dictionary with extracted data.
            
        Raises:
            ProcessingError: If extraction fails.
        """
        try:
            # Use Logic Pro scripting bridge to extract data
            script = f"""
                tell application "Logic Pro"
                    set proj to open "{project_path}"
                    set projData to {{}}
                    
                    -- Get basic properties
                    set tempo to tempo of proj
                    set sig to time signature of proj
                    set key to key signature of proj
                    
                    -- Get track data
                    set trackList to {{}}
                    repeat with t in tracks of proj
                        set trackData to {{
                            name:name of t,
                            type:track type of t,
                            file:file path of t,
                            mute:mute of t,
                            solo:solo of t,
                            volume:volume of t
                        }}
                        copy trackData to end of trackList
                    end repeat
                    
                    return {{tempo:tempo, signature:sig, key:key, tracks:trackList}}
                end tell
            """
            
            # Execute AppleScript
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode != 0:
                raise ProcessingError(f"Logic Pro script failed: {result.stderr}")
                
            # Parse result
            data = json.loads(result.stdout)
            
            return {
                "musical_properties": {
                    "bpm": float(data["tempo"]),
                    "key": data["key"],
                    "time_signature": data["signature"]
                },
                "tracks": [
                    {
                        "name": track["name"],
                        "type": track["type"],
                        "file_path": track["file"],
                        "properties": {
                            "mute": track["mute"],
                            "solo": track["solo"],
                            "volume": float(track["volume"])
                        }
                    }
                    for track in data["tracks"]
                ]
            }
            
        except Exception as e:
            raise ProcessingError(f"Failed to extract project data: {str(e)}")
            
    async def _process_track(
        self,
        track_data: Dict[str, Any],
        project_id: str
    ) -> TrackResource:
        """Process a track and create/update resource.
        
        Args:
            track_data: Track data.
            project_id: Parent project ID.
            
        Returns:
            Track resource.
            
        Raises:
            ProcessingError: If processing fails.
        """
        # Generate track ID based on file path
        file_path = track_data.get("file_path")
        track_id = f"track_{hashlib.md5(file_path.encode()).hexdigest()}"
        
        # Check if track already exists
        existing_track = await self.track_repository.find_by_id(track_id)
        if existing_track:
            track = existing_track
        else:
            # Create new track resource
            track = TrackResource(
                resource_id=track_id,
                resource_type="track",
                parent_resources=[project_id],
                basic_info={
                    "name": track_data.get("name"),
                    "type": track_data.get("type"),
                    "original_file_path": file_path
                },
                content_reference={
                    "storage_type": "file_system",
                    "location": file_path,
                    "format": os.path.splitext(file_path)[1][1:]
                }
            )
            
        # Process based on track type
        if track_data.get("type") == "audio":
            await self._process_audio_track(track, file_path)
        elif track_data.get("type") == "midi":
            await self._process_midi_track(track, file_path)
            
        # Save track
        await self.track_repository.save(track)
        
        return track
        
    async def _process_audio_track(self, track: TrackResource, file_path: str) -> None:
        """Process audio track.
        
        Args:
            track: Track resource.
            file_path: Path to audio file.
            
        Raises:
            ProcessingError: If processing fails.
        """
        try:
            # Load audio
            audio, sr = await self.audio_processor.load_audio(file_path)
            
            # Extract features
            features = await self.audio_processor.extract_features(audio, sr)
            
            # Update track with audio properties
            track.audio_properties = {
                "sample_rate": sr,
                "channels": 2 if len(audio.shape) > 1 else 1,
                "duration_seconds": len(audio) / sr,
                "peak_amplitude": float(abs(audio).max())
            }
            
            # Store feature references
            track.derived_features = {
                "extracted_audio_features": {
                    feature_name: f"feature_{track.resource_id}_{feature_name}"
                    for feature_name in features.keys()
                }
            }
            
            # Save features as separate resources
            for name, feature in features.items():
                feature_id = f"feature_{track.resource_id}_{name}"
                await self._save_feature(feature_id, feature)
                
        except Exception as e:
            raise ProcessingError(f"Failed to process audio track: {str(e)}")
            
    async def _process_midi_track(self, track: TrackResource, file_path: str) -> None:
        """Process MIDI track.
        
        Args:
            track: Track resource.
            file_path: Path to MIDI file.
            
        Raises:
            ProcessingError: If processing fails.
        """
        try:
            # Parse MIDI
            midi_data = await self.midi_processor.parse_midi(file_path)
            
            # Extract properties
            notes = await self.midi_processor.extract_notes(midi_data)
            
            # Update track with MIDI properties
            track.midi_properties = {
                "note_count": len(notes),
                "velocity_range": [
                    min(note.velocity for note in notes),
                    max(note.velocity for note in notes)
                ],
                "pitch_range": [
                    min(note.pitch for note in notes),
                    max(note.pitch for note in notes)
                ],
                "polyphony": await self.midi_processor.is_polyphonic(notes)
            }
            
            # Store derived features
            track.derived_features = {
                "extracted_midi_features": {
                    "notes": f"feature_{track.resource_id}_notes",
                    "piano_roll": f"feature_{track.resource_id}_piano_roll"
                }
            }
            
            # Save features as separate resources
            await self._save_feature(
                f"feature_{track.resource_id}_notes",
                notes
            )
            
            piano_roll = await self.midi_processor.create_piano_roll(notes)
            await self._save_feature(
                f"feature_{track.resource_id}_piano_roll",
                piano_roll
            )
            
        except Exception as e:
            raise ProcessingError(f"Failed to process MIDI track: {str(e)}")
            
    async def _analyze_project(
        self,
        project: ProjectResource,
        track_ids: List[str]
    ) -> List[str]:
        """Perform musical analysis on project.
        
        Args:
            project: Project resource.
            track_ids: List of track IDs.
            
        Returns:
            List of analysis result IDs.
            
        Raises:
            ProcessingError: If analysis fails.
        """
        try:
            analysis_ids = []
            
            # Chord/harmonic analysis
            chord_analysis = await self._analyze_harmony(project, track_ids)
            analysis_ids.append(chord_analysis)
            
            # Genre classification
            genre_analysis = await self._analyze_genre(project, track_ids)
            analysis_ids.append(genre_analysis)
            
            # Emotional content analysis
            emotion_analysis = await self._analyze_emotion(project, track_ids)
            analysis_ids.append(emotion_analysis)
            
            return analysis_ids
            
        except Exception as e:
            raise ProcessingError(f"Failed to analyze project: {str(e)}")
            
    async def _save_feature(self, feature_id: str, feature_data: Any) -> None:
        """Save feature data as resource.
        
        Args:
            feature_id: Feature resource ID.
            feature_data: Feature data to save.
            
        Raises:
            ProcessingError: If saving fails.
        """
        try:
            feature_resource = FeatureResource(
                resource_id=feature_id,
                resource_type="feature",
                data=feature_data
            )
            await self.feature_repository.save(feature_resource)
            
        except Exception as e:
            raise ProcessingError(f"Failed to save feature: {str(e)}")
            
    async def import_project(self, project_path: str) -> ProjectResource:
        """Import a new project and create structured environment.
        
        Args:
            project_path: Path to Logic project file.
            
        Returns:
            Created project resource.
            
        Raises:
            ProcessingError: If import fails.
            ValidationError: If project path is invalid.
        """
        try:
            # Validate project path
            if not os.path.exists(project_path):
                raise ValidationError(f"Project file not found: {project_path}")
                
            if not project_path.endswith('.logicx'):
                raise ValidationError("Only Logic Pro projects (.logicx) are supported")
                
            # Generate project ID
            project_id = f"project_{uuid.uuid4().hex}"
            
            # Export project files
            exported_files = await self.export_project_files(project_path)
            
            # Create project directory structure
            project_dir = os.path.join(self.base_path, project_id)
            os.makedirs(project_dir, exist_ok=True)
            
            # Organize exported files
            organized_files = await self.organize_project_files(project_id, exported_files)
            
            # Extract project data
            project_data = await self._extract_project_data(project_path)
            
            # Create project resource
            project = ProjectResource(
                resource_id=project_id,
                resource_type="project",
                basic_info={
                    "name": os.path.basename(project_path),
                    "original_file_path": project_path,
                    "project_directory": project_dir
                },
                musical_properties=project_data.get("musical_properties", {}),
                tracks=[],  # Will be populated during processing
                analysis_results=[],  # Will be populated during analysis
                creation_timestamp=datetime.utcnow(),
                modification_timestamp=datetime.utcnow()
            )
            
            # Save initial project state
            await self.project_repository.save(project)
            
            # Process tracks
            track_resources = []
            for track_name, file_path in organized_files.items():
                track_data = {
                    "name": track_name,
                    "type": "midi" if file_path.endswith('.mid') else "audio",
                    "file_path": file_path
                }
                
                # Process track
                track_resource = await self._process_track(track_data, project_id)
                track_resources.append(track_resource.resource_id)
                
            # Update project with track resources
            project.tracks = track_resources
            
            # Perform initial analysis
            analysis_results = await self._analyze_project(project, track_resources)
            project.analysis_results = analysis_results
            
            # Analyze musical style
            style_results = await self.analyze_musical_style(project, track_resources)
            project.musical_properties.update(style_results)
            
            # Save final project state
            await self.project_repository.save(project)
            
            return project
            
        except Exception as e:
            # Clean up on failure
            if 'project_dir' in locals():
                shutil.rmtree(project_dir, ignore_errors=True)
            raise ProcessingError(f"Failed to import project: {str(e)}")
            
    async def _copy_track_file(
        self,
        source_path: str,
        project_dir: str,
        track_type: str
    ) -> str:
        """Copy track file to project directory.
        
        Args:
            source_path: Original file path.
            project_dir: Project directory path.
            track_type: Type of track ('audio' or 'midi').
            
        Returns:
            New file path in project directory.
            
        Raises:
            ProcessingError: If copy fails.
        """
        try:
            # Determine target subdirectory
            subdir = 'audio' if track_type == 'audio' else 'midi'
            target_dir = os.path.join(project_dir, subdir)
            
            # Generate unique filename
            filename = f"{uuid.uuid4().hex}{os.path.splitext(source_path)[1]}"
            target_path = os.path.join(target_dir, filename)
            
            # Copy file
            shutil.copy2(source_path, target_path)
            
            return target_path
            
        except Exception as e:
            raise ProcessingError(f"Failed to copy track file: {str(e)}")
            
    async def quantize_project(
        self,
        project_id: str,
        quantize_settings: Dict[str, Any] = None
    ) -> ProjectResource:
        """Quantize project audio and MIDI to consistent grid.
        
        Args:
            project_id: Project ID.
            quantize_settings: Optional quantization settings:
                - grid_value: Grid value for quantization (e.g., "1/16")
                - swing: Swing amount (0-1)
                - strength: Quantization strength (0-1)
                - smart_quantize: Whether to use intelligent quantization
                
        Returns:
            Updated project resource.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If quantization fails.
        """
        try:
            # Get project
            project = await self.project_repository.find_by_id(project_id)
            if not project:
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            # Use default settings if none provided
            if quantize_settings is None:
                quantize_settings = {
                    "grid_value": "1/16",
                    "swing": 0.0,
                    "strength": 1.0,
                    "smart_quantize": True
                }
                
            # Standardize BPM
            await self._standardize_bpm(project)
            
            # Process each track
            for track_id in project.tracks:
                track = await self.track_repository.find_by_id(track_id)
                if not track:
                    continue
                    
                if track.basic_info.get("type") == "midi":
                    await self._quantize_midi_track(track, quantize_settings)
                elif track.basic_info.get("type") == "audio":
                    await self._quantize_audio_track(track, quantize_settings)
                    
            # Adjust project timeline
            await self._adjust_project_timeline(project)
            
            # Update project modification time
            project.modification_timestamp = datetime.utcnow()
            await self.project_repository.save(project)
            
            return project
            
        except Exception as e:
            raise ProcessingError(f"Failed to quantize project: {str(e)}")
            
    async def _standardize_bpm(self, project: ProjectResource) -> None:
        """Standardize project BPM.
        
        Args:
            project: Project resource.
            
        Raises:
            ProcessingError: If BPM standardization fails.
        """
        try:
            # Get current BPM from project properties
            current_bpm = project.musical_properties.get("bpm")
            if not current_bpm:
                raise ProcessingError("Project BPM not found")
                
            # Use Logic Pro scripting to update BPM
            script = f"""
                tell application "Logic Pro"
                    set proj to open "{project.basic_info.get('original_file_path')}"
                    set tempo of proj to {current_bpm}
                end tell
            """
            
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode != 0:
                raise ProcessingError(f"Failed to update BPM: {result.stderr}")
                
        except Exception as e:
            raise ProcessingError(f"Failed to standardize BPM: {str(e)}")
            
    async def _quantize_midi_track(
        self,
        track: TrackResource,
        settings: Dict[str, Any]
    ) -> None:
        """Quantize MIDI track.
        
        Args:
            track: Track resource.
            settings: Quantization settings.
            
        Raises:
            ProcessingError: If quantization fails.
        """
        try:
            # Load MIDI data
            midi_data = await self.midi_processor.parse_midi(
                track.basic_info.get("original_file_path")
            )
            
            # Extract notes
            notes = await self.midi_processor.extract_notes(midi_data)
            
            # Apply quantization
            quantized_notes = await self.midi_processor.quantize_notes(
                notes,
                grid_value=settings["grid_value"],
                swing=settings["swing"],
                strength=settings["strength"],
                smart=settings["smart_quantize"]
            )
            
            # Create new MIDI file
            output_path = os.path.join(
                os.path.dirname(track.basic_info.get("original_file_path")),
                f"quantized_{os.path.basename(track.basic_info.get('original_file_path'))}"
            )
            
            await self.midi_processor.save_midi(quantized_notes, output_path)
            
            # Update track reference
            track.basic_info["original_file_path"] = output_path
            await self.track_repository.save(track)
            
        except Exception as e:
            raise ProcessingError(f"Failed to quantize MIDI track: {str(e)}")
            
    async def _quantize_audio_track(
        self,
        track: TrackResource,
        settings: Dict[str, Any]
    ) -> None:
        """Quantize audio track using Flex Time.
        
        Args:
            track: Track resource.
            settings: Quantization settings.
            
        Raises:
            ProcessingError: If quantization fails.
        """
        try:
            # Load audio
            audio, sr = await self.audio_processor.load_audio(
                track.basic_info.get("original_file_path")
            )
            
            # Detect transients
            transients = await self.audio_processor.detect_transients(audio, sr)
            
            # Calculate target positions based on grid
            grid_positions = await self.audio_processor.calculate_grid_positions(
                len(audio),
                sr,
                settings["grid_value"],
                settings["swing"]
            )
            
            # Apply flex time quantization
            quantized_audio = await self.audio_processor.flex_quantize(
                audio,
                transients,
                grid_positions,
                strength=settings["strength"],
                preserve_formants=True
            )
            
            # Save quantized audio
            output_path = os.path.join(
                os.path.dirname(track.basic_info.get("original_file_path")),
                f"quantized_{os.path.basename(track.basic_info.get('original_file_path'))}"
            )
            
            await self.audio_processor.save_audio(quantized_audio, sr, output_path)
            
            # Update track reference
            track.basic_info["original_file_path"] = output_path
            await self.track_repository.save(track)
            
        except Exception as e:
            raise ProcessingError(f"Failed to quantize audio track: {str(e)}")
            
    async def _adjust_project_timeline(self, project: ProjectResource) -> None:
        """Adjust project timeline to start at second marker.
        
        Args:
            project: Project resource.
            
        Raises:
            ProcessingError: If timeline adjustment fails.
        """
        try:
            # Use Logic Pro scripting to adjust timeline
            script = f"""
                tell application "Logic Pro"
                    set proj to open "{project.basic_info.get('original_file_path')}"
                    
                    -- Move all regions to start at 1.1.1
                    repeat with t in tracks of proj
                        repeat with r in regions of t
                            set start of r to 1
                        end repeat
                    end repeat
                    
                    -- Add pre-roll if needed
                    set pre roll of proj to 1
                end tell
            """
            
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode != 0:
                raise ProcessingError(f"Failed to adjust timeline: {result.stderr}")
                
        except Exception as e:
            raise ProcessingError(f"Failed to adjust project timeline: {str(e)}")
            
    async def generate_tracks(
        self,
        project_id: str,
        generation_settings: Dict[str, Any] = None
    ) -> ProjectResource:
        """Generate new tracks following project harmony.
        
        Args:
            project_id: Project ID.
            generation_settings: Optional generation settings:
                - num_variations: Number of variations to generate (default: 5)
                - track_types: List of track types to generate
                - style_reference: Optional reference track for style
                - temperature: Generation temperature (0-1)
                
        Returns:
            Updated project resource.
            
        Raises:
            ResourceNotFoundError: If project not found.
            ProcessingError: If generation fails.
        """
        try:
            # Get project
            project = await self.project_repository.find_by_id(project_id)
            if not project:
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            # Use default settings if none provided
            if generation_settings is None:
                generation_settings = {
                    "num_variations": 5,
                    "track_types": ["vocal", "lead", "harmony", "drums"],
                    "temperature": 0.8
                }
                
            # Analyze harmony
            harmony_data = await self._analyze_harmony(project, project.tracks)
            
            # Prepare generation context
            context = {
                "harmony": harmony_data,
                "bpm": project.musical_properties.get("bpm"),
                "key": project.musical_properties.get("key"),
                "time_signature": project.musical_properties.get("time_signature"),
                "style_reference": generation_settings.get("style_reference")
            }
            
            # Generate tracks
            new_track_ids = []
            for track_type in generation_settings["track_types"]:
                track_ids = await self._generate_track_variations(
                    project,
                    track_type,
                    context,
                    generation_settings["num_variations"],
                    generation_settings["temperature"]
                )
                new_track_ids.extend(track_ids)
                
            # Add new tracks to project
            project.tracks.extend(new_track_ids)
            
            # Update project modification time
            project.modification_timestamp = datetime.utcnow()
            await self.project_repository.save(project)
            
            return project
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate tracks: {str(e)}")
            
    async def _generate_track_variations(
        self,
        project: ProjectResource,
        track_type: str,
        context: Dict[str, Any],
        num_variations: int,
        temperature: float
    ) -> List[str]:
        """Generate variations of a specific track type.
        
        Args:
            project: Project resource.
            track_type: Type of track to generate.
            context: Generation context.
            num_variations: Number of variations to generate.
            temperature: Generation temperature.
            
        Returns:
            List of generated track resource IDs.
            
        Raises:
            ProcessingError: If generation fails.
        """
        try:
            track_ids = []
            
            for i in range(num_variations):
                # Generate MIDI data
                midi_data = await self.generation_service.generate_midi(
                    track_type=track_type,
                    context=context,
                    temperature=temperature
                )
                
                # Create track file
                track_dir = os.path.join(
                    project.basic_info.get("project_directory"),
                    "generated",
                    track_type
                )
                os.makedirs(track_dir, exist_ok=True)
                
                midi_path = os.path.join(
                    track_dir,
                    f"variation_{i+1}.mid"
                )
                
                await self.midi_processor.save_midi(midi_data, midi_path)
                
                # Create track resource
                track = TrackResource(
                    resource_id=f"track_{uuid.uuid4().hex}",
                    resource_type="track",
                    parent_resources=[project.resource_id],
                    basic_info={
                        "name": f"{track_type.title()} Variation {i+1}",
                        "type": "midi",
                        "track_type": track_type,
                        "original_file_path": midi_path
                    }
                )
                
                # Process track
                await self._process_midi_track(track, midi_path)
                
                # Save track
                await self.track_repository.save(track)
                track_ids.append(track.resource_id)
                
            return track_ids
            
        except Exception as e:
            raise ProcessingError(f"Failed to generate {track_type} variations: {str(e)}")
            
    async def _analyze_harmony(
        self,
        project: ProjectResource,
        track_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze project harmony.
        
        Args:
            project: Project resource.
            track_ids: List of track IDs.
            
        Returns:
            Dictionary containing harmony analysis.
            
        Raises:
            ProcessingError: If analysis fails.
        """
        try:
            # Get all MIDI tracks
            midi_tracks = []
            for track_id in track_ids:
                track = await self.track_repository.find_by_id(track_id)
                if track and track.basic_info.get("type") == "midi":
                    midi_tracks.append(track)
                    
            # Extract notes from all tracks
            all_notes = []
            for track in midi_tracks:
                midi_data = await self.midi_processor.parse_midi(
                    track.basic_info.get("original_file_path")
                )
                notes = await self.midi_processor.extract_notes(midi_data)
                all_notes.extend(notes)
                
            # Sort notes by time
            all_notes.sort(key=lambda x: x.start_time)
            
            # Analyze harmony
            harmony_data = await self.midi_processor.analyze_harmony(
                all_notes,
                project.musical_properties.get("key"),
                project.musical_properties.get("time_signature")
            )
            
            return harmony_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to analyze harmony: {str(e)}")
            
    async def tune_vocals(
        self,
        project_id: str,
        track_id: str,
        tuning_settings: Dict[str, Any] = None
    ) -> ProjectResource:
        """Apply intelligent pitch correction to vocal track.
        
        Args:
            project_id: Project ID.
            track_id: Vocal track ID.
            tuning_settings: Optional tuning settings:
                - correction_strength: Pitch correction strength (0-1)
                - correction_speed: Speed of correction (ms)
                - preserve_formants: Whether to preserve formants
                - vibrato_amount: Amount of vibrato to preserve (0-1)
                - num_variations: Number of variations to create
                
        Returns:
            Updated project resource.
            
        Raises:
            ResourceNotFoundError: If project or track not found.
            ProcessingError: If tuning fails.
        """
        try:
            # Get project and track
            project = await self.project_repository.find_by_id(project_id)
            if not project:
                raise ResourceNotFoundError(f"Project {project_id} not found")
                
            track = await self.track_repository.find_by_id(track_id)
            if not track:
                raise ResourceNotFoundError(f"Track {track_id} not found")
                
            if track.basic_info.get("type") != "audio":
                raise ProcessingError("Track must be an audio track")
                
            # Use default settings if none provided
            if tuning_settings is None:
                tuning_settings = {
                    "correction_strength": 0.8,
                    "correction_speed": 20,
                    "preserve_formants": True,
                    "vibrato_amount": 0.5,
                    "num_variations": 3
                }
                
            # Get project context
            context = {
                "key": project.musical_properties.get("key"),
                "harmony": await self._analyze_harmony(project, project.tracks)
            }
            
            # Create variations directory
            variations_dir = os.path.join(
                project.basic_info.get("project_directory"),
                "generated",
                "tuned_vocals"
            )
            os.makedirs(variations_dir, exist_ok=True)
            
            # Process vocal track
            new_track_ids = await self._process_vocal_track(
                track,
                variations_dir,
                context,
                tuning_settings
            )
            
            # Add new tracks to project
            project.tracks.extend(new_track_ids)
            
            # Update project modification time
            project.modification_timestamp = datetime.utcnow()
            await self.project_repository.save(project)
            
            return project
            
        except Exception as e:
            raise ProcessingError(f"Failed to tune vocals: {str(e)}")
            
    async def _process_vocal_track(
        self,
        track: TrackResource,
        output_dir: str,
        context: Dict[str, Any],
        settings: Dict[str, Any]
    ) -> List[str]:
        """Process vocal track with pitch correction.
        
        Args:
            track: Track resource.
            output_dir: Output directory for variations.
            context: Musical context.
            settings: Tuning settings.
            
        Returns:
            List of new track resource IDs.
            
        Raises:
            ProcessingError: If processing fails.
        """
        try:
            # Load audio
            audio, sr = await self.audio_processor.load_audio(
                track.basic_info.get("original_file_path")
            )
            
            # Detect pitch
            pitch_data = await self.audio_processor.detect_pitch(audio, sr)
            
            # Create pitch correction map
            correction_map = await self._create_pitch_correction_map(
                pitch_data,
                context["key"],
                context["harmony"]
            )
            
            track_ids = []
            
            # Create variations with different settings
            for i in range(settings["num_variations"]):
                # Adjust settings for variation
                variation_settings = settings.copy()
                if i == 0:
                    # First variation: subtle correction
                    variation_settings["correction_strength"] *= 0.7
                    variation_settings["correction_speed"] *= 1.5
                elif i == settings["num_variations"] - 1:
                    # Last variation: stronger correction
                    variation_settings["correction_strength"] *= 1.2
                    variation_settings["correction_speed"] *= 0.7
                    
                # Apply pitch correction
                tuned_audio = await self.audio_processor.apply_pitch_correction(
                    audio,
                    sr,
                    pitch_data,
                    correction_map,
                    strength=variation_settings["correction_strength"],
                    speed=variation_settings["correction_speed"],
                    preserve_formants=variation_settings["preserve_formants"],
                    vibrato_amount=variation_settings["vibrato_amount"]
                )
                
                # Save tuned audio
                output_path = os.path.join(
                    output_dir,
                    f"tuned_variation_{i+1}{os.path.splitext(track.basic_info.get('original_file_path'))[1]}"
                )
                
                await self.audio_processor.save_audio(tuned_audio, sr, output_path)
                
                # Create track resource
                tuned_track = TrackResource(
                    resource_id=f"track_{uuid.uuid4().hex}",
                    resource_type="track",
                    parent_resources=[track.resource_id],
                    basic_info={
                        "name": f"Tuned Vocal Variation {i+1}",
                        "type": "audio",
                        "track_type": "vocal",
                        "original_file_path": output_path,
                        "tuning_settings": variation_settings
                    }
                )
                
                # Process track
                await self._process_audio_track(tuned_track, output_path)
                
                # Save track
                await self.track_repository.save(tuned_track)
                track_ids.append(tuned_track.resource_id)
                
            return track_ids
            
        except Exception as e:
            raise ProcessingError(f"Failed to process vocal track: {str(e)}")
            
    async def _create_pitch_correction_map(
        self,
        pitch_data: Dict[str, Any],
        key: str,
        harmony: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create pitch correction map based on musical context.
        
        Args:
            pitch_data: Detected pitch data.
            key: Project key.
            harmony: Harmony analysis data.
            
        Returns:
            Dictionary containing pitch correction mapping.
            
        Raises:
            ProcessingError: If mapping creation fails.
        """
        try:
            # Extract scale notes from key
            scale_notes = await self.midi_processor.get_scale_notes(key)
            
            # Create time-based pitch mapping
            pitch_map = {}
            
            for time, pitch in pitch_data.items():
                # Find current chord from harmony data
                current_chord = None
                for chord in harmony["chord_progression"]:
                    if chord["start_time"] <= time <= chord["end_time"]:
                        current_chord = chord
                        break
                        
                if current_chord:
                    # Get valid notes for current chord
                    valid_notes = await self.midi_processor.get_chord_notes(
                        current_chord["root"],
                        current_chord["quality"]
                    )
                else:
                    # Default to scale notes if no chord found
                    valid_notes = scale_notes
                    
                # Find closest valid note
                target_pitch = await self.midi_processor.find_closest_note(
                    pitch,
                    valid_notes
                )
                
                pitch_map[time] = target_pitch
                
            return pitch_map
            
        except Exception as e:
            raise ProcessingError(f"Failed to create pitch correction map: {str(e)}") 

    async def export_project_files(self, project_path: str) -> Dict[str, str]:
        """Export WAV and MIDI files from Logic Pro project.
        
        Args:
            project_path: Path to Logic Pro project file.
            
        Returns:
            Dictionary mapping track names to exported file paths.
            
        Raises:
            ProcessingError: If export fails.
        """
        try:
            # Use Logic Pro scripting to export files
            script = f"""
                tell application "Logic Pro"
                    set proj to open "{project_path}"
                    
                    -- Create export directory
                    set export_dir to (path to desktop as text) & "disco_musica_export_" & (random number from 1000 to 9999)
                    do shell script "mkdir " & quoted form of POSIX path of export_dir
                    
                    -- Export each track
                    repeat with t in tracks of proj
                        set track_name to name of t
                        set wav_path to export_dir & "/" & track_name & ".wav"
                        set midi_path to export_dir & "/" & track_name & ".mid"
                        
                        -- Export WAV
                        export t as audio file wav_path
                        
                        -- Export MIDI if track has MIDI data
                        if class of t is MIDI track then
                            export t as MIDI file midi_path
                        end if
                    end repeat
                    
                    return export_dir
                end tell
            """
            
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode != 0:
                raise ProcessingError(f"Failed to export project files: {result.stderr}")
                
            export_dir = result.stdout.strip()
            
            # Get list of exported files
            files = {}
            for file_path in Path(export_dir).glob("*"):
                if file_path.suffix in ['.wav', '.mid']:
                    files[file_path.stem] = str(file_path)
                    
            return files
            
        except Exception as e:
            raise ProcessingError(f"Failed to export project files: {str(e)}")

    async def organize_project_files(
        self,
        project_id: str,
        exported_files: Dict[str, str]
    ) -> Dict[str, str]:
        """Organize exported files into project directory structure.
        
        Args:
            project_id: Project ID.
            exported_files: Dictionary mapping track names to file paths.
            
        Returns:
            Dictionary mapping track names to organized file paths.
            
        Raises:
            ProcessingError: If organization fails.
        """
        try:
            project_dir = self.base_path / "projects" / project_id
            
            # Create subdirectories
            audio_dir = project_dir / "audio"
            midi_dir = project_dir / "midi"
            audio_dir.mkdir(parents=True, exist_ok=True)
            midi_dir.mkdir(parents=True, exist_ok=True)
            
            # Organize files
            organized_files = {}
            for track_name, file_path in exported_files.items():
                file_path = Path(file_path)
                if file_path.suffix == '.wav':
                    target_dir = audio_dir
                else:
                    target_dir = midi_dir
                    
                # Generate unique filename
                new_filename = f"{track_name}_{uuid.uuid4().hex}{file_path.suffix}"
                target_path = target_dir / new_filename
                
                # Copy file
                shutil.copy2(file_path, target_path)
                organized_files[track_name] = str(target_path)
                
            return organized_files
            
        except Exception as e:
            raise ProcessingError(f"Failed to organize project files: {str(e)}")

    async def analyze_musical_style(
        self,
        project: ProjectResource,
        track_resources: List[str]
    ) -> Dict[str, Any]:
        """Analyze musical style, genre, and emotion.
        
        Args:
            project: Project resource.
            track_resources: List of track resource IDs.
            
        Returns:
            Dictionary containing analysis results.
            
        Raises:
            ProcessingError: If analysis fails.
        """
        try:
            # Load pre-trained models
            genre_model = await self.model_service.load_model("genre_classifier")
            emotion_model = await self.model_service.load_model("emotion_classifier")
            
            # Analyze each track
            style_results = {
                "genres": [],
                "emotions": [],
                "style_tags": []
            }
            
            for track_id in track_resources:
                track = await self.track_repository.find_by_id(track_id)
                if not track:
                    continue
                    
                # Extract features
                features = await self._extract_track_features(track)
                
                # Get genre prediction
                genre_pred = await genre_model.predict(features)
                style_results["genres"].extend(genre_pred)
                
                # Get emotion prediction
                emotion_pred = await emotion_model.predict(features)
                style_results["emotions"].extend(emotion_pred)
                
            # Deduplicate and sort results
            style_results["genres"] = sorted(set(style_results["genres"]))
            style_results["emotions"] = sorted(set(style_results["emotions"]))
            
            # Generate style tags based on analysis
            style_results["style_tags"] = self._generate_style_tags(style_results)
            
            return style_results
            
        except Exception as e:
            raise ProcessingError(f"Failed to analyze musical style: {str(e)}")

    def _generate_style_tags(self, style_results: Dict[str, Any]) -> List[str]:
        """Generate style tags based on analysis results.
        
        Args:
            style_results: Dictionary containing analysis results.
            
        Returns:
            List of style tags.
        """
        tags = []
        
        # Add genre-based tags
        for genre in style_results["genres"]:
            tags.append(genre)
            
        # Add emotion-based tags
        for emotion in style_results["emotions"]:
            tags.append(emotion)
            
        # Add production style tags based on analysis
        if "electronic" in style_results["genres"]:
            tags.extend(["synthesizer-heavy", "modern"])
        if "rock" in style_results["genres"]:
            tags.extend(["guitar-driven", "energetic"])
        if "jazz" in style_results["genres"]:
            tags.extend(["improvisational", "complex"])
            
        return sorted(set(tags))