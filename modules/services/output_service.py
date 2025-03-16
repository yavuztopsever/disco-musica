"""
Output Management Service for Disco Musica.

This module provides services for managing, organizing, and exporting
generated music outputs.
"""

import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging

from modules.core.config import config
from modules.core.audio_processor import AudioProcessor
from modules.core.resources.resource_manager import ResourceManager
from modules.core.resources.generation_resource import GenerationResource
from modules.core.processors.processor_manager import ProcessorManager
from modules.exceptions.base_exceptions import (
    ProcessingError,
    ValidationError,
    ResourceNotFoundError
)


class OutputService:
    """
    Service for managing, organizing, and exporting generated music outputs.
    
    This class provides functionalities for saving, loading, organizing,
    and exporting generated music outputs.
    """
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        processor_manager: ProcessorManager,
        output_dir: Union[str, Path]
    ):
        """
        Initialize the OutputService.
        
        Args:
            resource_manager: Resource manager instance.
            processor_manager: Processor manager instance.
            output_dir: Directory for storing outputs.
        """
        self.resource_manager = resource_manager
        self.processor_manager = processor_manager
        self.output_dir = Path(output_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self._ensure_directories()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Output registry
        self.registry = self._load_registry()
    
    def _ensure_directories(self) -> None:
        """
        Ensure required directories exist.
        """
        # Main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Type-specific directories
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "midi").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        
        # Archive directory
        (self.output_dir / "archive").mkdir(exist_ok=True)
        
    def _load_registry(self) -> Dict:
        """
        Load the output registry from file.
        
        Returns:
            Registry dictionary.
        """
        registry_path = self.output_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {"outputs": {}, "collections": {}}
        else:
            return {"outputs": {}, "collections": {}}
    
    def _save_registry(self) -> None:
        """
        Save the output registry to file.
        """
        registry_path = self.output_dir / "registry.json"
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def _get_output_path(
        self,
        generation_id: str,
        file_format: str
    ) -> Path:
        """
        Get path for output file.
        
        Args:
            generation_id: Generation ID.
            file_format: File format.
            
        Returns:
            Output file path.
        """
        # Determine subdirectory based on format
        if file_format in [".wav", ".mp3", ".ogg", ".flac"]:
            subdir = "audio"
        elif file_format in [".mid", ".midi"]:
            subdir = "midi"
        else:
            subdir = "text"
            
        return self.output_dir / subdir / f"{generation_id}{file_format}"
    
    async def save_output(
        self,
        generation: GenerationResource,
        data: Any,
        file_format: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save generation output.
        
        Args:
            generation: Generation resource.
            data: Output data.
            file_format: Output format.
            metadata: Optional output metadata.
            
        Returns:
            Path to saved file.
            
        Raises:
            ProcessingError: If saving fails.
        """
        try:
            # Get output path
            output_path = self._get_output_path(
                generation.resource_id,
                file_format
            )
            
            # Save file based on format
            if file_format in [".wav", ".mp3", ".ogg", ".flac"]:
                self.processor_manager.audio.save_audio(
                    data,
                    output_path
                )
            elif file_format in [".mid", ".midi"]:
                self.processor_manager.midi.save_midi(
                    data,
                    output_path
                )
            else:
                # Save text/JSON data
                with open(output_path, "w") as f:
                    if isinstance(data, (dict, list)):
                        json.dump(data, f, indent=2)
                    else:
                        f.write(str(data))
                        
            # Save metadata if provided
            if metadata:
                metadata_path = output_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump({
                        "generation_id": generation.resource_id,
                        "created_at": datetime.datetime.utcnow().isoformat(),
                        **metadata
                    }, f, indent=2)
                    
            return output_path
            
        except Exception as e:
            raise ProcessingError(f"Error saving output: {e}")
            
    async def load_output(
        self,
        generation: GenerationResource
    ) -> Any:
        """
        Load generation output.
        
        Args:
            generation: Generation resource.
            
        Returns:
            Output data.
            
        Raises:
            ResourceNotFoundError: If output not found.
            ProcessingError: If loading fails.
        """
        try:
            if not generation.file_path:
                raise ResourceNotFoundError("Generation has no output file")
                
            file_path = Path(generation.file_path)
            if not file_path.exists():
                raise ResourceNotFoundError(f"Output file not found: {file_path}")
                
            # Load file based on format
            suffix = file_path.suffix.lower()
            if suffix in [".wav", ".mp3", ".ogg", ".flac"]:
                return self.processor_manager.audio.load_audio(file_path)
            elif suffix in [".mid", ".midi"]:
                return self.processor_manager.midi.load_midi(file_path)
            else:
                # Load text/JSON data
                with open(file_path, "r") as f:
                    if suffix == ".json":
                        return json.load(f)
                    else:
                        return f.read()
                        
        except Exception as e:
            raise ProcessingError(f"Error loading output: {e}")
            
    async def convert_output(
        self,
        generation: GenerationResource,
        target_format: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Convert generation output to different format.
        
        Args:
            generation: Generation resource.
            target_format: Target format.
            config: Optional conversion configuration.
            
        Returns:
            Path to converted file.
            
        Raises:
            ResourceNotFoundError: If output not found.
            ValidationError: If conversion not supported.
            ProcessingError: If conversion fails.
        """
        try:
            if not generation.file_path:
                raise ResourceNotFoundError("Generation has no output file")
                
            input_path = Path(generation.file_path)
            if not input_path.exists():
                raise ResourceNotFoundError(f"Output file not found: {input_path}")
                
            # Get output path
            output_path = self._get_output_path(
                generation.resource_id,
                target_format
            )
            
            # Convert based on formats
            input_format = input_path.suffix.lower()
            if input_format in [".wav", ".mp3", ".ogg", ".flac"] and \
               target_format in [".wav", ".mp3", ".ogg", ".flac"]:
                # Audio format conversion
                self.processor_manager.audio.convert_format(
                    input_path,
                    output_path,
                    **(config or {})
                )
            else:
                raise ValidationError(
                    f"Conversion from {input_format} to {target_format} not supported"
                )
                
            return output_path
            
        except Exception as e:
            raise ProcessingError(f"Error converting output: {e}")
            
    async def archive_output(
        self,
        generation: GenerationResource
    ) -> None:
        """
        Archive generation output.
        
        Args:
            generation: Generation resource.
            
        Raises:
            ResourceNotFoundError: If output not found.
            ProcessingError: If archiving fails.
        """
        try:
            if not generation.file_path:
                raise ResourceNotFoundError("Generation has no output file")
                
            file_path = Path(generation.file_path)
            if not file_path.exists():
                raise ResourceNotFoundError(f"Output file not found: {file_path}")
                
            # Move file to archive
            archive_path = self.output_dir / "archive" / file_path.name
            shutil.move(file_path, archive_path)
            
            # Move metadata if exists
            metadata_path = file_path.with_suffix(".json")
            if metadata_path.exists():
                archive_metadata_path = archive_path.with_suffix(".json")
                shutil.move(metadata_path, archive_metadata_path)
                
            # Update generation
            generation.file_path = str(archive_path)
            generation.archive()
            
        except Exception as e:
            raise ProcessingError(f"Error archiving output: {e}")
            
    async def delete_output(
        self,
        generation: GenerationResource
    ) -> None:
        """
        Delete generation output.
        
        Args:
            generation: Generation resource.
            
        Raises:
            ResourceNotFoundError: If output not found.
            ProcessingError: If deletion fails.
        """
        try:
            if not generation.file_path:
                raise ResourceNotFoundError("Generation has no output file")
                
            file_path = Path(generation.file_path)
            if not file_path.exists():
                raise ResourceNotFoundError(f"Output file not found: {file_path}")
                
            # Delete file
            file_path.unlink()
            
            # Delete metadata if exists
            metadata_path = file_path.with_suffix(".json")
            if metadata_path.exists():
                metadata_path.unlink()
                
            # Update generation
            generation.file_path = None
            generation.delete()
            
        except Exception as e:
            raise ProcessingError(f"Error deleting output: {e}")
            
    def get_output_info(
        self,
        generation: GenerationResource
    ) -> Dict[str, Any]:
        """
        Get information about generation output.
        
        Args:
            generation: Generation resource.
            
        Returns:
            Dictionary of output information.
            
        Raises:
            ResourceNotFoundError: If output not found.
        """
        if not generation.file_path:
            raise ResourceNotFoundError("Generation has no output file")
            
        file_path = Path(generation.file_path)
        if not file_path.exists():
            raise ResourceNotFoundError(f"Output file not found: {file_path}")
            
        # Get file information
        info = {
            "path": str(file_path),
            "format": file_path.suffix.lower(),
            "size": file_path.stat().st_size,
            "created_at": datetime.datetime.fromtimestamp(
                file_path.stat().st_ctime
            ).isoformat(),
            "modified_at": datetime.datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat()
        }
        
        # Get format-specific information
        suffix = file_path.suffix.lower()
        if suffix in [".wav", ".mp3", ".ogg", ".flac"]:
            # Audio file analysis
            try:
                info.update({
                    "duration": self.processor_manager.audio.get_duration(file_path),
                    "tempo": self.processor_manager.audio.get_tempo(file_path),
                    "key": self.processor_manager.audio.get_key(file_path)
                })
            except Exception as e:
                self.logger.warning(f"Error analyzing audio file: {e}")
                
        elif suffix in [".mid", ".midi"]:
            # MIDI file analysis
            try:
                midi = self.processor_manager.midi.load_midi(file_path)
                info.update({
                    "duration": self.processor_manager.midi.get_duration(midi),
                    "tempo": self.processor_manager.midi.get_tempo(midi),
                    "key": self.processor_manager.midi.get_key(midi),
                    "time_signature": self.processor_manager.midi.get_time_signature(midi)
                })
            except Exception as e:
                self.logger.warning(f"Error analyzing MIDI file: {e}")
                
        # Get metadata if exists
        metadata_path = file_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    info["metadata"] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading metadata: {e}")
                
        return info
    
    def save_audio_output(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        metadata: Dict,
        output_format: str = "wav",
        visualization: bool = True
    ) -> Dict:
        """
        Save an audio output with metadata.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio data.
            metadata: Metadata dictionary.
            output_format: Format to save the audio in.
            visualization: Whether to generate visualizations.
            
        Returns:
            Output info dictionary.
        """
        try:
            # Generate output ID
            output_id = metadata.get("generation_id", f"output_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{output_id}_{timestamp}.{output_format}"
            
            # Save audio
            audio_path = self.output_dir / "audio" / filename
            self.processor_manager.audio.save_audio(
                audio_data=audio_data,
                output_path=audio_path,
                sr=sample_rate,
                format=output_format
            )
            
            # Generate visualizations if requested
            visualization_paths = {}
            if visualization:
                # Generate waveform visualization
                waveform_path = self.output_dir / "visualizations" / f"{output_id}_waveform.png"
                self._create_waveform_visualization(audio_data, sample_rate, waveform_path)
                visualization_paths["waveform"] = str(waveform_path)
                
                # Generate spectrogram visualization
                spectrogram_path = self.output_dir / "visualizations" / f"{output_id}_spectrogram.png"
                self._create_spectrogram_visualization(audio_data, sample_rate, spectrogram_path)
                visualization_paths["spectrogram"] = str(spectrogram_path)
            
            # Save metadata
            metadata_path = self.output_dir / "metadata" / f"{output_id}.json"
            full_metadata = {
                "output_id": output_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "audio_path": str(audio_path),
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "format": output_format,
                "visualization_paths": visualization_paths,
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update registry
            output_info = {
                "id": output_id,
                "type": "audio",
                "path": str(audio_path),
                "metadata_path": str(metadata_path),
                "visualizations": visualization_paths,
                "timestamp": datetime.datetime.now().isoformat(),
                "tags": metadata.get("tags", [])
            }
            
            self.registry["outputs"][output_id] = output_info
            self._save_registry()
            
            return output_info
        
        except Exception as e:
            print(f"Error saving audio output: {e}")
            raise
    
    def save_midi_output(
        self,
        midi_data,
        metadata: Dict,
        visualization: bool = True
    ) -> Dict:
        """
        Save a MIDI output with metadata.
        
        Args:
            midi_data: MIDI data (music21 Score or mido MidiFile).
            metadata: Metadata dictionary.
            visualization: Whether to generate visualizations.
            
        Returns:
            Output info dictionary.
        """
        try:
            # Generate output ID
            output_id = metadata.get("generation_id", f"output_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{output_id}_{timestamp}.mid"
            
            # Save MIDI
            midi_path = self.output_dir / "midi" / filename
            
            # Determine MIDI type and save accordingly
            if hasattr(midi_data, 'write'):  # music21 Score
                midi_data.write('midi', fp=midi_path)
            else:  # mido MidiFile
                midi_data.save(midi_path)
            
            # Generate visualizations if requested
            visualization_paths = {}
            if visualization:
                # For now, just a placeholder
                # In a real implementation, this would generate piano roll or notation visualizations
                pass
            
            # Save metadata
            metadata_path = self.output_dir / "metadata" / f"{output_id}.json"
            full_metadata = {
                "output_id": output_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "midi_path": str(midi_path),
                "visualization_paths": visualization_paths,
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update registry
            output_info = {
                "id": output_id,
                "type": "midi",
                "path": str(midi_path),
                "metadata_path": str(metadata_path),
                "visualizations": visualization_paths,
                "timestamp": datetime.datetime.now().isoformat(),
                "tags": metadata.get("tags", [])
            }
            
            self.registry["outputs"][output_id] = output_info
            self._save_registry()
            
            return output_info
        
        except Exception as e:
            print(f"Error saving MIDI output: {e}")
            raise
    
    def load_output(self, output_id: str) -> Optional[Dict]:
        """
        Load an output by ID.
        
        Args:
            output_id: ID of the output.
            
        Returns:
            Output dictionary or None if not found.
        """
        # Check if output exists in registry
        if output_id not in self.registry["outputs"]:
            print(f"Output {output_id} not found in registry")
            return None
        
        output_info = self.registry["outputs"][output_id]
        
        try:
            # Load metadata
            metadata_path = output_info.get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load audio or MIDI data
                if output_info["type"] == "audio":
                    audio_path = output_info["path"]
                    if os.path.exists(audio_path):
                        audio_data, sample_rate = self.processor_manager.audio.load_audio(audio_path)
                        return {
                            **metadata,
                            "audio_data": audio_data,
                            "sample_rate": sample_rate
                        }
                    else:
                        print(f"Audio file {audio_path} not found")
                        return metadata
                
                elif output_info["type"] == "midi":
                    midi_path = output_info["path"]
                    if os.path.exists(midi_path):
                        try:
                            from music21 import converter
                            midi_data = converter.parse(midi_path)
                            return {
                                **metadata,
                                "midi_data": midi_data
                            }
                        except Exception as e:
                            print(f"Error loading MIDI file: {e}")
                            return metadata
                    else:
                        print(f"MIDI file {midi_path} not found")
                        return metadata
                
                return metadata
            else:
                print(f"Metadata file {metadata_path} not found")
                return output_info
        
        except Exception as e:
            print(f"Error loading output {output_id}: {e}")
            return output_info
    
    def list_outputs(
        self,
        output_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List outputs with optional filtering.
        
        Args:
            output_type: Type of outputs to list ('audio' or 'midi').
            tags: List of tags to filter by.
            limit: Maximum number of outputs to return.
            offset: Offset for pagination.
            
        Returns:
            List of output info dictionaries.
        """
        outputs = list(self.registry["outputs"].values())
        
        # Filter by type
        if output_type:
            outputs = [o for o in outputs if o["type"] == output_type]
        
        # Filter by tags
        if tags:
            outputs = [o for o in outputs if all(tag in o.get("tags", []) for tag in tags)]
        
        # Sort by timestamp (newest first)
        outputs.sort(key=lambda o: o.get("timestamp", ""), reverse=True)
        
        # Apply pagination
        outputs = outputs[offset:offset + limit]
        
        return outputs
    
    def delete_output(self, output_id: str) -> bool:
        """
        Delete an output.
        
        Args:
            output_id: ID of the output to delete.
            
        Returns:
            True if deletion succeeded, False otherwise.
        """
        if output_id not in self.registry["outputs"]:
            print(f"Output {output_id} not found in registry")
            return False
        
        try:
            output_info = self.registry["outputs"][output_id]
            
            # Delete files
            file_path = output_info.get("path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            metadata_path = output_info.get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Delete visualizations
            for vis_path in output_info.get("visualizations", {}).values():
                if os.path.exists(vis_path):
                    os.remove(vis_path)
            
            # Remove from registry
            del self.registry["outputs"][output_id]
            
            # Remove from collections
            for collection_id, collection in self.registry["collections"].items():
                if "outputs" in collection and output_id in collection["outputs"]:
                    collection["outputs"].remove(output_id)
            
            self._save_registry()
            
            return True
        
        except Exception as e:
            print(f"Error deleting output {output_id}: {e}")
            return False
    
    def tag_output(self, output_id: str, tags: List[str]) -> bool:
        """
        Add tags to an output.
        
        Args:
            output_id: ID of the output.
            tags: List of tags to add.
            
        Returns:
            True if tagging succeeded, False otherwise.
        """
        if output_id not in self.registry["outputs"]:
            print(f"Output {output_id} not found in registry")
            return False
        
        try:
            # Update registry
            current_tags = self.registry["outputs"][output_id].get("tags", [])
            updated_tags = list(set(current_tags + tags))
            self.registry["outputs"][output_id]["tags"] = updated_tags
            
            # Update metadata file
            metadata_path = self.registry["outputs"][output_id].get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["tags"] = updated_tags
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self._save_registry()
            
            return True
        
        except Exception as e:
            print(f"Error tagging output {output_id}: {e}")
            return False
    
    def create_collection(
        self,
        name: str,
        description: str = "",
        output_ids: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a new collection of outputs.
        
        Args:
            name: Name of the collection.
            description: Description of the collection.
            output_ids: List of output IDs to include in the collection.
            
        Returns:
            Collection ID or None if creation failed.
        """
        try:
            # Generate collection ID
            collection_id = f"collection_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create collection
            collection = {
                "id": collection_id,
                "name": name,
                "description": description,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "outputs": output_ids or []
            }
            
            # Add to registry
            self.registry["collections"][collection_id] = collection
            self._save_registry()
            
            # Create collection file
            collection_path = self.output_dir / "collections" / f"{collection_id}.json"
            with open(collection_path, 'w') as f:
                json.dump(collection, f, indent=2)
            
            return collection_id
        
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def add_to_collection(self, collection_id: str, output_ids: List[str]) -> bool:
        """
        Add outputs to a collection.
        
        Args:
            collection_id: ID of the collection.
            output_ids: List of output IDs to add.
            
        Returns:
            True if addition succeeded, False otherwise.
        """
        if collection_id not in self.registry["collections"]:
            print(f"Collection {collection_id} not found in registry")
            return False
        
        try:
            # Update registry
            current_outputs = self.registry["collections"][collection_id].get("outputs", [])
            updated_outputs = list(set(current_outputs + output_ids))
            self.registry["collections"][collection_id]["outputs"] = updated_outputs
            self.registry["collections"][collection_id]["updated_at"] = datetime.datetime.now().isoformat()
            
            # Update collection file
            collection_path = self.output_dir / "collections" / f"{collection_id}.json"
            with open(collection_path, 'w') as f:
                json.dump(self.registry["collections"][collection_id], f, indent=2)
            
            self._save_registry()
            
            return True
        
        except Exception as e:
            print(f"Error adding to collection {collection_id}: {e}")
            return False
    
    def export_output(
        self,
        output_id: str,
        export_path: Union[str, Path],
        format: Optional[str] = None,
        include_metadata: bool = True,
        include_visualizations: bool = False
    ) -> bool:
        """
        Export an output to a file.
        
        Args:
            output_id: ID of the output.
            export_path: Path to export the output to.
            format: Format to export in. If None, use the original format.
            include_metadata: Whether to include metadata in the export.
            include_visualizations: Whether to include visualizations in the export.
            
        Returns:
            True if export succeeded, False otherwise.
        """
        if output_id not in self.registry["outputs"]:
            print(f"Output {output_id} not found in registry")
            return False
        
        try:
            output_info = self.registry["outputs"][output_id]
            source_path = output_info["path"]
            
            if not os.path.exists(source_path):
                print(f"Source file {source_path} not found")
                return False
            
            export_path = Path(export_path)
            os.makedirs(export_path.parent, exist_ok=True)
            
            # Handle format conversion if needed
            if output_info["type"] == "audio":
                if format and format != os.path.splitext(source_path)[1][1:]:
                    # Convert audio format
                    self.processor_manager.audio.convert_format(
                        audio_path=source_path,
                        output_format=format,
                        output_path=export_path
                    )
                else:
                    # Just copy the file
                    shutil.copy2(source_path, export_path)
            else:
                # For MIDI, just copy the file
                shutil.copy2(source_path, export_path)
            
            # Export metadata if requested
            if include_metadata:
                metadata_path = output_info["metadata_path"]
                if os.path.exists(metadata_path):
                    metadata_export_path = export_path.with_suffix('.json')
                    shutil.copy2(metadata_path, metadata_export_path)
            
            # Export visualizations if requested
            if include_visualizations:
                for vis_type, vis_path in output_info.get("visualizations", {}).items():
                    if os.path.exists(vis_path):
                        vis_export_path = export_path.with_stem(f"{export_path.stem}_{vis_type}")
                        vis_export_path = vis_export_path.with_suffix(os.path.splitext(vis_path)[1])
                        shutil.copy2(vis_path, vis_export_path)
            
            return True
        
        except Exception as e:
            print(f"Error exporting output {output_id}: {e}")
            return False
    
    def export_collection(
        self,
        collection_id: str,
        export_dir: Union[str, Path],
        format: Optional[str] = None,
        include_metadata: bool = True,
        include_visualizations: bool = False
    ) -> bool:
        """
        Export a collection of outputs.
        
        Args:
            collection_id: ID of the collection.
            export_dir: Directory to export the collection to.
            format: Format to export in. If None, use the original formats.
            include_metadata: Whether to include metadata in the export.
            include_visualizations: Whether to include visualizations in the export.
            
        Returns:
            True if export succeeded, False otherwise.
        """
        if collection_id not in self.registry["collections"]:
            print(f"Collection {collection_id} not found in registry")
            return False
        
        try:
            collection = self.registry["collections"][collection_id]
            export_dir = Path(export_dir)
            os.makedirs(export_dir, exist_ok=True)
            
            # Export collection metadata
            collection_export_path = export_dir / f"{collection_id}.json"
            with open(collection_export_path, 'w') as f:
                json.dump(collection, f, indent=2)
            
            # Export each output
            for output_id in collection.get("outputs", []):
                if output_id in self.registry["outputs"]:
                    output_info = self.registry["outputs"][output_id]
                    output_type = output_info["type"]
                    
                    # Determine export path
                    if output_type == "audio":
                        output_extension = format or os.path.splitext(output_info["path"])[1][1:]
                        output_export_path = export_dir / f"{output_id}.{output_extension}"
                    else:
                        output_export_path = export_dir / f"{output_id}.mid"
                    
                    # Export output
                    self.export_output(
                        output_id=output_id,
                        export_path=output_export_path,
                        format=format,
                        include_metadata=include_metadata,
                        include_visualizations=include_visualizations
                    )
            
            return True
        
        except Exception as e:
            print(f"Error exporting collection {collection_id}: {e}")
            return False
    
    def _create_waveform_visualization(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_path: Union[str, Path]
    ) -> None:
        """
        Create a waveform visualization of audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio data.
            output_path: Path to save the visualization.
        """
        plt.figure(figsize=(10, 4))
        
        duration = len(audio_data) / sample_rate
        time = np.linspace(0, duration, len(audio_data))
        
        plt.plot(time, audio_data, color='blue', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _create_spectrogram_visualization(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_path: Union[str, Path]
    ) -> None:
        """
        Create a spectrogram visualization of audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio data.
            output_path: Path to save the visualization.
        """
        plt.figure(figsize=(10, 6))
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)), 
            ref=np.max
        )
        
        # Plot spectrogram
        librosa.display.specshow(
            D, 
            sr=sample_rate, 
            x_axis='time', 
            y_axis='log',
            cmap='viridis'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300)
        plt.close()


# Create a global output service instance
output_service = OutputService(
    resource_manager=ResourceManager(),
    processor_manager=ProcessorManager(),
    output_dir=Path(config.get("paths", "output_dir", "outputs"))
)


def get_output_service() -> OutputService:
    """
    Get the global output service instance.
    
    Returns:
        OutputService instance.
    """
    global output_service
    return output_service