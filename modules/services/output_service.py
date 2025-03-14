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
from typing import Dict, List, Optional, Set, Union

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from modules.core.config import config
from modules.core.audio_processor import AudioProcessor


class OutputService:
    """
    Service for managing, organizing, and exporting generated music outputs.
    
    This class provides functionalities for saving, loading, organizing,
    and exporting generated music outputs.
    """
    
    def __init__(self):
        """
        Initialize the OutputService.
        """
        self.output_dir = Path(config.get("paths", "output_dir", "outputs"))
        self.audio_dir = self.output_dir / "audio"
        self.midi_dir = self.output_dir / "midi"
        self.metadata_dir = self.output_dir / "metadata"
        self.visualization_dir = self.output_dir / "visualizations"
        self.collections_dir = self.output_dir / "collections"
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.midi_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.collections_dir, exist_ok=True)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Output registry
        self.registry = self._load_registry()
    
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
            audio_path = self.audio_dir / filename
            self.audio_processor.save_audio(
                audio_data=audio_data,
                output_path=audio_path,
                sr=sample_rate,
                format=output_format
            )
            
            # Generate visualizations if requested
            visualization_paths = {}
            if visualization:
                # Generate waveform visualization
                waveform_path = self.visualization_dir / f"{output_id}_waveform.png"
                self._create_waveform_visualization(audio_data, sample_rate, waveform_path)
                visualization_paths["waveform"] = str(waveform_path)
                
                # Generate spectrogram visualization
                spectrogram_path = self.visualization_dir / f"{output_id}_spectrogram.png"
                self._create_spectrogram_visualization(audio_data, sample_rate, spectrogram_path)
                visualization_paths["spectrogram"] = str(spectrogram_path)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{output_id}.json"
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
            midi_path = self.midi_dir / filename
            
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
            metadata_path = self.metadata_dir / f"{output_id}.json"
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
                        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
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
            collection_path = self.collections_dir / f"{collection_id}.json"
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
            collection_path = self.collections_dir / f"{collection_id}.json"
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
                    self.audio_processor.convert_format(
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
output_service = OutputService()


def get_output_service() -> OutputService:
    """
    Get the global output service instance.
    
    Returns:
        OutputService instance.
    """
    global output_service
    return output_service