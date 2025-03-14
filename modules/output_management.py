"""
Output Management Module

This module manages generated music outputs, including saving, organizing, tagging, and exporting.
It provides functionalities for audio visualization, MIDI display, and symbolic notation.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class OutputManager:
    """
    A class for managing generated music outputs.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the OutputManager.

        Args:
            output_dir: Directory to store the outputs.
        """
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.midi_dir = self.output_dir / "midi"
        self.visualization_dir = self.output_dir / "visualizations"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.midi_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, filename: Optional[str] = None, 
        format: str = "wav", tags: Optional[Dict] = None
    ) -> Path:
        """
        Save audio data to a file.

        Args:
            audio_data: Audio data to save.
            sample_rate: Sample rate of the audio data.
            filename: Name of the output file.
            format: Audio format.
            tags: Optional tags for the audio file.

        Returns:
            Path to the saved audio file.
        """
        import soundfile as sf
        
        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"audio_{timestamp}.{format}"
        
        # Ensure the filename has the correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        # Determine output path
        output_path = self.audio_dir / filename
        
        # Save audio file
        sf.write(output_path, audio_data, sample_rate)
        
        # Save metadata if tags are provided
        if tags:
            metadata_path = self.metadata_dir / f"{output_path.stem}.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "type": "audio",
                    "path": str(output_path),
                    "sample_rate": sample_rate,
                    "format": format,
                    "tags": tags,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        
        print(f"Saved audio to: {output_path}")
        return output_path

    def save_midi(
        self, midi_data, filename: Optional[str] = None, tags: Optional[Dict] = None
    ) -> Path:
        """
        Save MIDI data to a file.

        Args:
            midi_data: MIDI data to save (Music21 Score object).
            filename: Name of the output file.
            tags: Optional tags for the MIDI file.

        Returns:
            Path to the saved MIDI file.
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"midi_{timestamp}.mid"
        
        # Ensure the filename has the correct extension
        if not filename.endswith(".mid") and not filename.endswith(".midi"):
            filename = f"{filename}.mid"
        
        # Determine output path
        output_path = self.midi_dir / filename
        
        # Save MIDI file
        midi_data.write('midi', fp=output_path)
        
        # Save metadata if tags are provided
        if tags:
            metadata_path = self.metadata_dir / f"{output_path.stem}.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "type": "midi",
                    "path": str(output_path),
                    "tags": tags,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        
        print(f"Saved MIDI to: {output_path}")
        return output_path

    def visualize_audio_waveform(
        self, audio_data: np.ndarray, sample_rate: int, output_path: Optional[Union[str, Path]] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize audio data as a waveform.

        Args:
            audio_data: Audio data to visualize.
            sample_rate: Sample rate of the audio data.
            output_path: Path to save the visualization.

        Returns:
            Matplotlib figure or None if saved to file.
        """
        # Create figure
        plt.figure(figsize=(10, 4))
        
        # Plot waveform
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.title("Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        if output_path:
            # Determine output path
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save figure
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved waveform visualization to: {output_path}")
            return None
        else:
            # Return figure
            return plt.gcf()

    def visualize_audio_spectrogram(
        self, audio_data: np.ndarray, sample_rate: int, output_path: Optional[Union[str, Path]] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize audio data as a spectrogram.

        Args:
            audio_data: Audio data to visualize.
            sample_rate: Sample rate of the audio data.
            output_path: Path to save the visualization.

        Returns:
            Matplotlib figure or None if saved to file.
        """
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        
        # Create figure
        plt.figure(figsize=(10, 4))
        
        # Plot spectrogram
        librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        
        if output_path:
            # Determine output path
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save figure
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved spectrogram visualization to: {output_path}")
            return None
        else:
            # Return figure
            return plt.gcf()

    def visualize_midi_pianoroll(
        self, midi_data, output_path: Optional[Union[str, Path]] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize MIDI data as a piano roll.

        Args:
            midi_data: MIDI data to visualize (Music21 Score object).
            output_path: Path to save the visualization.

        Returns:
            Matplotlib figure or None if saved to file.
        """
        # Convert to piano roll
        import music21
        
        # Get all notes
        notes = []
        for part in midi_data.parts:
            for note in part.recurse().notes:
                if note.isNote:
                    notes.append({
                        'pitch': note.pitch.midi,
                        'offset': note.offset,
                        'duration': note.duration.quarterLength,
                        'velocity': note.volume.velocity if note.volume.velocity else 64
                    })
                elif note.isChord:
                    for pitch in note.pitches:
                        notes.append({
                            'pitch': pitch.midi,
                            'offset': note.offset,
                            'duration': note.duration.quarterLength,
                            'velocity': note.volume.velocity if note.volume.velocity else 64
                        })
        
        if not notes:
            print("No notes found in MIDI data")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot piano roll
        for note in notes:
            # Time range
            x = [note['offset'], note['offset'] + note['duration']]
            # Note pitch
            y = [note['pitch'], note['pitch']]
            # Velocity determines color intensity
            color = [0, 0, 1, note['velocity'] / 127]
            
            plt.plot(x, y, color=color, linewidth=2)
        
        plt.title("MIDI Piano Roll")
        plt.xlabel("Time (quarter notes)")
        plt.ylabel("MIDI Note")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        if output_path:
            # Determine output path
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save figure
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved piano roll visualization to: {output_path}")
            return None
        else:
            # Return figure
            return plt.gcf()

    def tag_output(
        self, output_path: Union[str, Path], tags: Dict
    ) -> None:
        """
        Add tags to an output file.

        Args:
            output_path: Path to the output file.
            tags: Tags to add.
        """
        output_path = Path(output_path)
        
        # Get metadata path
        metadata_path = self.metadata_dir / f"{output_path.stem}.json"
        
        if metadata_path.exists():
            # Load existing metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Update tags
            if "tags" in metadata:
                metadata["tags"].update(tags)
            else:
                metadata["tags"] = tags
        else:
            # Create new metadata
            metadata = {
                "path": str(output_path),
                "tags": tags,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Determine type
            if output_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                metadata["type"] = "audio"
            elif output_path.suffix.lower() in ['.mid', '.midi']:
                metadata["type"] = "midi"
            else:
                metadata["type"] = "other"
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Added tags to: {output_path}")

    def search_outputs(
        self, query: Dict
    ) -> List[Dict]:
        """
        Search for outputs based on tags.

        Args:
            query: Query dictionary.

        Returns:
            List of matching outputs.
        """
        results = []
        
        # List all metadata files
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        for metadata_path in metadata_files:
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Check if metadata matches query
            match = True
            for key, value in query.items():
                if key == "tags":
                    # Check if all query tags are in metadata tags
                    if "tags" not in metadata or not all(k in metadata["tags"] and metadata["tags"][k] == v for k, v in value.items()):
                        match = False
                        break
                elif key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                results.append(metadata)
        
        return results

    def create_playlist(
        self, name: str, outputs: List[Union[str, Path]]
    ) -> Path:
        """
        Create a playlist of outputs.

        Args:
            name: Name of the playlist.
            outputs: List of output paths.

        Returns:
            Path to the playlist file.
        """
        # Ensure the name has the correct extension
        if not name.endswith(".json"):
            name = f"{name}.json"
        
        # Determine playlist path
        playlist_path = self.output_dir / "playlists" / name
        
        # Create directory if it doesn't exist
        os.makedirs(playlist_path.parent, exist_ok=True)
        
        # Convert paths to strings
        output_paths = [str(Path(output)) for output in outputs]
        
        # Create playlist
        playlist = {
            "name": name,
            "outputs": output_paths,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save playlist
        with open(playlist_path, "w") as f:
            json.dump(playlist, f, indent=2)
        
        print(f"Created playlist: {playlist_path}")
        return playlist_path

    def export_output(
        self, output_path: Union[str, Path], export_path: Union[str, Path], 
        format: Optional[str] = None
    ) -> Path:
        """
        Export an output file to a different format.

        Args:
            output_path: Path to the output file.
            export_path: Path to export the file to.
            format: Export format.

        Returns:
            Path to the exported file.
        """
        import shutil
        
        output_path = Path(output_path)
        export_path = Path(export_path)
        
        # Create directory if it doesn't exist
        os.makedirs(export_path.parent, exist_ok=True)
        
        # Determine format
        if format is None:
            format = export_path.suffix.lstrip('.')
        
        # Check if format conversion is needed
        if output_path.suffix.lstrip('.') == format:
            # No conversion needed, just copy the file
            shutil.copy2(output_path, export_path)
        else:
            # Perform format conversion
            if output_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg'] and format in ['wav', 'mp3', 'flac', 'ogg']:
                # Audio format conversion
                from pydub import AudioSegment
                
                # Load audio file
                audio = AudioSegment.from_file(output_path)
                
                # Export to new format
                audio.export(export_path, format=format)
            elif output_path.suffix.lower() in ['.mid', '.midi'] and format in ['mid', 'midi']:
                # MIDI format conversion (essentially just copying)
                shutil.copy2(output_path, export_path)
            else:
                raise ValueError(f"Unsupported format conversion: {output_path.suffix} to {format}")
        
        print(f"Exported {output_path} to {export_path}")
        return export_path

    def watermark_audio(
        self, audio_data: np.ndarray, sample_rate: int, watermark_text: str
    ) -> np.ndarray:
        """
        Add a watermark to audio data.

        Args:
            audio_data: Audio data to watermark.
            sample_rate: Sample rate of the audio data.
            watermark_text: Watermark text.

        Returns:
            Watermarked audio data.
        """
        # This is a placeholder for audio watermarking
        # In a real implementation, this would add a watermark to the audio
        
        # For now, just add a simple text-to-speech watermark at the beginning
        from pydub import AudioSegment
        from pydub.playback import play
        import io
        import tempfile
        import os
        
        # Save the audio data to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp.wav")
        import soundfile as sf
        sf.write(temp_path, audio_data, sample_rate)
        
        # Load the audio using pydub
        original_audio = AudioSegment.from_file(temp_path)
        
        # Create a silent segment
        silence = AudioSegment.silent(duration=500)  # 500ms of silence
        
        # Combine silence and original audio
        watermarked_audio = silence + original_audio
        
        # Export the audio to a temporary file
        watermarked_path = os.path.join(temp_dir, "watermarked.wav")
        watermarked_audio.export(watermarked_path, format="wav")
        
        # Load the watermarked audio
        watermarked_data, _ = librosa.load(watermarked_path, sr=sample_rate)
        
        # Clean up temporary files
        os.remove(temp_path)
        os.remove(watermarked_path)
        os.rmdir(temp_dir)
        
        print(f"Added watermark: {watermark_text}")
        return watermarked_data


# Example usage
if __name__ == "__main__":
    output_manager = OutputManager()
    # Example: output_manager.save_audio(audio_data, 44100, "my_audio.wav")