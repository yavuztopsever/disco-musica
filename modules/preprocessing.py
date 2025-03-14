"""
Data Preprocessing Module

This module performs audio and MIDI preprocessing, including format conversion, segmentation,
feature extraction, and stem separation. It also offers tools for dataset creation, splitting,
and augmentation.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import music21
from pydub import AudioSegment
from sklearn.model_selection import train_test_split


class AudioPreprocessor:
    """
    A class for audio preprocessing operations.
    """

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the AudioPreprocessor.

        Args:
            sample_rate: Sample rate for audio processing.
            n_fft: FFT window size for spectrogram computation.
            hop_length: Hop length for spectrogram computation.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def convert_format(
        self, audio_path: Union[str, Path], target_format: str = "wav"
    ) -> Path:
        """
        Convert audio file to a specified format.

        Args:
            audio_path: Path to the audio file.
            target_format: Target audio format.

        Returns:
            Path to the converted audio file.
        """
        audio_path = Path(audio_path)
        output_path = audio_path.with_suffix(f".{target_format}")
        
        audio = AudioSegment.from_file(audio_path)
        audio.export(output_path, format=target_format)
        
        print(f"Converted audio to {target_format}: {output_path}")
        return output_path

    def segment_audio(
        self, audio_path: Union[str, Path], segment_length: float = 30.0, overlap: float = 0.0
    ) -> List[np.ndarray]:
        """
        Segment audio file into smaller chunks.

        Args:
            audio_path: Path to the audio file.
            segment_length: Length of each segment in seconds.
            overlap: Overlap between segments in seconds.

        Returns:
            List of audio segments as numpy arrays.
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate segment and hop lengths in samples
        segment_samples = int(segment_length * sr)
        hop_samples = int((segment_length - overlap) * sr)
        
        # Segment audio
        segments = []
        for i in range(0, len(y) - segment_samples + 1, hop_samples):
            segment = y[i:i + segment_samples]
            segments.append(segment)
        
        print(f"Segmented audio into {len(segments)} chunks")
        return segments

    def extract_features(
        self, audio: Union[str, Path, np.ndarray], feature_type: str = "mel_spectrogram"
    ) -> np.ndarray:
        """
        Extract audio features from an audio file or array.

        Args:
            audio: Path to the audio file or audio array.
            feature_type: Type of feature to extract ('mel_spectrogram', 'mfcc', 'chroma', etc.).

        Returns:
            Extracted features as a numpy array.
        """
        # Load audio if a path is provided
        if isinstance(audio, (str, Path)):
            y, sr = librosa.load(audio, sr=self.sample_rate)
        else:
            y = audio
            sr = self.sample_rate
        
        # Extract features based on the specified type
        if feature_type == "mel_spectrogram":
            features = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            # Convert to dB scale
            features = librosa.power_to_db(features, ref=np.max)
        elif feature_type == "mfcc":
            features = librosa.feature.mfcc(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=13
            )
        elif feature_type == "chroma":
            features = librosa.feature.chroma_stft(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        return features

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to a consistent range.

        Args:
            audio: Audio array to normalize.

        Returns:
            Normalized audio array.
        """
        return librosa.util.normalize(audio)

    def trim_silence(self, audio: np.ndarray, top_db: int = 60) -> np.ndarray:
        """
        Trim silence from the beginning and end of an audio array.

        Args:
            audio: Audio array to trim.
            top_db: Threshold for silence detection in dB.

        Returns:
            Trimmed audio array.
        """
        return librosa.effects.trim(audio, top_db=top_db)[0]


class MIDIPreprocessor:
    """
    A class for MIDI preprocessing operations.
    """

    def __init__(self):
        """
        Initialize the MIDIPreprocessor.
        """
        pass

    def standardize_midi(self, midi_path: Union[str, Path]) -> Path:
        """
        Standardize MIDI file format and encoding.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            Path to the standardized MIDI file.
        """
        midi_path = Path(midi_path)
        output_path = midi_path.with_stem(f"{midi_path.stem}_standardized")
        
        # Parse MIDI file
        midi = music21.converter.parse(midi_path)
        
        # Standardize MIDI (placeholder for actual standardization logic)
        # In a real implementation, this would standardize the MIDI file
        
        # Write standardized MIDI
        midi.write('midi', output_path)
        
        print(f"Standardized MIDI: {output_path}")
        return output_path

    def tokenize_midi(self, midi_path: Union[str, Path]) -> List[int]:
        """
        Convert MIDI data into a sequence of discrete tokens.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            List of tokens representing the MIDI file.
        """
        # Parse MIDI file
        midi = music21.converter.parse(midi_path)
        
        # Tokenize MIDI (placeholder for actual tokenization logic)
        # In a real implementation, this would convert MIDI events to tokens
        tokens = []
        
        # Example: Extract note events and convert to tokens
        for part in midi.parts:
            for note in part.recurse().notes:
                if note.isNote:
                    # Token format: (note pitch, offset, duration, velocity)
                    token = (note.pitch.midi, note.offset, note.duration.quarterLength, note.volume.velocity)
                    tokens.append(token)
                elif note.isChord:
                    # Token format: (chord pitches, offset, duration, velocity)
                    token = ([p.midi for p in note.pitches], note.offset, note.duration.quarterLength, note.volume.velocity)
                    tokens.append(token)
        
        print(f"Tokenized MIDI into {len(tokens)} tokens")
        return tokens

    def quantize_midi(
        self, midi_path: Union[str, Path], resolution: int = 16
    ) -> Path:
        """
        Quantize note timings to align with a musical grid.

        Args:
            midi_path: Path to the MIDI file.
            resolution: Quantization resolution (in divisions per quarter note).

        Returns:
            Path to the quantized MIDI file.
        """
        midi_path = Path(midi_path)
        output_path = midi_path.with_stem(f"{midi_path.stem}_quantized")
        
        # Parse MIDI file
        midi = music21.converter.parse(midi_path)
        
        # Quantize MIDI (placeholder for actual quantization logic)
        # In a real implementation, this would quantize note timings
        
        # Write quantized MIDI
        midi.write('midi', output_path)
        
        print(f"Quantized MIDI: {output_path}")
        return output_path


class DatasetManager:
    """
    A class for dataset creation and management.
    """

    def __init__(self, dataset_dir: str = "data/datasets"):
        """
        Initialize the DatasetManager.

        Args:
            dataset_dir: Directory to store the datasets.
        """
        self.dataset_dir = Path(dataset_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)

    def create_dataset(
        self, name: str, data_paths: List[Union[str, Path]], metadata: Optional[Dict] = None
    ) -> Path:
        """
        Create a dataset from a list of data paths.

        Args:
            name: Name of the dataset.
            data_paths: List of paths to the data files.
            metadata: Optional metadata for the dataset.

        Returns:
            Path to the created dataset directory.
        """
        dataset_path = self.dataset_dir / name
        os.makedirs(dataset_path, exist_ok=True)
        
        # Copy data files to the dataset directory
        import shutil
        for path in data_paths:
            path = Path(path)
            shutil.copy2(path, dataset_path / path.name)
        
        # Save metadata
        if metadata:
            import json
            with open(dataset_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Created dataset: {dataset_path}")
        return dataset_path

    def split_dataset(
        self, dataset_path: Union[str, Path], test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split a dataset into training, validation, and testing sets.

        Args:
            dataset_path: Path to the dataset directory.
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the dataset to include in the validation split.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_paths, val_paths, test_paths).
        """
        dataset_path = Path(dataset_path)
        
        # Get all data files in the dataset directory
        data_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.mid", "*.midi"]:
            data_files.extend(list(dataset_path.glob(ext)))
        
        # Split into train, validation, and test sets
        train_val_files, test_files = train_test_split(data_files, test_size=test_size, random_state=seed)
        val_ratio = val_size / (1 - test_size)
        train_files, val_files = train_test_split(train_val_files, test_size=val_ratio, random_state=seed)
        
        # Create subdirectories
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        test_dir = dataset_path / "test"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy files to subdirectories
        import shutil
        train_paths = []
        for file in train_files:
            target_path = train_dir / file.name
            shutil.copy2(file, target_path)
            train_paths.append(target_path)
        
        val_paths = []
        for file in val_files:
            target_path = val_dir / file.name
            shutil.copy2(file, target_path)
            val_paths.append(target_path)
        
        test_paths = []
        for file in test_files:
            target_path = test_dir / file.name
            shutil.copy2(file, target_path)
            test_paths.append(target_path)
        
        print(f"Split dataset into {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test files")
        return train_paths, val_paths, test_paths


# Example usage
if __name__ == "__main__":
    audio_preprocessor = AudioPreprocessor()
    midi_preprocessor = MIDIPreprocessor()
    dataset_manager = DatasetManager()
    # Example: audio_preprocessor.convert_format("path/to/audio.mp3", "wav")