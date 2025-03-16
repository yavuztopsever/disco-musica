"""MIDI Processor for handling MIDI data processing."""

import numpy as np
import pretty_midi
import torch
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from copy import deepcopy

from ..exceptions.base_exceptions import (
    ProcessingError,
    ValidationError
)


class MIDIProcessor:
    """Processor for handling MIDI data.
    
    This class provides functionality for processing MIDI data, including
    event encoding/decoding, feature extraction, and file manipulation.
    """
    
    def __init__(self):
        """Initialize the MIDI processor."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # MIDI constants
        self.NOTE_ON = 0
        self.NOTE_OFF = 1
        self.TIME_SHIFT = 2
        self.VELOCITY = 3
        
        # Event ranges
        self.NUM_NOTES = 128
        self.NUM_VELOCITIES = 128
        self.NUM_TIME_SHIFTS = 100  # Quantized time shifts
        
        # Event dimensions
        self.event_dims = {
            "note": self.NUM_NOTES,
            "velocity": self.NUM_VELOCITIES,
            "time_shift": self.NUM_TIME_SHIFTS
        }
        
        # Time quantization
        self.time_shift_bins = np.linspace(0, 2.0, self.NUM_TIME_SHIFTS)  # Max 2 second shift
        
    def load_midi(self, midi_path: Union[str, Path]) -> pretty_midi.PrettyMIDI:
        """Load MIDI file.
        
        Args:
            midi_path: Path to MIDI file.
            
        Returns:
            PrettyMIDI object.
            
        Raises:
            ProcessingError: If loading fails.
        """
        try:
            return pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            raise ProcessingError(f"Error loading MIDI file: {e}")
            
    def save_midi(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: Union[str, Path]
    ) -> None:
        """Save MIDI file.
        
        Args:
            midi_data: PrettyMIDI object.
            output_path: Path to save MIDI file.
            
        Raises:
            ProcessingError: If saving fails.
        """
        try:
            midi_data.write(str(output_path))
        except Exception as e:
            raise ProcessingError(f"Error saving MIDI file: {e}")
            
    def encode_midi(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        max_events: Optional[int] = None
    ) -> torch.Tensor:
        """Encode MIDI data into event sequence.
        
        Args:
            midi_data: PrettyMIDI object.
            max_events: Maximum number of events.
            
        Returns:
            Tensor of encoded events [batch_size, seq_len].
            
        Raises:
            ProcessingError: If encoding fails.
        """
        try:
            events = []
            current_time = 0.0
            
            # Get all notes sorted by start time
            notes = []
            for instrument in midi_data.instruments:
                notes.extend(instrument.notes)
            notes.sort(key=lambda x: x.start)
            
            # Process notes
            for note in notes:
                # Add time shift
                time_shift = note.start - current_time
                if time_shift > 0:
                    time_shift_idx = np.digitize(time_shift, self.time_shift_bins) - 1
                    events.append(self.TIME_SHIFT + time_shift_idx)
                    current_time = note.start
                
                # Add note on
                events.append(self.NOTE_ON + note.pitch)
                
                # Add velocity
                velocity_idx = min(note.velocity, self.NUM_VELOCITIES - 1)
                events.append(self.VELOCITY + velocity_idx)
                
                # Add note off (at end time)
                time_shift = note.end - current_time
                if time_shift > 0:
                    time_shift_idx = np.digitize(time_shift, self.time_shift_bins) - 1
                    events.append(self.TIME_SHIFT + time_shift_idx)
                    current_time = note.end
                events.append(self.NOTE_OFF + note.pitch)
            
            # Truncate if needed
            if max_events is not None:
                events = events[:max_events]
            
            # Convert to tensor
            return torch.tensor(events).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            raise ProcessingError(f"Error encoding MIDI: {e}")
            
    def decode_midi(
        self,
        events: torch.Tensor,
        program: int = 0
    ) -> pretty_midi.PrettyMIDI:
        """Decode event sequence into MIDI data.
        
        Args:
            events: Tensor of encoded events [batch_size, seq_len].
            program: MIDI program number.
            
        Returns:
            PrettyMIDI object.
            
        Raises:
            ProcessingError: If decoding fails.
        """
        try:
            # Create MIDI object
            midi_data = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=program)
            
            # State variables
            current_time = 0.0
            active_notes = {}  # pitch -> (start_time, velocity)
            
            # Process events
            events = events.squeeze(0).numpy()  # Remove batch dimension
            for event in events:
                if event >= self.VELOCITY:
                    # Velocity event
                    velocity = event - self.VELOCITY
                elif event >= self.TIME_SHIFT:
                    # Time shift event
                    shift_idx = event - self.TIME_SHIFT
                    time_shift = self.time_shift_bins[shift_idx]
                    current_time += time_shift
                elif event >= self.NOTE_OFF:
                    # Note off event
                    pitch = event - self.NOTE_OFF
                    if pitch in active_notes:
                        start_time, velocity = active_notes[pitch]
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=current_time
                        )
                        instrument.notes.append(note)
                        del active_notes[pitch]
                else:
                    # Note on event
                    pitch = event - self.NOTE_ON
                    active_notes[pitch] = (current_time, 0)  # Velocity will be set by next event
            
            # Add instrument
            midi_data.instruments.append(instrument)
            
            return midi_data
            
        except Exception as e:
            raise ProcessingError(f"Error decoding MIDI: {e}")
            
    def extract_features(
        self,
        midi_data: pretty_midi.PrettyMIDI
    ) -> Dict[str, Any]:
        """Extract features from MIDI data.
        
        Args:
            midi_data: PrettyMIDI object.
            
        Returns:
            Dictionary of features.
            
        Raises:
            ProcessingError: If feature extraction fails.
        """
        try:
            features = {}
            
            # Basic features
            features["duration"] = midi_data.get_end_time()
            features["tempo"] = midi_data.estimate_tempo()
            features["key"] = midi_data.estimate_key()
            features["time_signature"] = midi_data.time_signature_changes[0] if midi_data.time_signature_changes else None
            
            # Note statistics
            all_notes = []
            for instrument in midi_data.instruments:
                all_notes.extend(instrument.notes)
            
            if all_notes:
                pitches = [note.pitch for note in all_notes]
                velocities = [note.velocity for note in all_notes]
                durations = [note.end - note.start for note in all_notes]
                
                features.update({
                    "num_notes": len(all_notes),
                    "pitch_mean": np.mean(pitches),
                    "pitch_std": np.std(pitches),
                    "velocity_mean": np.mean(velocities),
                    "velocity_std": np.std(velocities),
                    "duration_mean": np.mean(durations),
                    "duration_std": np.std(durations),
                    "pitch_range": max(pitches) - min(pitches)
                })
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Error extracting features: {e}")
            
    def quantize_timing(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        grid: float = 0.25
    ) -> pretty_midi.PrettyMIDI:
        """Quantize note timing to grid.
        
        Args:
            midi_data: PrettyMIDI object.
            grid: Grid size in seconds.
            
        Returns:
            Quantized PrettyMIDI object.
            
        Raises:
            ProcessingError: If quantization fails.
        """
        try:
            # Create new MIDI object
            quantized = deepcopy(midi_data)
            
            # Quantize each instrument
            for instrument in quantized.instruments:
                for note in instrument.notes:
                    # Quantize start and end times
                    note.start = round(note.start / grid) * grid
                    note.end = round(note.end / grid) * grid
                    
                    # Ensure minimum duration
                    if note.end <= note.start:
                        note.end = note.start + grid
            
            return quantized
            
        except Exception as e:
            raise ProcessingError(f"Error quantizing MIDI: {e}")
            
    def transpose(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        semitones: int
    ) -> pretty_midi.PrettyMIDI:
        """Transpose MIDI by number of semitones.
        
        Args:
            midi_data: PrettyMIDI object.
            semitones: Number of semitones to transpose.
            
        Returns:
            Transposed PrettyMIDI object.
            
        Raises:
            ProcessingError: If transposition fails.
        """
        try:
            # Create new MIDI object
            transposed = deepcopy(midi_data)
            
            # Transpose each instrument
            for instrument in transposed.instruments:
                for note in instrument.notes:
                    # Transpose pitch
                    note.pitch += semitones
                    
                    # Ensure pitch is in valid range
                    while note.pitch < 0:
                        note.pitch += 12
                    while note.pitch > 127:
                        note.pitch -= 12
            
            return transposed
            
        except Exception as e:
            raise ProcessingError(f"Error transposing MIDI: {e}")
            
    def adjust_tempo(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        tempo_multiplier: float
    ) -> pretty_midi.PrettyMIDI:
        """Adjust tempo by multiplier.
        
        Args:
            midi_data: PrettyMIDI object.
            tempo_multiplier: Tempo multiplier.
            
        Returns:
            Tempo-adjusted PrettyMIDI object.
            
        Raises:
            ProcessingError: If tempo adjustment fails.
        """
        try:
            # Create new MIDI object
            adjusted = deepcopy(midi_data)
            
            # Adjust tempo changes
            for tempo_change in adjusted.tempo_changes:
                tempo_change.tempo *= tempo_multiplier
            
            # Adjust note timing
            for instrument in adjusted.instruments:
                for note in instrument.notes:
                    note.start /= tempo_multiplier
                    note.end /= tempo_multiplier
            
            return adjusted
            
        except Exception as e:
            raise ProcessingError(f"Error adjusting tempo: {e}")
            
    def merge_tracks(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        track_indices: Optional[List[int]] = None
    ) -> pretty_midi.PrettyMIDI:
        """Merge multiple tracks into single track.
        
        Args:
            midi_data: PrettyMIDI object.
            track_indices: List of track indices to merge. If None, merge all.
            
        Returns:
            PrettyMIDI object with merged tracks.
            
        Raises:
            ProcessingError: If merging fails.
        """
        try:
            # Create new MIDI object
            merged = pretty_midi.PrettyMIDI()
            
            # Create merged instrument
            merged_instrument = pretty_midi.Instrument(program=0)
            
            # Get tracks to merge
            if track_indices is None:
                tracks = midi_data.instruments
            else:
                tracks = [midi_data.instruments[i] for i in track_indices]
            
            # Merge notes
            for track in tracks:
                merged_instrument.notes.extend(deepcopy(track.notes))
            
            # Sort notes by start time
            merged_instrument.notes.sort(key=lambda x: x.start)
            
            # Add merged instrument
            merged.instruments.append(merged_instrument)
            
            return merged
            
        except Exception as e:
            raise ProcessingError(f"Error merging tracks: {e}")
            
    def get_memory_usage(self) -> float:
        """Get processor memory usage.
        
        Returns:
            Memory usage in bytes.
        """
        # Estimate memory usage of internal data structures
        memory = 0
        
        # Event dimensions dictionary
        memory += sum(8 for _ in self.event_dims.values())  # 8 bytes per int
        
        # Time shift bins array
        memory += self.NUM_TIME_SHIFTS * 8  # 8 bytes per float64
        
        return memory 