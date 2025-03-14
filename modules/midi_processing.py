"""
MIDI Processing Module

This module provides functionalities for MIDI parsing, manipulation, and generation.
It leverages libraries like Music21 for MIDI processing tasks.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import music21
import numpy as np


class MIDIProcessor:
    """
    A class for MIDI processing operations.
    """

    def __init__(self):
        """
        Initialize the MIDIProcessor.
        """
        pass

    def load_midi(self, midi_path: Union[str, Path]) -> music21.stream.Score:
        """
        Load a MIDI file.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            Music21 Score object.
        """
        midi_path = Path(midi_path)
        
        # Check file format
        if midi_path.suffix.lower() in ['.mid', '.midi']:
            return music21.converter.parse(midi_path)
        else:
            raise ValueError(f"Unsupported MIDI format: {midi_path.suffix}")

    def save_midi(
        self, score: music21.stream.Score, output_path: Union[str, Path]
    ) -> Path:
        """
        Save a Music21 Score to a MIDI file.

        Args:
            score: Music21 Score object.
            output_path: Path to save the MIDI file.

        Returns:
            Path to the saved MIDI file.
        """
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Save MIDI file
        score.write('midi', fp=output_path)
        
        print(f"Saved MIDI to: {output_path}")
        return output_path

    def extract_notes(
        self, score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract notes from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            List of note dictionaries.
        """
        notes = []
        
        for part in score.parts:
            for note in part.recurse().notes:
                if note.isNote:
                    notes.append({
                        'pitch': note.pitch.midi,
                        'offset': note.offset,
                        'duration': note.duration.quarterLength,
                        'velocity': note.volume.velocity
                    })
                elif note.isChord:
                    for pitch in note.pitches:
                        notes.append({
                            'pitch': pitch.midi,
                            'offset': note.offset,
                            'duration': note.duration.quarterLength,
                            'velocity': note.volume.velocity
                        })
        
        # Sort notes by offset
        notes.sort(key=lambda x: x['offset'])
        
        return notes

    def extract_chords(
        self, score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract chords from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            List of chord dictionaries.
        """
        chords = []
        
        for part in score.parts:
            for chord in part.recurse().getElementsByClass('Chord'):
                chords.append({
                    'pitches': [p.midi for p in chord.pitches],
                    'offset': chord.offset,
                    'duration': chord.duration.quarterLength,
                    'velocity': chord.volume.velocity
                })
        
        # Sort chords by offset
        chords.sort(key=lambda x: x['offset'])
        
        return chords

    def extract_key(
        self, score: music21.stream.Score
    ) -> Optional[str]:
        """
        Extract key from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            Key as a string, or None if no key is found.
        """
        key = score.analyze('key')
        if key:
            return str(key)
        else:
            return None

    def extract_tempo(
        self, score: music21.stream.Score
    ) -> Optional[float]:
        """
        Extract tempo from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            Tempo in BPM, or None if no tempo is found.
        """
        tempo_elements = score.recurse().getElementsByClass('MetronomeMark')
        if tempo_elements:
            return tempo_elements[0].number
        else:
            return None

    def extract_time_signature(
        self, score: music21.stream.Score
    ) -> Optional[str]:
        """
        Extract time signature from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            Time signature as a string, or None if no time signature is found.
        """
        time_signature_elements = score.recurse().getElementsByClass('TimeSignature')
        if time_signature_elements:
            return str(time_signature_elements[0])
        else:
            return None

    def transpose(
        self, score: music21.stream.Score, semitones: int
    ) -> music21.stream.Score:
        """
        Transpose a Music21 Score.

        Args:
            score: Music21 Score object.
            semitones: Number of semitones to transpose.

        Returns:
            Transposed Music21 Score object.
        """
        return score.transpose(semitones)

    def quantize(
        self, score: music21.stream.Score, resolution: int = 16
    ) -> music21.stream.Score:
        """
        Quantize a Music21 Score.

        Args:
            score: Music21 Score object.
            resolution: Quantization resolution (in divisions per quarter note).

        Returns:
            Quantized Music21 Score object.
        """
        # This is a placeholder for quantization
        # In a real implementation, this would quantize note timings
        quantized_score = score.deepcopy()
        
        # Get grid size in quarter notes
        grid_size = 1.0 / resolution
        
        # Quantize note offsets and durations
        for part in quantized_score.parts:
            for note in part.recurse().notes:
                # Quantize offset
                note.offset = round(note.offset / grid_size) * grid_size
                
                # Quantize duration
                note.duration.quarterLength = round(note.duration.quarterLength / grid_size) * grid_size
                
                # Ensure minimum duration
                if note.duration.quarterLength < grid_size:
                    note.duration.quarterLength = grid_size
        
        return quantized_score

    def create_score_from_notes(
        self, notes: List[Dict], time_signature: str = '4/4', tempo: float = 120.0
    ) -> music21.stream.Score:
        """
        Create a Music21 Score from a list of note dictionaries.

        Args:
            notes: List of note dictionaries.
            time_signature: Time signature as a string.
            tempo: Tempo in BPM.

        Returns:
            Music21 Score object.
        """
        # Create a new score
        score = music21.stream.Score()
        
        # Create a part
        part = music21.stream.Part()
        
        # Add time signature
        ts = music21.meter.TimeSignature(time_signature)
        part.append(ts)
        
        # Add tempo
        mm = music21.tempo.MetronomeMark(number=tempo)
        part.append(mm)
        
        # Add notes
        for note_dict in notes:
            n = music21.note.Note(pitch=note_dict['pitch'])
            n.offset = note_dict['offset']
            n.duration.quarterLength = note_dict['duration']
            n.volume.velocity = note_dict['velocity']
            part.append(n)
        
        # Add part to score
        score.append(part)
        
        return score

    def create_score_from_chords(
        self, chords: List[Dict], time_signature: str = '4/4', tempo: float = 120.0
    ) -> music21.stream.Score:
        """
        Create a Music21 Score from a list of chord dictionaries.

        Args:
            chords: List of chord dictionaries.
            time_signature: Time signature as a string.
            tempo: Tempo in BPM.

        Returns:
            Music21 Score object.
        """
        # Create a new score
        score = music21.stream.Score()
        
        # Create a part
        part = music21.stream.Part()
        
        # Add time signature
        ts = music21.meter.TimeSignature(time_signature)
        part.append(ts)
        
        # Add tempo
        mm = music21.tempo.MetronomeMark(number=tempo)
        part.append(mm)
        
        # Add chords
        for chord_dict in chords:
            chord_pitches = [music21.pitch.Pitch(p) for p in chord_dict['pitches']]
            c = music21.chord.Chord(chord_pitches)
            c.offset = chord_dict['offset']
            c.duration.quarterLength = chord_dict['duration']
            c.volume.velocity = chord_dict['velocity']
            part.append(c)
        
        # Add part to score
        score.append(part)
        
        return score

    def extract_melody(
        self, score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract melody from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            List of melody note dictionaries.
        """
        # This is a placeholder for melody extraction
        # In a real implementation, this would use more sophisticated methods
        
        # Get all notes
        notes = self.extract_notes(score)
        
        # Sort notes by pitch (highest first) and then by offset
        notes.sort(key=lambda x: (-x['pitch'], x['offset']))
        
        # Group notes by offset
        notes_by_offset = {}
        for note in notes:
            if note['offset'] not in notes_by_offset:
                notes_by_offset[note['offset']] = []
            notes_by_offset[note['offset']].append(note)
        
        # Extract highest note at each offset as melody
        melody = []
        for offset in sorted(notes_by_offset.keys()):
            melody.append(notes_by_offset[offset][0])
        
        return melody

    def extract_bass(
        self, score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract bass line from a Music21 Score.

        Args:
            score: Music21 Score object.

        Returns:
            List of bass note dictionaries.
        """
        # This is a placeholder for bass extraction
        # In a real implementation, this would use more sophisticated methods
        
        # Get all notes
        notes = self.extract_notes(score)
        
        # Sort notes by pitch (lowest first) and then by offset
        notes.sort(key=lambda x: (x['pitch'], x['offset']))
        
        # Group notes by offset
        notes_by_offset = {}
        for note in notes:
            if note['offset'] not in notes_by_offset:
                notes_by_offset[note['offset']] = []
            notes_by_offset[note['offset']].append(note)
        
        # Extract lowest note at each offset as bass
        bass = []
        for offset in sorted(notes_by_offset.keys()):
            bass.append(notes_by_offset[offset][0])
        
        return bass

    def midi_to_pianoroll(
        self, score: music21.stream.Score, resolution: int = 16
    ) -> np.ndarray:
        """
        Convert a Music21 Score to a piano roll representation.

        Args:
            score: Music21 Score object.
            resolution: Resolution of the piano roll (in divisions per quarter note).

        Returns:
            Piano roll as a 2D numpy array (time x pitch).
        """
        # Extract notes
        notes = self.extract_notes(score)
        
        if not notes:
            return np.zeros((0, 128))
        
        # Calculate total length
        max_offset = max(note['offset'] + note['duration'] for note in notes)
        total_steps = int(max_offset * resolution) + 1
        
        # Create piano roll
        piano_roll = np.zeros((total_steps, 128))
        
        # Fill piano roll
        for note in notes:
            start_step = int(note['offset'] * resolution)
            end_step = int((note['offset'] + note['duration']) * resolution)
            piano_roll[start_step:end_step + 1, note['pitch']] = note['velocity'] / 127.0
        
        return piano_roll

    def pianoroll_to_midi(
        self, piano_roll: np.ndarray, resolution: int = 16, threshold: float = 0.5
    ) -> music21.stream.Score:
        """
        Convert a piano roll representation to a Music21 Score.

        Args:
            piano_roll: Piano roll as a 2D numpy array (time x pitch).
            resolution: Resolution of the piano roll (in divisions per quarter note).
            threshold: Threshold for note detection.

        Returns:
            Music21 Score object.
        """
        # Create a new score
        score = music21.stream.Score()
        
        # Create a part
        part = music21.stream.Part()
        
        # Add time signature
        ts = music21.meter.TimeSignature('4/4')
        part.append(ts)
        
        # Add tempo
        mm = music21.tempo.MetronomeMark(number=120.0)
        part.append(mm)
        
        # Convert piano roll to notes
        for pitch in range(piano_roll.shape[1]):
            # Find note onsets and offsets
            active = piano_roll[:, pitch] > threshold
            changes = np.diff(active.astype(int))
            note_starts = np.where(changes > 0)[0]
            note_ends = np.where(changes < 0)[0]
            
            # Create notes
            for start, end in zip(note_starts, note_ends):
                # Calculate offset and duration
                offset = start / resolution
                duration = (end - start) / resolution
                
                # Get velocity
                velocity = int(127 * np.max(piano_roll[start:end + 1, pitch]))
                
                # Create note
                n = music21.note.Note(pitch=pitch)
                n.offset = offset
                n.duration.quarterLength = duration
                n.volume.velocity = velocity
                
                # Add note to part
                part.append(n)
        
        # Add part to score
        score.append(part)
        
        return score

    def visualize_midi(
        self, score: music21.stream.Score, output_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Visualize a Music21 Score as sheet music.

        Args:
            score: Music21 Score object.
            output_path: Path to save the visualization.

        Returns:
            Path to the saved visualization, or None if display only.
        """
        if output_path:
            output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            # Save visualization
            score.write('musicxml.png', fp=output_path)
            
            print(f"Saved MIDI visualization to: {output_path}")
            return output_path
        else:
            # Display visualization
            score.show()
            return None


# Example usage
if __name__ == "__main__":
    midi_processor = MIDIProcessor()
    # Example: score = midi_processor.load_midi("path/to/midi.mid")