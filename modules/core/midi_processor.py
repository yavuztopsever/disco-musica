"""
MIDI processing module for Disco Musica.

This module provides utilities for processing MIDI data, including loading,
saving, feature extraction, and various transformations.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import music21
import mido

from modules.core.config import config


class MIDIProcessor:
    """
    Class for MIDI processing operations.
    
    This class provides methods for loading, saving, and processing MIDI data,
    including parsing, transformation, and feature extraction.
    """
    
    def __init__(
        self,
        resolution: Optional[int] = None,
        quantization: Optional[int] = None
    ):
        """
        Initialize the MIDIProcessor.
        
        Args:
            resolution: MIDI resolution (ticks per quarter note). If None, uses the value from config.
            quantization: Quantization grid (e.g., 16 for 16th notes). If None, uses the value from config.
        """
        self.resolution = resolution or config.get("midi", "resolution", 480)
        self.quantization = quantization or config.get("midi", "quantization", 16)
    
    def load_midi(self, midi_path: Union[str, Path]) -> music21.stream.Score:
        """
        Load a MIDI file using music21.
        
        Args:
            midi_path: Path to the MIDI file.
            
        Returns:
            Music21 Score object.
        """
        midi_path = Path(midi_path)
        
        try:
            score = music21.converter.parse(midi_path)
            return score
        except Exception as e:
            raise IOError(f"Failed to load MIDI file {midi_path}: {e}")
    
    def load_midi_mido(self, midi_path: Union[str, Path]) -> mido.MidiFile:
        """
        Load a MIDI file using mido for low-level access.
        
        Args:
            midi_path: Path to the MIDI file.
            
        Returns:
            Mido MidiFile object.
        """
        midi_path = Path(midi_path)
        
        try:
            midi_file = mido.MidiFile(midi_path)
            return midi_file
        except Exception as e:
            raise IOError(f"Failed to load MIDI file {midi_path} with mido: {e}")
    
    def save_midi(
        self, 
        score: music21.stream.Score, 
        output_path: Union[str, Path]
    ) -> Path:
        """
        Save a music21 Score as a MIDI file.
        
        Args:
            score: Music21 Score object.
            output_path: Path to save the MIDI file.
            
        Returns:
            Path to the saved MIDI file.
        """
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            score.write('midi', fp=output_path)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save MIDI file {output_path}: {e}")
    
    def save_midi_mido(
        self,
        midi_file: mido.MidiFile,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Save a mido MidiFile.
        
        Args:
            midi_file: Mido MidiFile object.
            output_path: Path to save the MIDI file.
            
        Returns:
            Path to the saved MIDI file.
        """
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            midi_file.save(output_path)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save MIDI file {output_path} with mido: {e}")
    
    def extract_notes(
        self, 
        score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract notes from a music21 Score.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            List of note dictionaries.
        """
        notes = []
        
        try:
            for part in score.parts:
                for note in part.recurse().notes:
                    if note.isNote:
                        notes.append({
                            'pitch': note.pitch.midi,
                            'offset': note.offset,
                            'duration': note.duration.quarterLength,
                            'velocity': note.volume.velocity if note.volume and note.volume.velocity else 64,
                            'part': part.id
                        })
                    elif note.isChord:
                        for pitch in note.pitches:
                            notes.append({
                                'pitch': pitch.midi,
                                'offset': note.offset,
                                'duration': note.duration.quarterLength,
                                'velocity': note.volume.velocity if note.volume and note.volume.velocity else 64,
                                'part': part.id
                            })
            
            # Sort notes by offset
            notes.sort(key=lambda x: x['offset'])
            
            return notes
        except Exception as e:
            raise RuntimeError(f"Failed to extract notes: {e}")
    
    def extract_chords(
        self, 
        score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract chords from a music21 Score.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            List of chord dictionaries.
        """
        chords = []
        
        try:
            for part in score.parts:
                for chord in part.recurse().getElementsByClass('Chord'):
                    chords.append({
                        'pitches': [p.midi for p in chord.pitches],
                        'offset': chord.offset,
                        'duration': chord.duration.quarterLength,
                        'velocity': chord.volume.velocity if chord.volume and chord.volume.velocity else 64,
                        'part': part.id
                    })
            
            # Sort chords by offset
            chords.sort(key=lambda x: x['offset'])
            
            return chords
        except Exception as e:
            raise RuntimeError(f"Failed to extract chords: {e}")
    
    def extract_key(
        self, 
        score: music21.stream.Score
    ) -> Optional[str]:
        """
        Extract key from a music21 Score.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            Key as a string, or None if no key is found.
        """
        try:
            key = score.analyze('key')
            return str(key) if key else None
        except Exception as e:
            print(f"Warning: Could not extract key: {e}")
            return None
    
    def extract_tempo(
        self, 
        score: music21.stream.Score
    ) -> Optional[float]:
        """
        Extract tempo from a music21 Score.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            Tempo in BPM, or None if no tempo is found.
        """
        try:
            tempo_elements = score.recurse().getElementsByClass('MetronomeMark')
            if tempo_elements:
                return tempo_elements[0].number
            return None
        except Exception as e:
            print(f"Warning: Could not extract tempo: {e}")
            return None
    
    def extract_time_signature(
        self, 
        score: music21.stream.Score
    ) -> Optional[str]:
        """
        Extract time signature from a music21 Score.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            Time signature as a string, or None if no time signature is found.
        """
        try:
            time_signature_elements = score.recurse().getElementsByClass('TimeSignature')
            if time_signature_elements:
                return str(time_signature_elements[0])
            return None
        except Exception as e:
            print(f"Warning: Could not extract time signature: {e}")
            return None
    
    def transpose(
        self, 
        score: music21.stream.Score, 
        semitones: int
    ) -> music21.stream.Score:
        """
        Transpose a music21 Score.
        
        Args:
            score: Music21 Score object.
            semitones: Number of semitones to transpose.
            
        Returns:
            Transposed music21 Score object.
        """
        try:
            return score.transpose(semitones)
        except Exception as e:
            raise RuntimeError(f"Failed to transpose: {e}")
    
    def quantize(
        self, 
        score: music21.stream.Score, 
        grid: Optional[int] = None
    ) -> music21.stream.Score:
        """
        Quantize a music21 Score.
        
        Args:
            score: Music21 Score object.
            grid: Quantization grid (e.g., 16 for 16th notes). If None, uses the instance quantization.
            
        Returns:
            Quantized music21 Score object.
        """
        grid = grid or self.quantization
        quantized_score = score.deepcopy()
        
        try:
            # Calculate grid size in quarter notes
            grid_size = 1.0 / grid
            
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
        except Exception as e:
            raise RuntimeError(f"Failed to quantize: {e}")
    
    def create_score_from_notes(
        self, 
        notes: List[Dict], 
        time_signature: str = '4/4', 
        tempo: float = 120.0
    ) -> music21.stream.Score:
        """
        Create a music21 Score from a list of note dictionaries.
        
        Args:
            notes: List of note dictionaries.
            time_signature: Time signature as a string.
            tempo: Tempo in BPM.
            
        Returns:
            Music21 Score object.
        """
        try:
            # Create a new score
            score = music21.stream.Score()
            
            # Group notes by part
            notes_by_part = {}
            for note_dict in notes:
                part_id = note_dict.get('part', 'part1')
                if part_id not in notes_by_part:
                    notes_by_part[part_id] = []
                notes_by_part[part_id].append(note_dict)
            
            # Create parts
            for part_id, part_notes in notes_by_part.items():
                part = music21.stream.Part(id=part_id)
                
                # Add time signature
                ts = music21.meter.TimeSignature(time_signature)
                part.append(ts)
                
                # Add tempo
                mm = music21.tempo.MetronomeMark(number=tempo)
                part.append(mm)
                
                # Add notes
                for note_dict in part_notes:
                    n = music21.note.Note(pitch=note_dict['pitch'])
                    n.offset = note_dict['offset']
                    n.duration.quarterLength = note_dict['duration']
                    n.volume.velocity = note_dict['velocity']
                    part.append(n)
                
                # Add part to score
                score.append(part)
            
            return score
        except Exception as e:
            raise RuntimeError(f"Failed to create score from notes: {e}")
    
    def create_score_from_chords(
        self, 
        chords: List[Dict], 
        time_signature: str = '4/4', 
        tempo: float = 120.0
    ) -> music21.stream.Score:
        """
        Create a music21 Score from a list of chord dictionaries.
        
        Args:
            chords: List of chord dictionaries.
            time_signature: Time signature as a string.
            tempo: Tempo in BPM.
            
        Returns:
            Music21 Score object.
        """
        try:
            # Create a new score
            score = music21.stream.Score()
            
            # Group chords by part
            chords_by_part = {}
            for chord_dict in chords:
                part_id = chord_dict.get('part', 'part1')
                if part_id not in chords_by_part:
                    chords_by_part[part_id] = []
                chords_by_part[part_id].append(chord_dict)
            
            # Create parts
            for part_id, part_chords in chords_by_part.items():
                part = music21.stream.Part(id=part_id)
                
                # Add time signature
                ts = music21.meter.TimeSignature(time_signature)
                part.append(ts)
                
                # Add tempo
                mm = music21.tempo.MetronomeMark(number=tempo)
                part.append(mm)
                
                # Add chords
                for chord_dict in part_chords:
                    chord_pitches = [music21.pitch.Pitch(p) for p in chord_dict['pitches']]
                    c = music21.chord.Chord(chord_pitches)
                    c.offset = chord_dict['offset']
                    c.duration.quarterLength = chord_dict['duration']
                    c.volume.velocity = chord_dict['velocity']
                    part.append(c)
                
                # Add part to score
                score.append(part)
            
            return score
        except Exception as e:
            raise RuntimeError(f"Failed to create score from chords: {e}")
    
    def extract_melody(
        self, 
        score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract melody from a music21 Score using a heuristic approach.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            List of melody note dictionaries.
        """
        try:
            # Get all notes
            notes = self.extract_notes(score)
            
            # Group notes by offset
            notes_by_offset = {}
            for note in notes:
                offset = note['offset']
                if offset not in notes_by_offset:
                    notes_by_offset[offset] = []
                notes_by_offset[offset].append(note)
            
            # Identify melody using a simple heuristic:
            # 1. For each offset, take the highest pitched note
            # 2. Ensure continuity (avoid large jumps)
            melody = []
            last_pitch = None
            for offset in sorted(notes_by_offset.keys()):
                offset_notes = notes_by_offset[offset]
                
                # Sort by pitch (highest first)
                offset_notes.sort(key=lambda x: -x['pitch'])
                
                # Select best note based on pitch proximity to last note
                selected_note = offset_notes[0]  # Default to highest note
                
                if last_pitch is not None:
                    # Try to find a note close to the last pitch
                    min_distance = float('inf')
                    for note in offset_notes:
                        distance = abs(note['pitch'] - last_pitch)
                        if distance < min_distance:
                            min_distance = distance
                            selected_note = note
                        
                        # Prefer notes within an octave of the last note
                        if distance <= 12:
                            break
                
                melody.append(selected_note)
                last_pitch = selected_note['pitch']
            
            return melody
        except Exception as e:
            raise RuntimeError(f"Failed to extract melody: {e}")
    
    def extract_bass(
        self, 
        score: music21.stream.Score
    ) -> List[Dict]:
        """
        Extract bass line from a music21 Score.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            List of bass note dictionaries.
        """
        try:
            # Get all notes
            notes = self.extract_notes(score)
            
            # Group notes by offset
            notes_by_offset = {}
            for note in notes:
                offset = note['offset']
                if offset not in notes_by_offset:
                    notes_by_offset[offset] = []
                notes_by_offset[offset].append(note)
            
            # Extract lowest note at each offset as bass
            bass = []
            for offset in sorted(notes_by_offset.keys()):
                offset_notes = notes_by_offset[offset]
                
                # Find the lowest note
                lowest_note = min(offset_notes, key=lambda x: x['pitch'])
                bass.append(lowest_note)
            
            return bass
        except Exception as e:
            raise RuntimeError(f"Failed to extract bass: {e}")
    
    def midi_to_pianoroll(
        self, 
        score: music21.stream.Score, 
        resolution: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert a music21 Score to a piano roll representation.
        
        Args:
            score: Music21 Score object.
            resolution: Resolution of the piano roll (in divisions per quarter note).
                        If None, uses the instance resolution.
            
        Returns:
            Piano roll as a 2D numpy array (time x pitch).
        """
        resolution = resolution or self.resolution
        
        try:
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
                
                # Ensure valid indices
                if start_step >= total_steps:
                    continue
                
                end_step = min(end_step, total_steps - 1)
                
                # Set velocity in piano roll
                velocity = note['velocity'] / 127.0 if note['velocity'] is not None else 0.8
                piano_roll[start_step:end_step + 1, note['pitch']] = velocity
            
            return piano_roll
        except Exception as e:
            raise RuntimeError(f"Failed to convert MIDI to piano roll: {e}")
    
    def pianoroll_to_midi(
        self, 
        piano_roll: np.ndarray, 
        resolution: Optional[int] = None, 
        threshold: float = 0.5
    ) -> music21.stream.Score:
        """
        Convert a piano roll representation to a music21 Score.
        
        Args:
            piano_roll: Piano roll as a 2D numpy array (time x pitch).
            resolution: Resolution of the piano roll (in divisions per quarter note).
                        If None, uses the instance resolution.
            threshold: Threshold for note detection.
            
        Returns:
            Music21 Score object.
        """
        resolution = resolution or self.resolution
        
        try:
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
                
                if not np.any(active):
                    continue
                
                changes = np.diff(active.astype(int))
                note_starts = np.where(changes > 0)[0]
                note_ends = np.where(changes < 0)[0]
                
                # Handle edge cases
                if active[0]:
                    # First note is already active
                    note_starts = np.insert(note_starts, 0, 0)
                if active[-1]:
                    # Last note is still active at the end
                    note_ends = np.append(note_ends, len(active) - 1)
                
                # Create notes
                for start, end in zip(note_starts, note_ends):
                    # Calculate offset and duration
                    offset = start / resolution
                    duration = (end - start) / resolution
                    
                    # Skip very short notes
                    if duration < 0.1:
                        continue
                    
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
        except Exception as e:
            raise RuntimeError(f"Failed to convert piano roll to MIDI: {e}")
    
    def analyze_midi(
        self, 
        score: music21.stream.Score
    ) -> Dict[str, any]:
        """
        Analyze a music21 Score to extract musical features.
        
        Args:
            score: Music21 Score object.
            
        Returns:
            Dictionary of analysis results.
        """
        try:
            results = {}
            
            # Basic information
            time_sig = self.extract_time_signature(score)
            key = self.extract_key(score)
            tempo = self.extract_tempo(score)
            
            results['time_signature'] = time_sig
            results['key'] = key
            results['tempo'] = tempo
            
            # Extract notes and calculate statistics
            notes = self.extract_notes(score)
            
            if notes:
                # Pitch range
                pitches = [note['pitch'] for note in notes]
                results['pitch_range'] = {
                    'min': min(pitches),
                    'max': max(pitches),
                    'mean': np.mean(pitches),
                    'median': np.median(pitches)
                }
                
                # Note durations
                durations = [note['duration'] for note in notes]
                results['duration_stats'] = {
                    'min': min(durations),
                    'max': max(durations),
                    'mean': np.mean(durations),
                    'median': np.median(durations)
                }
                
                # Velocities
                velocities = [note['velocity'] for note in notes if note['velocity'] is not None]
                if velocities:
                    results['velocity_stats'] = {
                        'min': min(velocities),
                        'max': max(velocities),
                        'mean': np.mean(velocities),
                        'median': np.median(velocities)
                    }
                
                # Calculate total duration
                max_offset = max(note['offset'] + note['duration'] for note in notes)
                results['total_duration'] = max_offset
                
                # Note density (notes per quarter note)
                results['note_density'] = len(notes) / max_offset if max_offset > 0 else 0
            
            # Chords analysis
            chords = self.extract_chords(score)
            if chords:
                results['chord_count'] = len(chords)
                
                # Average notes per chord
                avg_notes = np.mean([len(chord['pitches']) for chord in chords])
                results['avg_notes_per_chord'] = avg_notes
            
            # Parts analysis
            part_ids = set()
            for note in notes:
                if 'part' in note:
                    part_ids.add(note['part'])
            
            results['part_count'] = len(part_ids)
            
            # Advanced analysis using music21
            try:
                # Chord symbol analysis
                chord_symbols = score.recurse().getElementsByClass('ChordSymbol')
                if chord_symbols:
                    chord_types = {}
                    for cs in chord_symbols:
                        chord_type = cs.commonName
                        chord_types[chord_type] = chord_types.get(chord_type, 0) + 1
                    
                    results['chord_types'] = chord_types
                
                # Melodic intervals
                melody = self.extract_melody(score)
                if len(melody) > 1:
                    intervals = []
                    for i in range(1, len(melody)):
                        interval = melody[i]['pitch'] - melody[i-1]['pitch']
                        intervals.append(interval)
                    
                    results['melodic_intervals'] = {
                        'min': min(intervals),
                        'max': max(intervals),
                        'mean': np.mean(intervals),
                        'abs_mean': np.mean(np.abs(intervals))
                    }
            except Exception as e:
                print(f"Warning: Advanced analysis error: {e}")
            
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to analyze MIDI: {e}")
    
    def midi_to_tokens(
        self, 
        midi_file: mido.MidiFile,
        encode_time: bool = True,
        encode_velocity: bool = True
    ) -> List[int]:
        """
        Convert a MIDI file to a sequence of tokens for language models.
        
        Args:
            midi_file: Mido MidiFile object.
            encode_time: Whether to encode time (delta) information.
            encode_velocity: Whether to encode velocity information.
            
        Returns:
            List of integer tokens.
        """
        try:
            tokens = []
            
            # Define token mappings
            NOTE_ON_START = 0    # Token range: 0-127 for MIDI pitches
            NOTE_OFF_START = 128  # Token range: 128-255 for note offs
            VELOCITY_START = 256  # Token range: 256-383 for velocities
            TIME_START = 384      # Token range: 384+ for time deltas
            
            for track in midi_file.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Note on event
                        note_token = NOTE_ON_START + msg.note
                        tokens.append(note_token)
                        
                        # Add velocity token if requested
                        if encode_velocity:
                            velocity_token = VELOCITY_START + msg.velocity // 2  # Quantize to 64 levels
                            tokens.append(velocity_token)
                    
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        # Note off event
                        note_token = NOTE_OFF_START + msg.note
                        tokens.append(note_token)
                    
                    # Add time token if requested
                    if encode_time and msg.time > 0:
                        # Quantize time to a reasonable range
                        time_value = min(127, int(msg.time / midi_file.ticks_per_beat * 100))
                        if time_value > 0:
                            time_token = TIME_START + time_value
                            tokens.append(time_token)
            
            return tokens
        except Exception as e:
            raise RuntimeError(f"Failed to convert MIDI to tokens: {e}")
    
    def tokens_to_midi(
        self, 
        tokens: List[int],
        ticks_per_beat: int = 480
    ) -> mido.MidiFile:
        """
        Convert a sequence of tokens back to a MIDI file.
        
        Args:
            tokens: List of integer tokens.
            ticks_per_beat: Ticks per beat (resolution) for the MIDI file.
            
        Returns:
            Mido MidiFile object.
        """
        try:
            # Define token mappings (same as in midi_to_tokens)
            NOTE_ON_START = 0
            NOTE_OFF_START = 128
            VELOCITY_START = 256
            TIME_START = 384
            
            # Create a new MIDI file with one track
            midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
            track = mido.MidiTrack()
            midi_file.tracks.append(track)
            
            # Add a time signature and tempo
            track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
            
            current_time = 0
            current_velocity = 64  # Default velocity
            
            # Process tokens
            i = 0
            while i < len(tokens):
                token = tokens[i]
                i += 1
                
                if NOTE_ON_START <= token < NOTE_OFF_START:
                    # Note on event
                    note = token - NOTE_ON_START
                    
                    # Check if next token is a velocity token
                    if i < len(tokens) and VELOCITY_START <= tokens[i] < TIME_START:
                        current_velocity = (tokens[i] - VELOCITY_START) * 2
                        i += 1
                    
                    # Add note on message
                    track.append(mido.Message('note_on', note=note, velocity=current_velocity, time=current_time))
                    current_time = 0
                
                elif NOTE_OFF_START <= token < VELOCITY_START:
                    # Note off event
                    note = token - NOTE_OFF_START
                    track.append(mido.Message('note_off', note=note, velocity=0, time=current_time))
                    current_time = 0
                
                elif TIME_START <= token:
                    # Time delta event
                    delta_time = token - TIME_START
                    current_time += int(delta_time * ticks_per_beat / 100)
            
            # End of track marker
            track.append(mido.MetaMessage('end_of_track', time=current_time))
            
            return midi_file
        except Exception as e:
            raise RuntimeError(f"Failed to convert tokens to MIDI: {e}")