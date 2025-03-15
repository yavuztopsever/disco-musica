"""
Visualization module for Disco Musica.

This module provides functions for creating visualizations of audio and MIDI data.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import pretty_midi


def create_audio_visualization(
    audio_data: np.ndarray,
    sr: int,
    plot_type: str = "all",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create visualizations for audio data.
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        plot_type: Type of plot ("waveform", "spectrogram", "mel", "all")
        figsize: Figure size
        
    Returns:
        Matplotlib figure with visualizations
    """
    if plot_type == "all":
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Waveform
        librosa.display.waveshow(
            audio_data,
            sr=sr,
            ax=axes[0]
        )
        axes[0].set_title("Waveform")
        
        # Spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)),
            ref=np.max
        )
        librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='log',
            ax=axes[1]
        )
        axes[1].set_title("Spectrogram")
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=axes[2]
        )
        axes[2].set_title("Mel Spectrogram")
        
        plt.tight_layout()
        return fig
    
    elif plot_type == "waveform":
        fig, ax = plt.subplots(figsize=figsize)
        librosa.display.waveshow(audio_data, sr=sr, ax=ax)
        ax.set_title("Waveform")
        return fig
    
    elif plot_type == "spectrogram":
        fig, ax = plt.subplots(figsize=figsize)
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)),
            ref=np.max
        )
        librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='log',
            ax=ax
        )
        ax.set_title("Spectrogram")
        return fig
    
    elif plot_type == "mel":
        fig, ax = plt.subplots(figsize=figsize)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
        ax.set_title("Mel Spectrogram")
        return fig
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def create_midi_visualization(
    midi_data: Union[str, pretty_midi.PrettyMIDI],
    plot_type: str = "piano_roll",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create visualizations for MIDI data.
    
    Args:
        midi_data: MIDI data as PrettyMIDI object or path to MIDI file
        plot_type: Type of plot ("piano_roll", "note_density", "all")
        figsize: Figure size
        
    Returns:
        Matplotlib figure with visualizations
    """
    # Load MIDI if path is provided
    if isinstance(midi_data, str):
        midi_data = pretty_midi.PrettyMIDI(midi_data)
    
    if plot_type == "all":
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Piano roll
        piano_roll = midi_data.get_piano_roll()
        librosa.display.specshow(
            piano_roll,
            x_axis='time',
            y_axis='midi',
            ax=axes[0]
        )
        axes[0].set_title("Piano Roll")
        
        # Note density
        times = np.arange(0, midi_data.get_end_time(), 0.1)
        density = np.zeros_like(times)
        for note in midi_data.instruments[0].notes:
            idx_start = int(note.start / 0.1)
            idx_end = int(note.end / 0.1)
            if idx_end >= len(density):
                idx_end = len(density) - 1
            density[idx_start:idx_end] += 1
        
        axes[1].plot(times, density)
        axes[1].set_title("Note Density")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Number of active notes")
        
        plt.tight_layout()
        return fig
    
    elif plot_type == "piano_roll":
        fig, ax = plt.subplots(figsize=figsize)
        piano_roll = midi_data.get_piano_roll()
        librosa.display.specshow(
            piano_roll,
            x_axis='time',
            y_axis='midi',
            ax=ax
        )
        ax.set_title("Piano Roll")
        return fig
    
    elif plot_type == "note_density":
        fig, ax = plt.subplots(figsize=figsize)
        times = np.arange(0, midi_data.get_end_time(), 0.1)
        density = np.zeros_like(times)
        for note in midi_data.instruments[0].notes:
            idx_start = int(note.start / 0.1)
            idx_end = int(note.end / 0.1)
            if idx_end >= len(density):
                idx_end = len(density) - 1
            density[idx_start:idx_end] += 1
        
        ax.plot(times, density)
        ax.set_title("Note Density")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of active notes")
        return fig
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def create_advanced_visualization(
    audio_data: Optional[np.ndarray] = None,
    midi_data: Optional[Union[str, pretty_midi.PrettyMIDI]] = None,
    sr: Optional[int] = None,
    plot_type: str = "combined",
    figsize: Tuple[int, int] = (12, 12)
) -> plt.Figure:
    """
    Create advanced visualizations combining audio and MIDI data.
    
    Args:
        audio_data: Audio data as numpy array
        midi_data: MIDI data as PrettyMIDI object or path
        sr: Sample rate for audio data
        plot_type: Type of plot ("combined", "aligned", "comparison")
        figsize: Figure size
        
    Returns:
        Matplotlib figure with visualizations
    """
    if audio_data is None and midi_data is None:
        raise ValueError("Either audio_data or midi_data must be provided")
    
    if plot_type == "combined" and audio_data is not None and midi_data is not None:
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Audio waveform
        librosa.display.waveshow(
            audio_data,
            sr=sr,
            ax=axes[0]
        )
        axes[0].set_title("Audio Waveform")
        
        # Spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_data)),
            ref=np.max
        )
        librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='log',
            ax=axes[1]
        )
        axes[1].set_title("Audio Spectrogram")
        
        # MIDI piano roll
        if isinstance(midi_data, str):
            midi_data = pretty_midi.PrettyMIDI(midi_data)
        piano_roll = midi_data.get_piano_roll()
        librosa.display.specshow(
            piano_roll,
            x_axis='time',
            y_axis='midi',
            ax=axes[2]
        )
        axes[2].set_title("MIDI Piano Roll")
        
        plt.tight_layout()
        return fig
    
    elif plot_type == "aligned" and audio_data is not None and midi_data is not None:
        # Create aligned visualization of audio and MIDI
        # This would require additional alignment logic
        raise NotImplementedError("Aligned visualization not yet implemented")
    
    elif plot_type == "comparison":
        if audio_data is not None:
            return create_audio_visualization(audio_data, sr, "all", figsize)
        else:
            return create_midi_visualization(midi_data, "all", figsize)
    
    else:
        raise ValueError(f"Invalid plot type or data combination: {plot_type}") 