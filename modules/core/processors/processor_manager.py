"""Processor manager for the system."""
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np

from .audio_processor import AudioProcessor
from .midi_processor import MIDIProcessor
from .text_processor import TextProcessor
from ..exceptions.base_exceptions import ProcessingError, ValidationError


class ProcessorManager:
    """Manager for system processors."""
    
    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        midi_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the processor manager.
        
        Args:
            audio_config: Configuration for audio processor.
            midi_config: Configuration for MIDI processor.
            text_config: Configuration for text processor.
        """
        # Initialize processors with configs
        self.audio = AudioProcessor(**(audio_config or {}))
        self.midi = MIDIProcessor(**(midi_config or {}))
        self.text = TextProcessor(**(text_config or {}))
        
        # File type mappings
        self._file_types = {
            "audio": [".wav", ".mp3", ".ogg", ".flac"],
            "midi": [".mid", ".midi"],
            "text": [".txt", ".json"]
        }
        
    def get_file_type(self, file_path: Path) -> str:
        """Get type of file.
        
        Args:
            file_path: Path to file.
            
        Returns:
            File type ("audio", "midi", "text").
            
        Raises:
            ValidationError: If file type is unsupported.
        """
        suffix = file_path.suffix.lower()
        for file_type, extensions in self._file_types.items():
            if suffix in extensions:
                return file_type
        raise ValidationError(f"Unsupported file type: {suffix}")
        
    def process_file(
        self,
        file_path: Path,
        operations: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Process file with specified operations.
        
        Args:
            file_path: Path to file.
            operations: List of operations to perform.
            **kwargs: Operation-specific arguments.
            
        Returns:
            Dictionary of operation results.
            
        Raises:
            ValidationError: If file type or operations are invalid.
            ProcessingError: If processing fails.
        """
        # Get file type
        file_type = self.get_file_type(file_path)
        
        # Get appropriate processor
        processor = getattr(self, file_type)
        
        # Process operations
        results = {}
        for operation in operations:
            if not hasattr(processor, operation):
                raise ValidationError(
                    f"Invalid operation for {file_type}: {operation}"
                )
                
            try:
                # Get operation method
                method = getattr(processor, operation)
                
                # Get operation arguments
                op_args = {
                    k.replace(f"{operation}_", ""): v
                    for k, v in kwargs.items()
                    if k.startswith(f"{operation}_")
                }
                
                # Execute operation
                if operation in ["load_audio", "load_midi"]:
                    results[operation] = method(file_path, **op_args)
                else:
                    results[operation] = method(**op_args)
                    
            except Exception as e:
                raise ProcessingError(
                    f"Error in operation {operation}: {e}"
                )
                
        return results
        
    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> None:
        """Convert file between formats.
        
        Args:
            input_path: Input file path.
            output_path: Output file path.
            **kwargs: Format-specific arguments.
            
        Raises:
            ValidationError: If conversion is unsupported.
            ProcessingError: If conversion fails.
        """
        # Get file types
        input_type = self.get_file_type(input_path)
        output_type = self.get_file_type(output_path)
        
        try:
            if input_type == output_type:
                # Same type conversion (e.g., wav to mp3)
                if input_type == "audio":
                    self.audio.convert_format(
                        input_path,
                        output_path,
                        **kwargs
                    )
                else:
                    raise ValidationError(
                        f"Format conversion not supported for {input_type}"
                    )
            else:
                raise ValidationError(
                    f"Conversion from {input_type} to {output_type} not supported"
                )
                
        except Exception as e:
            raise ProcessingError(f"Error converting file: {e}")
            
    def analyze_file(
        self,
        file_path: Path,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Analyze file content.
        
        Args:
            file_path: Path to file.
            analysis_type: Type of analysis to perform.
            
        Returns:
            Dictionary of analysis results.
            
        Raises:
            ValidationError: If analysis type is invalid.
            ProcessingError: If analysis fails.
        """
        # Get file type
        file_type = self.get_file_type(file_path)
        
        try:
            if file_type == "audio":
                if analysis_type == "duration":
                    return {"duration": self.audio.get_duration(file_path)}
                elif analysis_type == "tempo":
                    return {"tempo": self.audio.get_tempo(file_path)}
                elif analysis_type == "key":
                    return {"key": self.audio.get_key(file_path)}
                elif analysis_type == "waveform":
                    return {"waveform": self.audio.get_waveform(file_path)}
                elif analysis_type == "spectrogram":
                    return {"spectrogram": self.audio.get_spectrogram(file_path)}
                    
            elif file_type == "midi":
                midi = self.midi.load_midi(file_path)
                if analysis_type == "duration":
                    return {"duration": self.midi.get_duration(midi)}
                elif analysis_type == "tempo":
                    return {"tempo": self.midi.get_tempo(midi)}
                elif analysis_type == "key":
                    return {"key": self.midi.get_key(midi)}
                elif analysis_type == "time_signature":
                    return {
                        "time_signature": self.midi.get_time_signature(midi)
                    }
                elif analysis_type == "piano_roll":
                    return {"piano_roll": self.midi.get_piano_roll(midi)}
                    
            elif file_type == "text":
                with open(file_path, "r") as f:
                    text = f.read()
                if analysis_type == "musical_terms":
                    return {
                        "terms": self.text.extract_musical_terms(text)
                    }
                elif analysis_type == "sentiment":
                    return {"sentiment": self.text.analyze_sentiment(text)}
                elif analysis_type == "key_phrases":
                    return {"phrases": self.text.get_key_phrases(text)}
                    
            raise ValidationError(
                f"Invalid analysis type for {file_type}: {analysis_type}"
            )
            
        except Exception as e:
            raise ProcessingError(f"Error analyzing file: {e}")
            
    def process_generation_prompt(
        self,
        prompt: str,
        operations: List[str] = ["normalize", "augment"]
    ) -> str:
        """Process generation prompt.
        
        Args:
            prompt: Input prompt.
            operations: List of operations to perform.
            
        Returns:
            Processed prompt.
            
        Raises:
            ValidationError: If operations are invalid.
            ProcessingError: If processing fails.
        """
        try:
            processed = prompt
            for operation in operations:
                if operation == "normalize":
                    processed = self.text.normalize_prompt(processed)
                elif operation == "augment":
                    processed = self.text.augment_prompt(processed)
                elif operation == "split":
                    processed = " ".join(self.text.split_prompt(processed))
                else:
                    raise ValidationError(f"Invalid prompt operation: {operation}")
            return processed
            
        except Exception as e:
            raise ProcessingError(f"Error processing prompt: {e}")
            
    def combine_generation_prompts(
        self,
        prompts: List[str],
        weights: Optional[List[float]] = None
    ) -> str:
        """Combine multiple generation prompts.
        
        Args:
            prompts: List of prompts.
            weights: Optional weights for each prompt.
            
        Returns:
            Combined prompt.
            
        Raises:
            ValidationError: If weights are invalid.
            ProcessingError: If combination fails.
        """
        try:
            # Process each prompt first
            processed_prompts = [
                self.process_generation_prompt(prompt)
                for prompt in prompts
            ]
            
            # Combine processed prompts
            return self.text.combine_prompts(
                processed_prompts,
                weights=weights
            )
            
        except Exception as e:
            raise ProcessingError(f"Error combining prompts: {e}")
            
    def get_processor_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about processors.
        
        Returns:
            Dictionary of processor configurations.
        """
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "format": self.audio.format,
                "supported_formats": self.audio._formats
            },
            "midi": {
                "resolution": self.midi.resolution,
                "tempo": self.midi.tempo,
                "time_signature": self.midi.time_signature,
                "program_map": self.midi._program_map
            },
            "text": {
                "model_name": self.text.model_name,
                "max_length": self.text.max_length,
                "device": self.text.device,
                "musical_terms": self.text._musical_terms
            }
        } 