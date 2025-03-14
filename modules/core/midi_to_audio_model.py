"""
MIDI-to-Audio model module for Disco Musica.

This module provides a wrapper for MIDI-to-audio generation models,
supporting models that can synthesize audio from MIDI input.
"""

import os
import time
import datetime
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from modules.core.base_model import BaseModel, PretrainedModelMixin, TorchModelMixin
from modules.core.config import config
from modules.core.midi_processor import MIDIProcessor


class MIDIToAudioModel(BaseModel, PretrainedModelMixin, TorchModelMixin):
    """
    Model for generating audio from MIDI input, optionally with text descriptions.
    
    This class wraps MIDI-to-audio models, providing a consistent interface for loading,
    saving, and generating audio from MIDI data.
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/musicgen-melody", 
        model_type: str = "pretrained"
    ):
        """
        Initialize the MIDIToAudioModel.
        
        Args:
            model_name: Name of the model. Default is "facebook/musicgen-melody".
            model_type: Type of model ('pretrained' or 'finetuned').
        """
        super().__init__(model_name, model_type)
        
        self.metadata.update({
            "capabilities": ["midi_to_audio"],
            "version": "0.1.0",
            "creation_date": datetime.datetime.now().isoformat(),
            "parameters": {
                "max_duration": config.get("inference", "max_generation_length", 30.0),
                "sample_rate": config.get("audio", "sample_rate", 44100),
                "temperature": config.get("inference", "temperature", 1.0),
                "top_k": config.get("inference", "top_k", 250),
                "top_p": config.get("inference", "top_p", 0.0),
                "cfg_coef": config.get("inference", "classifier_free_guidance", 3.0)
            }
        })
        
        # Flag to track model loading status
        self._is_loaded = False
        
        # Specific model and processor
        self.model = None
        self.processor = None
        self.midi_processor = MIDIProcessor()
    
    def load(self) -> None:
        """
        Load the model from disk or download if necessary.
        """
        if self._is_loaded:
            return
            
        model_path = self.get_model_path()
        
        try:
            # Try to import transformers
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            
            # Check if model exists locally
            if model_path.exists() and (model_path / "config.json").exists():
                print(f"Loading model from {model_path}")
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                # Download from Hugging Face Hub
                print(f"Downloading model {self.model_name} from Hugging Face Hub")
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                
                # Save model locally
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)
                self.processor.save_pretrained(model_path)
            
            # Update metadata
            self.metadata.update({
                "last_modified": datetime.datetime.now().isoformat(),
                "parameters": {
                    **self.metadata["parameters"],
                    "model_size": self.get_model_size()
                }
            })
            self.save_metadata()
            
            self._is_loaded = True
            print(f"Model {self.model_name} loaded successfully")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the model to disk.
        
        Args:
            output_path: Path to save the model. If None, a default path based on 
                         model name and type will be used.
        
        Returns:
            Path where the model was saved.
        """
        if not self._is_loaded or self.model is None:
            raise ValueError("Model must be loaded before saving")
        
        if output_path is None:
            output_path = self.get_model_path()
        else:
            output_path = Path(output_path)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Save model and processor
            self.model.save_pretrained(output_path)
            if self.processor is not None:
                self.processor.save_pretrained(output_path)
            
            # Update and save metadata
            self.metadata.update({
                "last_modified": datetime.datetime.now().isoformat()
            })
            self.save_metadata(output_path / "metadata.json")
            
            print(f"Model saved to {output_path}")
            return output_path
        
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    def generate(
        self, 
        midi_path: Union[str, Path],
        prompt: Optional[str] = None,
        instrument_description: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg_coef: Optional[float] = None,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate audio from a MIDI file, optionally with text description.
        
        Args:
            midi_path: Path to the MIDI file to synthesize.
            prompt: Optional text prompt describing the desired sound.
            instrument_description: Optional description of instruments to use.
            temperature: Sampling temperature (higher = more random).
            top_k: Number of top tokens to consider (0 = disabled).
            top_p: Nucleus sampling probability threshold (0 = disabled).
            cfg_coef: Classifier-free guidance coefficient (higher = more adherence to prompt).
            return_tensor: Whether to return a torch.Tensor (True) or numpy array (False).
            
        Returns:
            Generated audio as a numpy array or torch.Tensor.
        """
        if not self._is_loaded:
            self.load()
        
        # Use default values from metadata if not provided
        temperature = temperature or self.metadata["parameters"].get("temperature", 1.0)
        top_k = top_k or self.metadata["parameters"].get("top_k", 250)
        top_p = top_p or self.metadata["parameters"].get("top_p", 0.0)
        cfg_coef = cfg_coef or self.metadata["parameters"].get("cfg_coef", 3.0)
        
        try:
            # Load and process MIDI
            midi_score = self.midi_processor.load_midi(midi_path)
            
            # Extract MIDI information for conditioning
            key = self.midi_processor.extract_key(midi_score) or "C major"
            tempo = self.midi_processor.extract_tempo(midi_score) or 120
            time_sig = self.midi_processor.extract_time_signature(midi_score) or "4/4"
            
            # Extract melody for possible audio conditioning
            melody_notes = self.midi_processor.extract_melody(midi_score)
            
            # Create piano roll representation for visualization (not used in generation yet)
            piano_roll = self.midi_processor.midi_to_pianoroll(midi_score)
            
            # Analyze MIDI to create a detailed text description
            midi_info = self.midi_processor.analyze_midi(midi_score)
            
            # Create a comprehensive prompt if none is provided
            if not prompt:
                # Detect if it's a melody, chord progression, or full arrangement
                if len(midi_info.get("part_count", 1)) == 1:
                    midi_type = "melody"
                elif piano_roll.shape[1] < 20:  # Simple heuristic for chord progressions
                    midi_type = "chord progression"
                else:
                    midi_type = "musical arrangement"
                
                # Create automatic prompt from MIDI analysis
                auto_prompt = f"Generate music based on this {midi_type} in {key} at {tempo} BPM in {time_sig} time."
                
                if instrument_description:
                    auto_prompt += f" Use {instrument_description}."
                    
                prompt = auto_prompt
            
            # For now, we handle MIDI as a text-to-music generation with a detailed prompt
            # In the future, we could implement direct MIDI conditioning if supported by the model
            
            # Prepare inputs
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Calculate appropriate generation length based on MIDI duration
            duration = float(midi_info.get("total_duration", 30.0))
            sample_rate = self.model.config.audio_encoder.sampling_rate
            chunk_length = self.model.config.audio_encoder.chunk_length
            max_new_tokens = int(duration * sample_rate / chunk_length)
            
            # Generate audio
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
            }
            
            if top_k > 0:
                generation_kwargs["top_k"] = top_k
                
            if top_p > 0:
                generation_kwargs["top_p"] = top_p
                
            if cfg_coef > 0:
                generation_kwargs["guidance_scale"] = cfg_coef
            
            # Start generation
            start_time = time.time()
            
            with torch.inference_mode():
                generated_audio = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            generation_time = time.time() - start_time
            print(f"Generation took {generation_time:.2f} seconds")
            
            # Convert to audio waveforms
            audio_values = self.processor.batch_decode(generated_audio, return_tensors="pt")
            
            # Return the generated audio
            if return_tensor:
                return audio_values[0]  # Return first batch item as tensor
            else:
                return audio_values[0].cpu().numpy()  # Return first batch item as numpy array
                
        except Exception as e:
            raise RuntimeError(f"Music generation from MIDI failed: {e}")
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if the model is loaded, False otherwise.
        """
        return self._is_loaded and self.model is not None
    
    def generate_with_instrument(
        self,
        midi_path: Union[str, Path],
        instrument: str,
        **kwargs
    ) -> np.ndarray:
        """
        Generate audio from a MIDI file with a specific instrument sound.
        
        Args:
            midi_path: Path to the MIDI file to synthesize.
            instrument: Instrument to use for synthesis (e.g., "piano", "guitar").
            **kwargs: Additional arguments for the generate method.
            
        Returns:
            Generated audio as a numpy array.
        """
        return self.generate(
            midi_path=midi_path,
            instrument_description=instrument,
            **kwargs
        )
    
    def generate_arranged_version(
        self,
        midi_path: Union[str, Path],
        arrangement_description: str,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a fully arranged version of a MIDI file.
        
        Args:
            midi_path: Path to the MIDI file to arrange.
            arrangement_description: Description of the desired arrangement.
            **kwargs: Additional arguments for the generate method.
            
        Returns:
            Generated audio as a numpy array.
        """
        prompt = f"Create a full arrangement of this music with {arrangement_description}"
        
        return self.generate(
            midi_path=midi_path,
            prompt=prompt,
            **kwargs
        )