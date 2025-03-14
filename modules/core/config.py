"""
Configuration module for Disco Musica.

This module provides utilities for managing application configuration,
including model parameters, processing settings, and user preferences.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigManager:
    """
    Manages configuration settings for the application.
    
    This class handles loading, saving, and accessing configuration settings
    from various sources (default values, config files, environment variables).
    """
    
    DEFAULT_CONFIG = {
        "audio": {
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 2,
            "format": "wav",
            "normalization_level": -14.0,  # LUFS
            "segment_length": 30.0,  # seconds
            "overlap": 5.0,  # seconds
        },
        "midi": {
            "resolution": 480,  # ticks per quarter note
            "quantization": 16,  # 16th note grid
        },
        "models": {
            "default_text_to_music": "facebook/musicgen-small",
            "default_audio_to_audio": "facebook/musicgen-small",
            "default_midi_to_audio": "github/midi-ddsp",
            "model_cache_dir": "models/pretrained",
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "epochs": 50,
            "checkpoint_interval": 1000,  # steps
            "validation_interval": 500,  # steps
            "early_stopping_patience": 5,  # epochs
            "mixed_precision": True,
            "lora_rank": 16,
            "lora_alpha": 32,
        },
        "inference": {
            "max_generation_length": 30.0,  # seconds
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.0,
            "classifier_free_guidance": 3.0,
        },
        "paths": {
            "data_dir": "data",
            "output_dir": "outputs",
            "logs_dir": "logs",
            "temp_dir": "temp",
        },
        "ui": {
            "theme": "dark",
            "default_view": "text_to_music",
            "audio_visualization": "waveform",
            "midi_visualization": "piano_roll",
        },
        "resources": {
            "max_memory_usage": 0.8,  # fraction of available RAM
            "max_gpu_memory_usage": 0.9,  # fraction of available VRAM
            "cpu_threads": 4,
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the configuration file. If None, the default 
                         configuration will be used.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_path = Path(config_path) if config_path else None
        
        # Load configuration from file if provided
        if self.config_path and self.config_path.exists():
            self.load_from_file(self.config_path)
        
        # Load configuration from environment variables
        self.load_from_env()
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
        """
        config_path = Path(config_path)
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                self._update_nested_dict(self.config, file_config)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
    
    def load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be in the format DISCO_MUSICA_SECTION_KEY=VALUE.
        For example, DISCO_MUSICA_AUDIO_SAMPLE_RATE=48000.
        """
        prefix = "DISCO_MUSICA_"
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Remove prefix and split by underscore
                parts = env_var[len(prefix):].lower().split('_')
                
                if len(parts) >= 2:
                    section, key = parts[0], '_'.join(parts[1:])
                    
                    # Try to convert value to appropriate type
                    try:
                        if value.lower() in ('true', 'yes', '1'):
                            typed_value = True
                        elif value.lower() in ('false', 'no', '0'):
                            typed_value = False
                        elif '.' in value and value.replace('.', '', 1).isdigit():
                            typed_value = float(value)
                        elif value.isdigit():
                            typed_value = int(value)
                        else:
                            typed_value = value
                        
                        # Update config if section and key exist
                        if section in self.config and key in self.config[section]:
                            self.config[section][key] = typed_value
                    except Exception as e:
                        print(f"Error setting {env_var}: {e}")
    
    def save_to_file(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration file. If None, the path
                         provided during initialization will be used.
        """
        save_path = Path(config_path) if config_path else self.config_path
        if not save_path:
            raise ValueError("No configuration file path specified.")
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving configuration to {save_path}: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Section of the configuration.
            key: Key within the section.
            default: Default value to return if the section or key doesn't exist.
        
        Returns:
            The configuration value or the default if not found.
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Section of the configuration.
            key: Key within the section.
            value: Value to set.
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section of the configuration.
        
        Returns:
            Dictionary containing the section configuration or an empty dict if
            the section doesn't exist.
        """
        return self.config.get(section, {}).copy()
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Update a nested dictionary recursively.
        
        Args:
            base_dict: Base dictionary to update.
            update_dict: Dictionary with updates.
        """
        for key, value in update_dict.items():
            if (
                key in base_dict 
                and isinstance(base_dict[key], dict) 
                and isinstance(value, dict)
            ):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value


# Create a global config instance
config = ConfigManager()


def initialize_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Initialize the global configuration.
    
    Args:
        config_path: Path to the configuration file.
    
    Returns:
        Initialized ConfigManager instance.
    """
    global config
    config = ConfigManager(config_path)
    return config