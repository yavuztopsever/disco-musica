"""
Training interface for Disco Musica.

This module provides a UI interface for model training and fine-tuning.
"""

import os
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import gradio as gr
import torch

from modules.core.config import config
from modules.services.model_service import get_model_service
from modules.core.model_trainer import ModelTrainer


class TrainingInterface:
    """
    Training interface for Disco Musica.
    
    This class provides a UI for training and fine-tuning models.
    """
    
    def __init__(self):
        """
        Initialize the TrainingInterface.
        """
        self.model_service = get_model_service()
        self.trainer = ModelTrainer()
        
        # Data directories
        self.data_dir = Path(config.get("paths", "data_dir", "data"))
        self.datasets_dir = self.data_dir / "datasets"
        
        # Output directory
        self.output_dir = Path(config.get("paths", "output_dir", "outputs")) / "training"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_dataset_interface(self) -> List[gr.Component]:
        """
        Create the dataset creation interface components.
        
        Returns:
            List of Gradio components.
        """
        # Dataset name and description
        dataset_name = gr.Textbox(
            label="Dataset Name",
            placeholder="Enter a name for your dataset"
        )
        
        dataset_description = gr.Textbox(
            label="Dataset Description",
            placeholder="Enter a description of your dataset",
            lines=3
        )
        
        # Dataset type
        dataset_type = gr.Radio(
            label="Dataset Type",
            choices=["Text-to-Music", "Audio-to-Music", "MIDI-to-Audio"],
            value="Text-to-Music"
        )
        
        # File uploads
        with gr.Row():
            audio_files = gr.File(
                label="Audio Files",
                file_count="multiple",
                file_types=["audio"]
            )
            
            text_files = gr.File(
                label="Text Files (Optional)",
                file_count="multiple",
                file_types=["text"]
            )
        
        midi_files = gr.File(
            label="MIDI Files (Optional)",
            file_count="multiple",
            file_types=["midi"]
        )
        
        # Processing options
        with gr.Row():
            sample_rate = gr.Dropdown(
                label="Sample Rate",
                choices=["22050", "44100", "48000"],
                value="44100"
            )
            
            audio_duration = gr.Slider(
                label="Max Audio Duration (seconds)",
                minimum=5,
                maximum=60,
                value=30,
                step=5
            )
        
        with gr.Row():
            augmentation = gr.Checkbox(
                label="Apply Data Augmentation",
                value=True
            )
            
            normalize_audio = gr.Checkbox(
                label="Normalize Audio",
                value=True
            )
        
        # Create dataset button
        create_dataset_button = gr.Button("Create Dataset", variant="primary")
        
        # Output components
        dataset_status = gr.Textbox(
            label="Dataset Creation Status",
            interactive=False
        )
        
        dataset_info = gr.JSON(
            label="Dataset Information"
        )
        
        # Function to create dataset
        def create_dataset(
            name, description, dataset_type, audio_files, text_files, midi_files,
            sample_rate, audio_duration, augmentation, normalize_audio
        ):
            try:
                if not name:
                    return "Error: Dataset name is required", None
                
                # Create a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                
                # Create dataset directory
                dataset_dir = self.datasets_dir / f"{name}_{timestamp}"
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Create subdirectories
                audio_dir = dataset_dir / "audio"
                text_dir = dataset_dir / "text"
                midi_dir = dataset_dir / "midi"
                
                os.makedirs(audio_dir, exist_ok=True)
                os.makedirs(text_dir, exist_ok=True)
                os.makedirs(midi_dir, exist_ok=True)
                
                # Process audio files
                processed_audio = []
                if audio_files:
                    for audio_file in audio_files:
                        # Get base filename
                        filename = Path(audio_file.name).name
                        
                        # Save audio file
                        target_path = audio_dir / filename
                        with open(target_path, "wb") as f:
                            f.write(open(audio_file.name, "rb").read())
                        
                        processed_audio.append(str(target_path))
                
                # Process text files
                processed_text = []
                if text_files:
                    for text_file in text_files:
                        # Get base filename
                        filename = Path(text_file.name).name
                        
                        # Save text file
                        target_path = text_dir / filename
                        with open(target_path, "w") as f:
                            f.write(open(text_file.name, "r").read())
                        
                        processed_text.append(str(target_path))
                
                # Process MIDI files
                processed_midi = []
                if midi_files:
                    for midi_file in midi_files:
                        # Get base filename
                        filename = Path(midi_file.name).name
                        
                        # Save MIDI file
                        target_path = midi_dir / filename
                        with open(target_path, "wb") as f:
                            f.write(open(midi_file.name, "rb").read())
                        
                        processed_midi.append(str(target_path))
                
                # Create metadata file
                metadata = {
                    "name": name,
                    "description": description,
                    "type": dataset_type,
                    "created_at": datetime.datetime.now().isoformat(),
                    "sample_rate": int(sample_rate),
                    "max_duration": audio_duration,
                    "augmentation": augmentation,
                    "normalize_audio": normalize_audio,
                    "audio_files": processed_audio,
                    "text_files": processed_text,
                    "midi_files": processed_midi
                }
                
                # Save metadata
                with open(dataset_dir / "metadata.json", "w") as f:
                    import json
                    json.dump(metadata, f, indent=2)
                
                # Prepare dataset info for display
                info = {
                    "name": name,
                    "path": str(dataset_dir),
                    "num_audio_files": len(processed_audio),
                    "num_text_files": len(processed_text),
                    "num_midi_files": len(processed_midi),
                    "type": dataset_type,
                    "sample_rate": sample_rate,
                    "created_at": metadata["created_at"]
                }
                
                return f"Dataset '{name}' created successfully at {dataset_dir}", info
                
            except Exception as e:
                return f"Error creating dataset: {str(e)}", None
        
        # Connect create dataset button
        create_dataset_button.click(
            create_dataset,
            inputs=[
                dataset_name, dataset_description, dataset_type,
                audio_files, text_files, midi_files,
                sample_rate, audio_duration, augmentation, normalize_audio
            ],
            outputs=[dataset_status, dataset_info]
        )
        
        return [
            dataset_name, dataset_description, dataset_type,
            audio_files, text_files, midi_files,
            sample_rate, audio_duration, augmentation, normalize_audio,
            create_dataset_button, dataset_status, dataset_info
        ]
    
    def training_interface(self) -> List[gr.Component]:
        """
        Create the model training interface components.
        
        Returns:
            List of Gradio components.
        """
        # Get available models
        models = []
        for task in ["text_to_music", "audio_to_music", "midi_to_audio"]:
            models.extend(self.model_service.get_models_for_task(task))
        
        model_choices = [model["id"] for model in models] if models else []
        
        # Model selection
        model_dropdown = gr.Dropdown(
            label="Base Model",
            choices=model_choices,
            value=model_choices[0] if model_choices else None
        )
        
        # Training name
        training_name = gr.Textbox(
            label="Training Run Name",
            placeholder="Enter a name for this training run"
        )
        
        # Dataset selection
        dataset_dropdown = gr.Dropdown(
            label="Dataset",
            choices=self._get_available_datasets(),
            value=None
        )
        
        # LoRA parameters
        with gr.Row():
            use_lora = gr.Checkbox(
                label="Use LoRA (Parameter-Efficient Fine-Tuning)",
                value=True
            )
            
            lora_rank = gr.Slider(
                label="LoRA Rank",
                minimum=1,
                maximum=64,
                value=16,
                step=1
            )
        
        # Training parameters
        with gr.Row():
            batch_size = gr.Slider(
                label="Batch Size",
                minimum=1,
                maximum=32,
                value=4,
                step=1
            )
            
            learning_rate = gr.Slider(
                label="Learning Rate",
                minimum=1e-6,
                maximum=1e-3,
                value=2e-5,
                step=1e-6
            )
        
        with gr.Row():
            num_epochs = gr.Slider(
                label="Number of Epochs",
                minimum=1,
                maximum=30,
                value=3,
                step=1
            )
            
            weight_decay = gr.Slider(
                label="Weight Decay",
                minimum=0,
                maximum=0.1,
                value=0.01,
                step=0.01
            )
        
        # Advanced parameters
        with gr.Accordion("Advanced Parameters", open=False):
            warmup_steps = gr.Slider(
                label="Warmup Steps",
                minimum=0,
                maximum=1000,
                value=500,
                step=100
            )
            
            gradient_accumulation = gr.Slider(
                label="Gradient Accumulation Steps",
                minimum=1,
                maximum=16,
                value=4,
                step=1
            )
            
            use_fp16 = gr.Checkbox(
                label="Use Mixed Precision (FP16)",
                value=torch.cuda.is_available()
            )
            
            early_stopping_patience = gr.Slider(
                label="Early Stopping Patience",
                minimum=1,
                maximum=10,
                value=3,
                step=1
            )
        
        # Training device
        training_device = gr.Radio(
            label="Training Device",
            choices=["Local", "Google Colab"],
            value="Local"
        )
        
        # Start training button
        start_training_button = gr.Button("Start Training", variant="primary")
        
        # Output components
        training_status = gr.Textbox(
            label="Training Status",
            interactive=False
        )
        
        training_progress = gr.HTML(
            label="Training Progress"
        )
        
        # Function to prepare for training
        def prepare_training(
            model_id, training_name, dataset_name, use_lora, lora_rank,
            batch_size, learning_rate, num_epochs, weight_decay,
            warmup_steps, gradient_accumulation, use_fp16, early_stopping_patience,
            training_device
        ):
            try:
                if not model_id:
                    return "Error: Please select a base model", "<p>No model selected</p>"
                
                if not dataset_name:
                    return "Error: Please select a dataset", "<p>No dataset selected</p>"
                
                if not training_name:
                    training_name = f"training_{model_id.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                
                # Check if model is available locally
                if not self.model_service.is_model_available_locally(model_id):
                    return f"Error: Model {model_id} is not available locally. Please download it first.", "<p>Model not available</p>"
                
                # Check if dataset exists
                dataset_path = self._get_dataset_path(dataset_name)
                if not dataset_path:
                    return f"Error: Dataset {dataset_name} not found", "<p>Dataset not found</p>"
                
                # Check if PEFT is available if LoRA is requested
                if use_lora:
                    try:
                        import peft
                    except ImportError:
                        return "Error: LoRA requested but PEFT library not installed. Please install it with: pip install peft", "<p>PEFT not installed</p>"
                
                # Check if training device is available
                if training_device == "Google Colab":
                    return "Training on Google Colab will be available soon. Please use local training for now.", "<p>Google Colab not yet supported</p>"
                
                # Prepare training parameters
                training_params = {
                    "model_id": model_id,
                    "run_name": training_name,
                    "dataset_path": str(dataset_path),
                    "use_lora": use_lora,
                    "lora_rank": lora_rank,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "gradient_accumulation_steps": gradient_accumulation,
                    "use_fp16": use_fp16,
                    "early_stopping_patience": early_stopping_patience,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
                
                # Generate a training plan
                dataset_info = self._get_dataset_info(dataset_name)
                dataset_type = dataset_info.get("type", "Unknown")
                
                html = f"""
                <h3>Training Plan</h3>
                <ul>
                    <li><strong>Model:</strong> {model_id}</li>
                    <li><strong>Dataset:</strong> {dataset_name} ({dataset_type})</li>
                    <li><strong>Fine-tuning Method:</strong> {"LoRA" if use_lora else "Full Fine-tuning"}</li>
                    <li><strong>Batch Size:</strong> {batch_size}</li>
                    <li><strong>Learning Rate:</strong> {learning_rate}</li>
                    <li><strong>Epochs:</strong> {num_epochs}</li>
                    <li><strong>Device:</strong> {training_params["device"]}</li>
                    <li><strong>Estimated Time:</strong> {"Unknown - depends on dataset size and hardware"}</li>
                </ul>
                <p>Click 'Start Training' to begin, or adjust parameters as needed.</p>
                """
                
                return "Training plan prepared. Click 'Start Training' to begin.", html
                
            except Exception as e:
                return f"Error preparing training: {str(e)}", "<p>Error preparing training</p>"
        
        # Function to start training
        def start_training(
            model_id, training_name, dataset_name, use_lora, lora_rank,
            batch_size, learning_rate, num_epochs, weight_decay,
            warmup_steps, gradient_accumulation, use_fp16, early_stopping_patience,
            training_device
        ):
            try:
                # This is a placeholder for actual training
                # In a real implementation, this would:
                # 1. Load the model and dataset
                # 2. Set up the trainer
                # 3. Start training in a separate thread or process
                # 4. Return updates on training progress
                
                # For now, just show a simulated progress
                status = "Training started. This is a simulation for UI demonstration."
                
                # Create a progress HTML that would be updated in a real implementation
                html = f"""
                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                    <h3>Training Progress</h3>
                    <p><strong>Model:</strong> {model_id}</p>
                    <p><strong>Dataset:</strong> {dataset_name}</p>
                    <p><strong>Status:</strong> Initializing...</p>
                    <div style="width: 100%; background-color: #f1f1f1; border-radius: 5px;">
                        <div style="width: 5%; height: 20px; background-color: #4CAF50; border-radius: 5px;"></div>
                    </div>
                    <p><strong>Current Epoch:</strong> 0/{num_epochs}</p>
                    <p><strong>Loss:</strong> N/A</p>
                    <p><strong>Learning Rate:</strong> {learning_rate}</p>
                    <p><strong>Elapsed Time:</strong> 00:00:00</p>
                    <p><strong>Estimated Time Remaining:</strong> Unknown</p>
                </div>
                """
                
                return status, html
                
            except Exception as e:
                return f"Error starting training: {str(e)}", "<p>Error starting training</p>"
        
        # Connect components
        # First prepare the training plan when parameters change
        parameter_inputs = [
            model_dropdown, training_name, dataset_dropdown, use_lora, lora_rank,
            batch_size, learning_rate, num_epochs, weight_decay,
            warmup_steps, gradient_accumulation, use_fp16, early_stopping_patience,
            training_device
        ]
        
        for param in parameter_inputs:
            param.change(
                prepare_training,
                inputs=parameter_inputs,
                outputs=[training_status, training_progress]
            )
        
        # Then start training when the button is clicked
        start_training_button.click(
            start_training,
            inputs=parameter_inputs,
            outputs=[training_status, training_progress]
        )
        
        # Add a refresh button for datasets
        refresh_datasets_button = gr.Button("Refresh Datasets")
        
        def refresh_datasets():
            return gr.Dropdown.update(choices=self._get_available_datasets())
        
        refresh_datasets_button.click(
            refresh_datasets,
            inputs=[],
            outputs=[dataset_dropdown]
        )
        
        return [
            model_dropdown, training_name, dataset_dropdown, refresh_datasets_button,
            use_lora, lora_rank, batch_size, learning_rate, num_epochs, weight_decay,
            warmup_steps, gradient_accumulation, use_fp16, early_stopping_patience,
            training_device, start_training_button, training_status, training_progress
        ]
    
    def _get_available_datasets(self) -> List[str]:
        """
        Get a list of available datasets.
        
        Returns:
            List of dataset names.
        """
        if not self.datasets_dir.exists():
            return []
            
        datasets = []
        for item in self.datasets_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                datasets.append(item.name)
        
        return datasets
    
    def _get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """
        Get the path to a dataset.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Path to the dataset or None if not found.
        """
        dataset_path = self.datasets_dir / dataset_name
        if dataset_path.exists() and (dataset_path / "metadata.json").exists():
            return dataset_path
        
        return None
    
    def _get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Dictionary with dataset information.
        """
        dataset_path = self._get_dataset_path(dataset_name)
        if not dataset_path:
            return {}
        
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading dataset metadata: {e}")
        
        return {}