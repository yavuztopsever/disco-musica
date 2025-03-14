"""
Model training module for Disco Musica.

This module provides utilities for fine-tuning AI music generation models
with different techniques and parameter-efficient methods.
"""

import os
import time
import datetime
import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)

from modules.core.config import config
from modules.utils.logging_utils import TrainingLogger


class ModelTrainer:
    """
    Class for managing model training and fine-tuning.
    
    This class provides utilities for fine-tuning AI music generation models
    with different techniques, including parameter-efficient methods like LoRA.
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        use_peft: bool = True
    ):
        """
        Initialize the ModelTrainer.
        
        Args:
            output_dir: Directory to save training outputs. If None, uses the default from config.
            use_peft: Whether to use Parameter-Efficient Fine-Tuning (e.g., LoRA) techniques.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(config.get("paths", "output_dir", "outputs")) / "training"
        self.use_peft = use_peft
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(self.output_dir / "logs", "model_trainer")
        
        # Initialize PEFT if requested
        if use_peft:
            self._initialize_peft()
    
    def _initialize_peft(self) -> None:
        """
        Initialize Parameter-Efficient Fine-Tuning (PEFT) library.
        """
        try:
            import peft
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            print("PEFT library initialized successfully")
            self.peft_available = True
        except ImportError:
            print("PEFT library not found. Install with: pip install peft")
            print("Falling back to regular fine-tuning")
            self.peft_available = False
    
    def prepare_model_for_training(
        self,
        model: Any,
        lora_config: Optional[Dict] = None
    ) -> Any:
        """
        Prepare a model for training, potentially with PEFT techniques.
        
        Args:
            model: The model to prepare for training.
            lora_config: Configuration for LoRA. If None, uses default values.
            
        Returns:
            The prepared model.
        """
        # Ensure model is in training mode
        model.train()
        
        # Apply LoRA if PEFT is available and requested
        if self.use_peft and self.peft_available:
            try:
                from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
                
                # Prepare defaults for LoRA
                default_lora_config = {
                    "r": config.get("training", "lora_rank", 16),
                    "lora_alpha": config.get("training", "lora_alpha", 32),
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
                
                # Update with provided config if any
                if lora_config:
                    default_lora_config.update(lora_config)
                
                # Get target modules (layers to apply LoRA to)
                # Default to specific attention layers if no modules specified
                if "target_modules" not in default_lora_config:
                    # This is a common pattern for transformer models - may need adjustment per model
                    default_lora_config["target_modules"] = ["q_proj", "v_proj"]
                
                # Create LoRA config
                peft_config = LoraConfig(**default_lora_config)
                
                # Prepare model for training if it's a quantized model
                if hasattr(model, 'is_quantized') and model.is_quantized:
                    model = prepare_model_for_kbit_training(model)
                
                # Apply LoRA
                model = get_peft_model(model, peft_config)
                
                # Print trainable parameters info
                model.print_trainable_parameters()
                
                return model
                
            except Exception as e:
                print(f"Error applying LoRA: {e}")
                print("Falling back to regular fine-tuning")
        
        # Return the original model if PEFT is not available or failed
        return model
    
    def create_training_args(
        self,
        run_name: str,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        use_fp16: Optional[bool] = None,
        **kwargs
    ) -> TrainingArguments:
        """
        Create training arguments for the Trainer.
        
        Args:
            run_name: Name of the training run.
            batch_size: Training batch size.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of training epochs.
            warmup_steps: Number of warmup steps.
            weight_decay: Weight decay for the optimizer.
            use_fp16: Whether to use fp16 training. If None, determined automatically.
            **kwargs: Additional arguments for TrainingArguments.
            
        Returns:
            TrainingArguments instance.
        """
        # Determine if fp16 should be used
        if use_fp16 is None:
            # Enable fp16 if CUDA is available and not using CPU-only mode
            use_fp16 = torch.cuda.is_available() and not kwargs.get('no_cuda', False)
        
        # Create output dir with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = self.output_dir / f"{run_name}_{timestamp}"
        
        # Prepare default training arguments
        default_args = {
            "output_dir": output_dir,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": 4,  # To effectively increase batch size
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "warmup_steps": warmup_steps,
            "logging_dir": output_dir / "logs",
            "logging_steps": 100,
            "weight_decay": weight_decay,
            "fp16": use_fp16,
            "report_to": "tensorboard",
            "run_name": run_name,
            "remove_unused_columns": False,  # Often needed for custom datasets
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False
        }
        
        # Update with additional arguments
        default_args.update(kwargs)
        
        # Create TrainingArguments
        return TrainingArguments(**default_args)
    
    def prepare_callbacks(self, patience: int = 3) -> List:
        """
        Prepare callbacks for training.
        
        Args:
            patience: Number of evaluation calls with no improvement after which
                     training will be stopped.
            
        Returns:
            List of callbacks.
        """
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=patience)
        ]
        
        return callbacks
    
    def fine_tune(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        collate_fn: Optional[Any] = None,
        training_args: Optional[TrainingArguments] = None,
        callbacks: Optional[List] = None,
        run_name: str = "fine_tune",
        **kwargs
    ) -> Any:
        """
        Fine-tune a model on a dataset.
        
        Args:
            model: The model to fine-tune.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset. If None, a portion of the train dataset will be used.
            collate_fn: The collation function for batching. If None, a default one will be used.
            training_args: The training arguments. If None, default args will be created.
            callbacks: List of callbacks. If None, default callbacks will be used.
            run_name: Name of the training run.
            **kwargs: Additional arguments for TrainingArguments.
            
        Returns:
            The fine-tuned model.
        """
        # Prepare model for training (potentially with PEFT)
        prepared_model = self.prepare_model_for_training(model, kwargs.get("lora_config"))
        
        # Create training arguments if not provided
        if training_args is None:
            training_args = self.create_training_args(run_name=run_name, **kwargs)
        
        # Create callbacks if not provided
        if callbacks is None:
            callbacks = self.prepare_callbacks(patience=kwargs.get("patience", 3))
        
        # Create collation function if not provided
        if collate_fn is None:
            # Default to DataCollatorForLanguageModeling for language models
            try:
                collate_fn = DataCollatorForLanguageModeling(
                    tokenizer=getattr(model, "tokenizer", None),
                    mlm=False
                )
            except Exception as e:
                print(f"Could not create default collate_fn: {e}")
                print("You may need to provide a custom collate_fn")
        
        # Create a portion of train_dataset as eval_dataset if not provided
        if eval_dataset is None and train_dataset is not None:
            from torch.utils.data import random_split
            
            # Use 10% of training data for evaluation
            eval_size = max(1, int(len(train_dataset) * 0.1))
            train_size = len(train_dataset) - eval_size
            
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size]
            )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=prepared_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            callbacks=callbacks
        )
        
        # Start training
        try:
            # Log training start
            self.logger.logger.info(f"Starting training run: {run_name}")
            
            # Train the model
            train_result = trainer.train()
            
            # Log training metrics
            self.logger.logger.info(f"Training metrics: {train_result.metrics}")
            
            # Save the model
            trainer.save_model(training_args.output_dir)
            
            # Save training arguments
            with open(Path(training_args.output_dir) / "training_args.json", "w") as f:
                json.dump(training_args.to_dict(), f, indent=2)
            
            # Save trainer state
            trainer.save_state()
            
            # Log completion
            self.logger.logger.info(f"Training completed successfully")
            
            return prepared_model
            
        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(
        self,
        model: Any,
        eval_dataset: Any,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: The model to evaluate.
            eval_dataset: The evaluation dataset.
            batch_size: Batch size for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Set model to evaluation mode
        model.eval()
        
        # Create evaluation arguments
        eval_args = TrainingArguments(
            output_dir=self.output_dir / "eval_temp",
            per_device_eval_batch_size=batch_size,
            report_to="none",
            remove_unused_columns=False
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset
        )
        
        # Evaluate the model
        metrics = trainer.evaluate()
        
        return metrics
    
    def save_lora_adapters(
        self,
        model: Any,
        adapter_name: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save LoRA adapters separately from the base model.
        
        Args:
            model: The model with LoRA adapters.
            adapter_name: Name of the adapter.
            output_dir: Directory to save the adapter. If None, uses the default.
            
        Returns:
            Path to the saved adapter.
        """
        if not hasattr(model, "save_pretrained"):
            raise ValueError("Model does not have save_pretrained method")
        
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = self.output_dir / f"lora_adapter_{adapter_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save adapter
        model.save_pretrained(output_dir)
        
        # Save adapter metadata
        adapter_metadata = {
            "name": adapter_name,
            "created_at": datetime.datetime.now().isoformat(),
            "peft_type": "LORA"
        }
        
        with open(output_dir / "adapter_metadata.json", "w") as f:
            json.dump(adapter_metadata, f, indent=2)
        
        return output_dir
    
    def load_lora_adapters(
        self,
        model: Any,
        adapter_path: Union[str, Path]
    ) -> Any:
        """
        Load LoRA adapters into a base model.
        
        Args:
            model: The base model.
            adapter_path: Path to the LoRA adapter.
            
        Returns:
            Model with loaded LoRA adapter.
        """
        if not self.peft_available:
            raise ImportError("PEFT library is required to load LoRA adapters")
        
        try:
            from peft import PeftModel, PeftConfig
            
            # Load the configuration
            adapter_path = Path(adapter_path)
            config = PeftConfig.from_pretrained(adapter_path)
            
            # Load the model with the adapter
            model = PeftModel.from_pretrained(model, adapter_path)
            
            return model
            
        except Exception as e:
            self.logger.logger.error(f"Failed to load LoRA adapter: {e}")
            raise


# Helper functions for dataset preparation

def create_music_dataset_from_audio_files(
    audio_dir: Union[str, Path],
    tokenizer: Any,
    sample_rate: int = 44100,
    max_length: int = 512,
    audio_processor: Optional[Any] = None
) -> Any:
    """
    Create a dataset from audio files for model training.
    
    Args:
        audio_dir: Directory containing audio files.
        tokenizer: Tokenizer to use for text processing.
        sample_rate: Sample rate for audio processing.
        max_length: Maximum sequence length.
        audio_processor: Audio processor to use. If None, a basic one will be created.
        
    Returns:
        PyTorch Dataset.
    """
    from torch.utils.data import Dataset
    import librosa
    import os
    
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    class AudioDataset(Dataset):
        def __init__(self, audio_files, tokenizer, sample_rate, max_length, audio_processor):
            self.audio_files = audio_files
            self.tokenizer = tokenizer
            self.sample_rate = sample_rate
            self.max_length = max_length
            self.audio_processor = audio_processor
            
            # Create default audio processor if not provided
            if self.audio_processor is None:
                from modules.core.audio_processor import AudioProcessor
                self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        def __len__(self):
            return len(self.audio_files)
        
        def __getitem__(self, idx):
            audio_path = self.audio_files[idx]
            
            # Get file name as a simple description
            file_name = os.path.basename(audio_path).split('.')[0].replace('_', ' ')
            
            # Load audio with the processor
            audio_data, _ = self.audio_processor.load_audio(audio_path, self.sample_rate)
            
            # Create inputs
            inputs = {
                "audio": audio_data,
                "text": f"Generate music similar to {file_name}"
            }
            
            # Process text with tokenizer
            tokenized_text = self.tokenizer(
                inputs["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Return inputs
            return {
                "audio": inputs["audio"],
                "input_ids": tokenized_text["input_ids"].squeeze(),
                "attention_mask": tokenized_text["attention_mask"].squeeze()
            }
    
    return AudioDataset(audio_files, tokenizer, sample_rate, max_length, audio_processor)


def create_dataset_from_text_audio_pairs(
    dataset_dir: Union[str, Path],
    tokenizer: Any,
    sample_rate: int = 44100,
    max_length: int = 512,
    audio_processor: Optional[Any] = None
) -> Any:
    """
    Create a dataset from text-audio pairs for model training.
    
    Args:
        dataset_dir: Directory containing text-audio pairs.
        tokenizer: Tokenizer to use for text processing.
        sample_rate: Sample rate for audio processing.
        max_length: Maximum sequence length.
        audio_processor: Audio processor to use. If None, a basic one will be created.
        
    Returns:
        PyTorch Dataset.
    """
    from torch.utils.data import Dataset
    import json
    import os
    
    # Look for a metadata file that maps audio files to descriptions
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    pairs = []
    
    if os.path.exists(metadata_file):
        # Load pairs from metadata file
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
            for item in metadata:
                audio_path = os.path.join(dataset_dir, item["audio_file"])
                text = item["text"]
                pairs.append((audio_path, text))
    else:
        # Look for text files with the same base name as audio files
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_path = os.path.join(root, file)
                    text_path = os.path.join(root, os.path.splitext(file)[0] + ".txt")
                    
                    if os.path.exists(text_path):
                        with open(text_path, "r") as f:
                            text = f.read().strip()
                            pairs.append((audio_path, text))
    
    class TextAudioDataset(Dataset):
        def __init__(self, pairs, tokenizer, sample_rate, max_length, audio_processor):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.sample_rate = sample_rate
            self.max_length = max_length
            self.audio_processor = audio_processor
            
            # Create default audio processor if not provided
            if self.audio_processor is None:
                from modules.core.audio_processor import AudioProcessor
                self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            audio_path, text = self.pairs[idx]
            
            # Load audio with the processor
            audio_data, _ = self.audio_processor.load_audio(audio_path, self.sample_rate)
            
            # Process text with tokenizer
            tokenized_text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Return inputs
            return {
                "audio": audio_data,
                "input_ids": tokenized_text["input_ids"].squeeze(),
                "attention_mask": tokenized_text["attention_mask"].squeeze(),
                "text": text  # Keep original text for reference
            }
    
    return TextAudioDataset(pairs, tokenizer, sample_rate, max_length, audio_processor)