"""
Logging Utilities

This module provides utilities for logging in the Disco Musica application.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str, log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with the specified name, log file, and level.

    Args:
        name: Name of the logger.
        log_file: Path to the log file.
        level: Logging level.

    Returns:
        Configured logger.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_file = Path(log_file)
        
        # Create directory if it doesn't exist
        os.makedirs(log_file.parent, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger.

    Returns:
        Logger with the specified name.
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    A class for logging training information.
    """

    def __init__(self, log_dir: Union[str, Path], model_name: str):
        """
        Initialize the TrainingLogger.

        Args:
            log_dir: Directory to store the logs.
            model_name: Name of the model being trained.
        """
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = self.log_dir / f"{model_name}_{timestamp}.log"
        
        # Set up logger
        self.logger = setup_logger(f"training.{model_name}", log_file)
        
        # Initialize training metrics
        self.metrics = {
            "loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": []
        }
    
    def log_batch(self, step: int, loss: float, learning_rate: float, epoch: int) -> None:
        """
        Log batch information.

        Args:
            step: Current step.
            loss: Loss value.
            learning_rate: Learning rate.
            epoch: Current epoch.
        """
        # Log to file
        self.logger.info(f"Step: {step}, Epoch: {epoch}, Loss: {loss:.4f}, LR: {learning_rate:.6f}")
        
        # Store metrics
        self.metrics["loss"].append(loss)
        self.metrics["learning_rate"].append(learning_rate)
        self.metrics["epoch"].append(epoch)
        self.metrics["step"].append(step)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None) -> None:
        """
        Log epoch information.

        Args:
            epoch: Current epoch.
            train_loss: Training loss.
            val_loss: Validation loss.
        """
        if val_loss is not None:
            self.logger.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            self.logger.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")
    
    def log_model_save(self, path: Union[str, Path]) -> None:
        """
        Log model save information.

        Args:
            path: Path where the model was saved.
        """
        self.logger.info(f"Model saved to: {path}")
    
    def log_error(self, error: str) -> None:
        """
        Log an error.

        Args:
            error: Error message.
        """
        self.logger.error(error)
    
    def get_metrics(self) -> dict:
        """
        Get the training metrics.

        Returns:
            Dictionary of training metrics.
        """
        return self.metrics
    
    def save_metrics(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the training metrics to a file.

        Args:
            path: Path to save the metrics to.

        Returns:
            Path to the saved metrics file.
        """
        import json
        
        if path is None:
            # Generate metrics file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.log_dir / f"{self.model_name}_{timestamp}_metrics.json"
        else:
            path = Path(path)
        
        # Create directory if it doesn't exist
        os.makedirs(path.parent, exist_ok=True)
        
        # Save metrics to file
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {path}")
        return path


class InferenceLogger:
    """
    A class for logging inference information.
    """

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize the InferenceLogger.

        Args:
            log_dir: Directory to store the logs.
        """
        self.log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = self.log_dir / f"inference_{timestamp}.log"
        
        # Set up logger
        self.logger = setup_logger("inference", log_file)
    
    def log_generation(
        self, model_name: str, input_type: str, generation_time: float, output_path: Union[str, Path]
    ) -> None:
        """
        Log generation information.

        Args:
            model_name: Name of the model used for generation.
            input_type: Type of input ('text', 'audio', 'midi', 'image').
            generation_time: Generation time in seconds.
            output_path: Path to the generated output.
        """
        self.logger.info(
            f"Generation: Model={model_name}, Input={input_type}, " +
            f"Time={generation_time:.2f}s, Output={output_path}"
        )
    
    def log_error(self, error: str, model_name: Optional[str] = None) -> None:
        """
        Log an error.

        Args:
            error: Error message.
            model_name: Name of the model that encountered the error.
        """
        if model_name:
            self.logger.error(f"Model={model_name}, Error={error}")
        else:
            self.logger.error(error)


# Example usage
if __name__ == "__main__":
    # Set up a logger
    logger = setup_logger("disco_musica", "logs/disco_musica.log")
    logger.info("Disco Musica logger initialized")
    
    # Create a training logger
    training_logger = TrainingLogger("logs/training", "musicgen-small")
    training_logger.log_batch(1, 2.5, 0.001, 1)
    
    # Create an inference logger
    inference_logger = InferenceLogger("logs/inference")
    inference_logger.log_generation("musicgen-small", "text", 1.5, "outputs/audio/generation.wav")