"""Time series analysis model for handling time series data."""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource
from ...core.config import ModelConfig

logger = logging.getLogger(__name__)

class TimeSeriesConfig(BaseModel):
    """Configuration for the time series model."""
    model_name: str
    model_version: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    sequence_length: int = 100
    prediction_length: int = 10
    metadata: Dict[str, Any] = {}

class TimeSeriesModel:
    """Model for handling time series data."""
    
    def __init__(
        self,
        config: TimeSeriesConfig,
        model: Optional[nn.Module] = None
    ):
        """Initialize the time series model.
        
        Args:
            config: Model configuration.
            model: Optional pre-trained model.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model()
            
    def _load_model(self) -> nn.Module:
        """Load the pre-trained model.
        
        Returns:
            Loaded model.
        """
        try:
            # Load model from registry
            model = ModelRegistry.get_model(
                self.config.model_name,
                self.config.model_version
            )
            return model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}")
            
    async def predict(
        self,
        input_sequence: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict future values from input sequence.
        
        Args:
            input_sequence: Input time series sequence.
            **kwargs: Additional prediction parameters.
            
        Returns:
            Predictions and metadata.
        """
        try:
            # Move input to device
            input_sequence = input_sequence.to(self.device)
            
            # Update prediction parameters
            pred_kwargs = self.config.dict()
            pred_kwargs.update(kwargs)
            
            # Predict
            with torch.no_grad():
                outputs = self.model.predict(
                    input_sequence=input_sequence,
                    **pred_kwargs
                )
                
            # Process outputs
            predictions = outputs.predictions
            confidence_intervals = outputs.confidence_intervals if hasattr(outputs, "confidence_intervals") else None
            
            # Convert to numpy arrays
            predictions = predictions.cpu().numpy()
            if confidence_intervals is not None:
                confidence_intervals = confidence_intervals.cpu().numpy()
                
            return {
                "predictions": predictions,
                "confidence_intervals": confidence_intervals,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "prediction_params": pred_kwargs,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise DiscoMusicaError(f"Prediction failed: {e}")
            
    async def batch_predict(
        self,
        input_sequences: List[torch.Tensor],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Predict future values from multiple input sequences.
        
        Args:
            input_sequences: List of input time series sequences.
            **kwargs: Additional prediction parameters.
            
        Returns:
            List of predictions and metadata.
        """
        try:
            results = []
            for input_sequence in input_sequences:
                # Predict sequence
                result = await self.predict(
                    input_sequence=input_sequence,
                    **kwargs
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise DiscoMusicaError(f"Batch prediction failed: {e}")
            
    async def detect_anomalies(
        self,
        sequence: torch.Tensor,
        threshold: float = 2.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Detect anomalies in time series sequence.
        
        Args:
            sequence: Input time series sequence.
            threshold: Anomaly detection threshold.
            **kwargs: Additional detection parameters.
            
        Returns:
            Anomaly detection results.
        """
        try:
            # Move sequence to device
            sequence = sequence.to(self.device)
            
            # Update detection parameters
            detect_kwargs = self.config.dict()
            detect_kwargs.update(kwargs)
            
            # Detect anomalies
            with torch.no_grad():
                outputs = self.model.detect_anomalies(
                    sequence=sequence,
                    threshold=threshold,
                    **detect_kwargs
                )
                
            # Process outputs
            anomalies = outputs.anomalies
            scores = outputs.scores if hasattr(outputs, "scores") else None
            
            # Convert to numpy arrays
            anomalies = anomalies.cpu().numpy()
            if scores is not None:
                scores = scores.cpu().numpy()
                
            return {
                "anomalies": anomalies,
                "scores": scores,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "detection_params": detect_kwargs,
                    "threshold": threshold,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            raise DiscoMusicaError(f"Anomaly detection failed: {e}")
            
    async def analyze_trends(
        self,
        sequence: torch.Tensor,
        window_size: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze trends in time series sequence.
        
        Args:
            sequence: Input time series sequence.
            window_size: Size of sliding window.
            **kwargs: Additional analysis parameters.
            
        Returns:
            Trend analysis results.
        """
        try:
            # Move sequence to device
            sequence = sequence.to(self.device)
            
            # Update analysis parameters
            analyze_kwargs = self.config.dict()
            analyze_kwargs.update(kwargs)
            
            # Analyze trends
            with torch.no_grad():
                outputs = self.model.analyze_trends(
                    sequence=sequence,
                    window_size=window_size,
                    **analyze_kwargs
                )
                
            # Process outputs
            trends = outputs.trends
            components = outputs.components if hasattr(outputs, "components") else None
            
            # Convert to numpy arrays
            trends = trends.cpu().numpy()
            if components is not None:
                components = components.cpu().numpy()
                
            return {
                "trends": trends,
                "components": components,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "analysis_params": analyze_kwargs,
                    "window_size": window_size,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            raise DiscoMusicaError(f"Trend analysis failed: {e}")
            
    async def forecast(
        self,
        sequence: torch.Tensor,
        horizon: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Forecast future values from time series sequence.
        
        Args:
            sequence: Input time series sequence.
            horizon: Forecast horizon.
            **kwargs: Additional forecast parameters.
            
        Returns:
            Forecast results.
        """
        try:
            # Move sequence to device
            sequence = sequence.to(self.device)
            
            # Update forecast parameters
            forecast_kwargs = self.config.dict()
            forecast_kwargs.update(kwargs)
            
            # Forecast
            with torch.no_grad():
                outputs = self.model.forecast(
                    sequence=sequence,
                    horizon=horizon,
                    **forecast_kwargs
                )
                
            # Process outputs
            forecast = outputs.forecast
            confidence_intervals = outputs.confidence_intervals if hasattr(outputs, "confidence_intervals") else None
            
            # Convert to numpy arrays
            forecast = forecast.cpu().numpy()
            if confidence_intervals is not None:
                confidence_intervals = confidence_intervals.cpu().numpy()
                
            return {
                "forecast": forecast,
                "confidence_intervals": confidence_intervals,
                "metadata": {
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                    "forecast_params": forecast_kwargs,
                    "horizon": horizon,
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Forecasting failed: {e}")
            raise DiscoMusicaError(f"Forecasting failed: {e}")
            
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model.
        """
        try:
            # Save model state
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config.dict()
                },
                path
            )
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise DiscoMusicaError(f"Failed to save model: {e}")
            
    @classmethod
    def load(cls, path: str) -> "TimeSeriesModel":
        """Load a model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            Loaded model.
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(path)
            
            # Create config
            config = TimeSeriesConfig(**checkpoint["config"])
            
            # Create model
            model = cls(config)
            
            # Load state dict
            model.model.load_state_dict(checkpoint["model_state_dict"])
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise DiscoMusicaError(f"Failed to load model: {e}") 