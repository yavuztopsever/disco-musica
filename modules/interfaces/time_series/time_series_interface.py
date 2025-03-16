"""Time series analytics interface for handling time series data analysis."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from dtaidistance import dtw
from pydantic import BaseModel
import torch

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource
from ...data.storage.time_series_storage import TimeSeriesStorage
from ...models.time_series.time_series_model import TimeSeriesModel

logger = logging.getLogger(__name__)

class TimeSeriesQuery(BaseModel):
    """Model for time series queries."""
    metric_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    interval: Optional[timedelta] = None
    aggregation: str = "mean"
    metadata_filter: Optional[Dict[str, Any]] = None

class TimeSeriesAnalysis(BaseModel):
    """Model for time series analysis results."""
    metric_id: str
    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]

class TimeSeriesInterface:
    """Interface for handling time series data analysis."""
    
    def __init__(
        self,
        time_series_storage: TimeSeriesStorage,
        time_series_model: TimeSeriesModel
    ):
        """Initialize the time series interface.
        
        Args:
            time_series_storage: Time series storage service.
            time_series_model: Time series model.
        """
        self.storage = time_series_storage
        self.model = time_series_model
        self.logger = logging.getLogger(__name__)
        
    async def analyze_metric(
        self,
        query: TimeSeriesQuery
    ) -> TimeSeriesAnalysis:
        """Analyze a time series metric.
        
        Args:
            query: Time series query.
            
        Returns:
            Analysis results.
        """
        try:
            # Get metric data
            metric = await self.storage.get_metric(
                metric_id=query.metric_id,
                start_time=query.start_time,
                end_time=query.end_time
            )
            
            if not metric:
                raise ValueError(f"Metric {query.metric_id} not found")
                
            # Get aggregated data if interval specified
            if query.interval:
                data = await self.storage.get_aggregated_metric(
                    metric_id=query.metric_id,
                    start_time=query.start_time,
                    end_time=query.end_time,
                    interval=query.interval,
                    aggregation=query.aggregation
                )
            else:
                data = metric
                
            # Convert to tensor for model input
            sequence = self._prepare_sequence(data)
            
            # Analyze trends
            trend_analysis = await self.model.analyze_trends(
                sequence=sequence,
                window_size=10  # Configurable
            )
            
            # Detect anomalies
            anomaly_analysis = await self.model.detect_anomalies(
                sequence=sequence,
                threshold=2.0  # Configurable
            )
            
            # Generate forecast
            forecast_analysis = await self.model.forecast(
                sequence=sequence,
                horizon=10  # Configurable
            )
            
            return TimeSeriesAnalysis(
                metric_id=query.metric_id,
                analysis_type="comprehensive",
                results={
                    "trends": trend_analysis,
                    "anomalies": anomaly_analysis,
                    "forecast": forecast_analysis
                },
                metadata={
                    "start_time": query.start_time,
                    "end_time": query.end_time,
                    "interval": str(query.interval) if query.interval else None,
                    "aggregation": query.aggregation
                }
            )
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise DiscoMusicaError(f"Analysis failed: {e}")
            
    async def batch_analyze_metrics(
        self,
        queries: List[TimeSeriesQuery]
    ) -> List[TimeSeriesAnalysis]:
        """Analyze multiple time series metrics.
        
        Args:
            queries: List of time series queries.
            
        Returns:
            List of analysis results.
        """
        try:
            results = []
            for query in queries:
                analysis = await self.analyze_metric(query)
                results.append(analysis)
            return results
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            raise DiscoMusicaError(f"Batch analysis failed: {e}")
            
    async def compare_metrics(
        self,
        metric_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Compare multiple time series metrics.
        
        Args:
            metric_ids: List of metric IDs to compare.
            start_time: Optional start time.
            end_time: Optional end time.
            interval: Optional time interval for aggregation.
            
        Returns:
            Comparison results.
        """
        try:
            # Get metrics data
            metrics = []
            for metric_id in metric_ids:
                metric = await self.storage.get_metric(
                    metric_id=metric_id,
                    start_time=start_time,
                    end_time=end_time
                )
                if metric:
                    metrics.append(metric)
                    
            if not metrics:
                return {
                    "comparison_type": "empty",
                    "metrics": [],
                    "results": {}
                }
                
            # Align time series if needed
            aligned_metrics = self._align_metrics(metrics, interval)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(aligned_metrics)
            
            # Calculate similarity scores
            similarity_scores = self._calculate_similarity_scores(aligned_metrics)
            
            # Analyze co-movement patterns
            comovement_analysis = self._analyze_comovement(aligned_metrics)
            
            return {
                "comparison_type": "comprehensive",
                "metrics": metric_ids,
                "results": {
                    "correlation_matrix": correlation_matrix,
                    "similarity_scores": similarity_scores,
                    "comovement_analysis": comovement_analysis
                },
                "metadata": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "interval": str(interval) if interval else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comparison failed: {e}")
            raise DiscoMusicaError(f"Comparison failed: {e}")
            
    async def detect_patterns(
        self,
        metric_id: str,
        pattern_type: str = "seasonal",
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Detect patterns in time series data.
        
        Args:
            metric_id: Metric ID.
            pattern_type: Type of pattern to detect.
            window_size: Size of analysis window.
            
        Returns:
            Pattern detection results.
        """
        try:
            # Get metric data
            metric = await self.storage.get_metric(metric_id)
            if not metric:
                raise ValueError(f"Metric {metric_id} not found")
                
            # Convert to tensor
            sequence = self._prepare_sequence(metric)
            
            if pattern_type == "seasonal":
                # Detect seasonal patterns
                results = await self._detect_seasonal_patterns(
                    sequence,
                    window_size
                )
            elif pattern_type == "cyclic":
                # Detect cyclic patterns
                results = await self._detect_cyclic_patterns(
                    sequence,
                    window_size
                )
            elif pattern_type == "trend":
                # Detect trend patterns
                results = await self._detect_trend_patterns(
                    sequence,
                    window_size
                )
            else:
                raise ValueError(f"Unsupported pattern type: {pattern_type}")
                
            return {
                "pattern_type": pattern_type,
                "metric_id": metric_id,
                "results": results,
                "metadata": {
                    "window_size": window_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            raise DiscoMusicaError(f"Pattern detection failed: {e}")
            
    def _prepare_sequence(
        self,
        data: Union[Dict[str, Any], List[float]]
    ) -> torch.Tensor:
        """Prepare time series data for model input.
        
        Args:
            data: Time series data.
            
        Returns:
            Tensor for model input.
        """
        if isinstance(data, dict):
            # Extract values from metric data
            if "points" in data:
                values = [point.value for point in data["points"]]
            elif "data" in data:
                values = [point["value"] for point in data["data"]]
            else:
                raise ValueError("Invalid data format")
        elif isinstance(data, list):
            values = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        # Convert to tensor
        return torch.tensor(values, dtype=torch.float32)
        
    def _align_metrics(
        self,
        metrics: List[Dict[str, Any]],
        interval: Optional[timedelta] = None
    ) -> List[np.ndarray]:
        """Align multiple time series to common time points.
        
        Args:
            metrics: List of metrics data.
            interval: Optional time interval for resampling.
            
        Returns:
            List of aligned time series.
        """
        try:
            # Convert metrics to pandas DataFrames
            dfs = []
            for metric in metrics:
                # Extract timestamps and values
                if "points" in metric:
                    timestamps = [point.timestamp for point in metric["points"]]
                    values = [point.value for point in metric["points"]]
                elif "data" in metric:
                    timestamps = [point["timestamp"] for point in metric["data"]]
                    values = [point["value"] for point in metric["data"]]
                else:
                    raise ValueError("Invalid metric format")
                
                # Create DataFrame
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps),
                    "value": values
                }).set_index("timestamp")
                dfs.append(df)
            
            # Find common time range
            start_time = max(df.index.min() for df in dfs)
            end_time = min(df.index.max() for df in dfs)
            
            # Resample if interval specified
            if interval:
                freq = pd.Timedelta(interval)
                aligned_dfs = [
                    df.loc[start_time:end_time].resample(freq).mean().interpolate()
                    for df in dfs
                ]
            else:
                # Use smallest interval in data
                min_intervals = [
                    df.index.to_series().diff().min()
                    for df in dfs
                ]
                freq = max(min_intervals)
                aligned_dfs = [
                    df.loc[start_time:end_time].resample(freq).mean().interpolate()
                    for df in dfs
                ]
            
            # Convert to numpy arrays
            return [df["value"].to_numpy() for df in aligned_dfs]
            
        except Exception as e:
            self.logger.error(f"Time series alignment failed: {e}")
            raise DiscoMusicaError(f"Time series alignment failed: {e}")
            
    def _calculate_correlation_matrix(
        self,
        aligned_metrics: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate correlation matrix between time series.
        
        Args:
            aligned_metrics: List of aligned time series.
            
        Returns:
            Correlation matrix.
        """
        try:
            n_metrics = len(aligned_metrics)
            corr_matrix = np.zeros((n_metrics, n_metrics))
            
            for i in range(n_metrics):
                for j in range(n_metrics):
                    # Calculate Pearson correlation
                    pearson_corr, _ = stats.pearsonr(
                        aligned_metrics[i],
                        aligned_metrics[j]
                    )
                    corr_matrix[i, j] = pearson_corr
                    
            return corr_matrix
            
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            raise DiscoMusicaError(f"Correlation calculation failed: {e}")
            
    def _calculate_similarity_scores(
        self,
        aligned_metrics: List[np.ndarray]
    ) -> List[float]:
        """Calculate similarity scores between time series.
        
        Args:
            aligned_metrics: List of aligned time series.
            
        Returns:
            List of similarity scores.
        """
        try:
            n_metrics = len(aligned_metrics)
            similarity_scores = []
            
            # Calculate DTW distance between each pair
            for i in range(n_metrics):
                for j in range(i + 1, n_metrics):
                    # Normalize sequences
                    seq1 = (aligned_metrics[i] - np.mean(aligned_metrics[i])) / np.std(aligned_metrics[i])
                    seq2 = (aligned_metrics[j] - np.mean(aligned_metrics[j])) / np.std(aligned_metrics[j])
                    
                    # Calculate DTW distance
                    distance = dtw.distance(seq1, seq2)
                    
                    # Convert distance to similarity score (0 to 1)
                    similarity = 1.0 / (1.0 + distance)
                    similarity_scores.append(similarity)
                    
            return similarity_scores
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            raise DiscoMusicaError(f"Similarity calculation failed: {e}")
            
    def _analyze_comovement(
        self,
        aligned_metrics: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze co-movement patterns between time series.
        
        Args:
            aligned_metrics: List of aligned time series.
            
        Returns:
            Co-movement analysis results.
        """
        try:
            n_metrics = len(aligned_metrics)
            results = {
                "lead_lag_relationships": [],
                "synchronization": [],
                "granger_causality": []
            }
            
            for i in range(n_metrics):
                for j in range(i + 1, n_metrics):
                    # Calculate cross-correlation
                    cross_corr = signal.correlate(
                        aligned_metrics[i],
                        aligned_metrics[j],
                        mode="full"
                    )
                    lags = np.arange(-(len(aligned_metrics[i])-1), len(aligned_metrics[j]))
                    max_lag = lags[np.argmax(np.abs(cross_corr))]
                    
                    # Determine lead/lag relationship
                    relationship = {
                        "series_1": i,
                        "series_2": j,
                        "max_lag": int(max_lag),
                        "correlation_at_lag": float(cross_corr[np.argmax(np.abs(cross_corr))])
                    }
                    results["lead_lag_relationships"].append(relationship)
                    
                    # Calculate synchronization
                    sync_score = np.corrcoef(
                        np.diff(aligned_metrics[i]),
                        np.diff(aligned_metrics[j])
                    )[0, 1]
                    results["synchronization"].append({
                        "series_1": i,
                        "series_2": j,
                        "sync_score": float(sync_score)
                    })
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Co-movement analysis failed: {e}")
            raise DiscoMusicaError(f"Co-movement analysis failed: {e}")
            
    async def _detect_seasonal_patterns(
        self,
        sequence: torch.Tensor,
        window_size: int
    ) -> Dict[str, Any]:
        """Detect seasonal patterns in time series.
        
        Args:
            sequence: Input sequence.
            window_size: Analysis window size.
            
        Returns:
            Seasonal pattern analysis results.
        """
        try:
            # Convert to numpy array
            data = sequence.numpy()
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                data,
                period=window_size,
                extrapolate_trend=True
            )
            
            # Calculate seasonal strength
            seasonal_strength = 1.0 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
            
            # Detect significant seasons
            seasonal_pattern = decomposition.seasonal[:window_size]
            peaks = signal.find_peaks(seasonal_pattern)[0]
            troughs = signal.find_peaks(-seasonal_pattern)[0]
            
            return {
                "seasonal_strength": float(seasonal_strength),
                "seasonal_pattern": seasonal_pattern.tolist(),
                "peaks": peaks.tolist(),
                "troughs": troughs.tolist(),
                "trend": decomposition.trend.tolist(),
                "residual": decomposition.resid.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Seasonal pattern detection failed: {e}")
            raise DiscoMusicaError(f"Seasonal pattern detection failed: {e}")
            
    async def _detect_cyclic_patterns(
        self,
        sequence: torch.Tensor,
        window_size: int
    ) -> Dict[str, Any]:
        """Detect cyclic patterns in time series.
        
        Args:
            sequence: Input sequence.
            window_size: Analysis window size.
            
        Returns:
            Cyclic pattern analysis results.
        """
        try:
            # Convert to numpy array
            data = sequence.numpy()
            
            # Perform FFT
            n = len(data)
            fft_values = fft(data)
            freqs = fftfreq(n)
            
            # Get positive frequencies
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_values = np.abs(fft_values[pos_mask])
            
            # Find dominant frequencies
            peak_indices = signal.find_peaks(fft_values)[0]
            dominant_freqs = freqs[peak_indices]
            dominant_amplitudes = fft_values[peak_indices]
            
            # Sort by amplitude
            sort_idx = np.argsort(dominant_amplitudes)[::-1]
            dominant_freqs = dominant_freqs[sort_idx]
            dominant_amplitudes = dominant_amplitudes[sort_idx]
            
            # Calculate autocorrelation
            acf_values = acf(data, nlags=window_size)
            pacf_values = pacf(data, nlags=window_size)
            
            return {
                "dominant_frequencies": dominant_freqs[:5].tolist(),
                "dominant_amplitudes": dominant_amplitudes[:5].tolist(),
                "autocorrelation": acf_values.tolist(),
                "partial_autocorrelation": pacf_values.tolist(),
                "periodogram": {
                    "frequencies": freqs.tolist(),
                    "amplitudes": fft_values.tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Cyclic pattern detection failed: {e}")
            raise DiscoMusicaError(f"Cyclic pattern detection failed: {e}")
            
    async def _detect_trend_patterns(
        self,
        sequence: torch.Tensor,
        window_size: int
    ) -> Dict[str, Any]:
        """Detect trend patterns in time series.
        
        Args:
            sequence: Input sequence.
            window_size: Analysis window size.
            
        Returns:
            Trend pattern analysis results.
        """
        try:
            # Convert to numpy array
            data = sequence.numpy()
            
            # Calculate moving averages
            ma_short = pd.Series(data).rolling(window=window_size).mean().to_numpy()
            ma_long = pd.Series(data).rolling(window=window_size*2).mean().to_numpy()
            
            # Calculate trend direction changes
            trend_changes = np.diff(np.sign(np.diff(ma_short)))
            change_points = np.where(trend_changes != 0)[0] + 1
            
            # Fit polynomial trend
            x = np.arange(len(data))
            poly_coeffs = np.polyfit(x, data, deg=3)
            poly_trend = np.polyval(poly_coeffs, x)
            
            # Calculate trend strength
            trend_strength = 1.0 - np.var(data - poly_trend) / np.var(data)
            
            # Detect local extrema
            peaks = signal.find_peaks(data)[0]
            troughs = signal.find_peaks(-data)[0]
            
            return {
                "trend_strength": float(trend_strength),
                "trend_coefficients": poly_coeffs.tolist(),
                "trend_values": poly_trend.tolist(),
                "moving_averages": {
                    "short_term": ma_short.tolist(),
                    "long_term": ma_long.tolist()
                },
                "change_points": change_points.tolist(),
                "local_extrema": {
                    "peaks": peaks.tolist(),
                    "troughs": troughs.tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Trend pattern detection failed: {e}")
            raise DiscoMusicaError(f"Trend pattern detection failed: {e}") 