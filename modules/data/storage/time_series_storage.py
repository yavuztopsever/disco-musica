"""Time series storage service for handling time-series data."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource

logger = logging.getLogger(__name__)

class TimeSeriesPoint(BaseModel):
    """Model for a time series data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]

class TimeSeriesMetric(BaseModel):
    """Model for a time series metric."""
    metric_id: str
    name: str
    description: str
    unit: str
    points: List[TimeSeriesPoint]
    metadata: Dict[str, Any]

class TimeSeriesStorage:
    """Service for managing time series data."""
    
    def __init__(self):
        """Initialize the time series storage."""
        self.metrics: Dict[str, TimeSeriesMetric] = {}
        self.logger = logging.getLogger(__name__)
        
    async def create_metric(
        self,
        metric_id: str,
        name: str,
        description: str,
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new time series metric.
        
        Args:
            metric_id: Unique identifier for the metric.
            name: Display name for the metric.
            description: Description of what the metric measures.
            unit: Unit of measurement.
            metadata: Optional metadata.
            
        Returns:
            ID of the created metric.
        """
        if metric_id in self.metrics:
            raise ValueError(f"Metric {metric_id} already exists")
            
        metric = TimeSeriesMetric(
            metric_id=metric_id,
            name=name,
            description=description,
            unit=unit,
            points=[],
            metadata=metadata or {}
        )
        
        self.metrics[metric_id] = metric
        return metric_id
        
    async def add_point(
        self,
        metric_id: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a data point to a metric.
        
        Args:
            metric_id: ID of the metric.
            value: Value of the data point.
            timestamp: Optional timestamp (defaults to current time).
            metadata: Optional metadata.
        """
        if metric_id not in self.metrics:
            raise ValueError(f"Metric {metric_id} not found")
            
        point = TimeSeriesPoint(
            timestamp=timestamp or datetime.utcnow(),
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics[metric_id].points.append(point)
        
    async def get_metric(
        self,
        metric_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[TimeSeriesMetric]:
        """Get a metric and its data points within a time range.
        
        Args:
            metric_id: ID of the metric.
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            
        Returns:
            Metric with filtered points if found, None otherwise.
        """
        if metric_id not in self.metrics:
            return None
            
        metric = self.metrics[metric_id]
        
        # Filter points by time range if specified
        if start_time or end_time:
            filtered_points = []
            for point in metric.points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_points.append(point)
                
            metric.points = filtered_points
            
        return metric
        
    async def get_aggregated_metric(
        self,
        metric_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        aggregation: str = "mean"
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated metric data within a time range.
        
        Args:
            metric_id: ID of the metric.
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            interval: Optional time interval for aggregation.
            aggregation: Aggregation function ("mean", "min", "max", "sum").
            
        Returns:
            Dictionary with aggregated data if found, None otherwise.
        """
        metric = await self.get_metric(metric_id, start_time, end_time)
        if not metric:
            return None
            
        if not interval:
            # Return simple statistics
            values = [point.value for point in metric.points]
            if not values:
                return None
                
            stats = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values))
            }
            
            return {
                "metric_id": metric_id,
                "name": metric.name,
                "unit": metric.unit,
                "start_time": start_time,
                "end_time": end_time,
                "statistics": stats
            }
            
        # Aggregate by interval
        if aggregation not in ["mean", "min", "max", "sum"]:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
            
        # Sort points by timestamp
        sorted_points = sorted(metric.points, key=lambda x: x.timestamp)
        
        # Group points by interval
        intervals = []
        current_interval = []
        current_interval_start = sorted_points[0].timestamp if sorted_points else None
        
        for point in sorted_points:
            if point.timestamp - current_interval_start >= interval:
                if current_interval:
                    intervals.append(current_interval)
                current_interval = [point]
                current_interval_start = point.timestamp
            else:
                current_interval.append(point)
                
        if current_interval:
            intervals.append(current_interval)
            
        # Calculate aggregated values
        aggregated_values = []
        for interval_points in intervals:
            values = [point.value for point in interval_points]
            if aggregation == "mean":
                value = float(np.mean(values))
            elif aggregation == "min":
                value = float(np.min(values))
            elif aggregation == "max":
                value = float(np.max(values))
            else:  # sum
                value = float(np.sum(values))
                
            aggregated_values.append({
                "timestamp": interval_points[0].timestamp,
                "value": value,
                "count": len(values)
            })
            
        return {
            "metric_id": metric_id,
            "name": metric.name,
            "unit": metric.unit,
            "interval": str(interval),
            "aggregation": aggregation,
            "data": aggregated_values
        }
        
    async def delete_metric(self, metric_id: str) -> bool:
        """Delete a metric and its data points.
        
        Args:
            metric_id: ID of the metric.
            
        Returns:
            True if deleted, False if not found.
        """
        if metric_id in self.metrics:
            del self.metrics[metric_id]
            return True
        return False
        
    async def list_metrics(
        self,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List all metrics, optionally filtered by metadata.
        
        Args:
            metadata_filter: Optional metadata filter.
            
        Returns:
            List of metric summaries.
        """
        metrics = []
        for metric_id, metric in self.metrics.items():
            # Apply metadata filter if provided
            if metadata_filter:
                if not all(metric.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
                    
            metrics.append({
                "metric_id": metric_id,
                "name": metric.name,
                "description": metric.description,
                "unit": metric.unit,
                "point_count": len(metric.points),
                "metadata": metric.metadata
            })
            
        return metrics 