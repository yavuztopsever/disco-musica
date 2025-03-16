"""Base resource models for the system."""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Resource(BaseModel):
    """Base model for all resources in the system."""
    
    resource_id: str = Field(default_factory=lambda: str(uuid4()))
    resource_type: str
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    modification_timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    parent_resources: List[str] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)
    access_control: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow extra fields for resource-specific data


class ProjectResource(Resource):
    """Resource model for music projects."""
    
    basic_info: Dict[str, Any] = Field(default_factory=dict)
    musical_properties: Dict[str, Any] = Field(default_factory=dict)
    tracks: List[str] = Field(default_factory=list)
    analysis_results: List[str] = Field(default_factory=list)


class TrackResource(Resource):
    """Resource model for audio/midi tracks."""
    
    basic_info: Dict[str, Any] = Field(default_factory=dict)
    audio_properties: Optional[Dict[str, Any]] = None
    midi_properties: Optional[Dict[str, Any]] = None
    derived_features: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    content_reference: Dict[str, Any] = Field(default_factory=dict)


class ModelResource(Resource):
    """Resource model for AI models."""
    
    model_info: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    weights_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict) 