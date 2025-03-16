"""Logic Pro plugin interface for handling Logic Pro plugin communication."""

import logging
from typing import Dict, Any, List, Optional, Union
import json
import socket
import threading
from pydantic import BaseModel

from ...core.exceptions import DiscoMusicaError
from ...core.resources import BaseResource

logger = logging.getLogger(__name__)

class LogicProMessage(BaseModel):
    """Model for Logic Pro plugin messages."""
    message_type: str
    data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = {}

class LogicProInterface:
    """Interface for handling Logic Pro plugin communication."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        timeout: float = 5.0
    ):
        """Initialize the Logic Pro interface.
        
        Args:
            host: Host address.
            port: Port number.
            timeout: Socket timeout in seconds.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> None:
        """Connect to Logic Pro plugin."""
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # Connect
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start message listener
            self._start_listener()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Logic Pro: {e}")
            raise DiscoMusicaError(f"Failed to connect to Logic Pro: {e}")
            
    def disconnect(self) -> None:
        """Disconnect from Logic Pro plugin."""
        try:
            if self.socket is not None:
                self.socket.close()
                self.socket = None
                self.connected = False
                
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Logic Pro: {e}")
            raise DiscoMusicaError(f"Failed to disconnect from Logic Pro: {e}")
            
    def _start_listener(self) -> None:
        """Start message listener thread."""
        def listener():
            while self.connected:
                try:
                    # Receive message
                    data = self.socket.recv(4096)
                    if not data:
                        break
                        
                    # Parse message
                    message = json.loads(data.decode())
                    self._handle_message(message)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in message listener: {e}")
                    break
                    
        # Start listener thread
        self.listener_thread = threading.Thread(target=listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle received message.
        
        Args:
            message: Received message.
        """
        try:
            # Parse message
            logic_pro_message = LogicProMessage(**message)
            
            # Handle message based on type
            if logic_pro_message.message_type == "audio_data":
                self._handle_audio_data(logic_pro_message.data)
            elif logic_pro_message.message_type == "control_data":
                self._handle_control_data(logic_pro_message.data)
            elif logic_pro_message.message_type == "status_update":
                self._handle_status_update(logic_pro_message.data)
            else:
                self.logger.warning(f"Unknown message type: {logic_pro_message.message_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle message: {e}")
            raise DiscoMusicaError(f"Failed to handle message: {e}")
            
    def _handle_audio_data(self, data: Dict[str, Any]) -> None:
        """Handle audio data message.
        
        Args:
            data: Audio data.
        """
        try:
            # Process audio data
            # TODO: Implement audio data processing
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to handle audio data: {e}")
            raise DiscoMusicaError(f"Failed to handle audio data: {e}")
            
    def _handle_control_data(self, data: Dict[str, Any]) -> None:
        """Handle control data message.
        
        Args:
            data: Control data.
        """
        try:
            # Process control data
            # TODO: Implement control data processing
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to handle control data: {e}")
            raise DiscoMusicaError(f"Failed to handle control data: {e}")
            
    def _handle_status_update(self, data: Dict[str, Any]) -> None:
        """Handle status update message.
        
        Args:
            data: Status data.
        """
        try:
            # Process status update
            # TODO: Implement status update processing
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to handle status update: {e}")
            raise DiscoMusicaError(f"Failed to handle status update: {e}")
            
    async def send_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send message to Logic Pro plugin.
        
        Args:
            message_type: Type of message.
            data: Message data.
            metadata: Optional metadata.
        """
        try:
            if not self.connected:
                raise DiscoMusicaError("Not connected to Logic Pro")
                
            # Create message
            message = LogicProMessage(
                message_type=message_type,
                data=data,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Send message
            self.socket.send(json.dumps(message.dict()).encode())
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise DiscoMusicaError(f"Failed to send message: {e}")
            
    async def send_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        channels: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send audio data to Logic Pro plugin.
        
        Args:
            audio_data: Audio data array.
            sample_rate: Sample rate.
            channels: Number of channels.
            metadata: Optional metadata.
        """
        try:
            # Prepare audio data
            data = {
                "audio_data": audio_data.tolist(),
                "sample_rate": sample_rate,
                "channels": channels
            }
            
            # Send message
            await self.send_message(
                message_type="audio_data",
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send audio data: {e}")
            raise DiscoMusicaError(f"Failed to send audio data: {e}")
            
    async def send_control_data(
        self,
        control_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send control data to Logic Pro plugin.
        
        Args:
            control_data: Control data.
            metadata: Optional metadata.
        """
        try:
            # Send message
            await self.send_message(
                message_type="control_data",
                data=control_data,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send control data: {e}")
            raise DiscoMusicaError(f"Failed to send control data: {e}")
            
    async def send_status_update(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send status update to Logic Pro plugin.
        
        Args:
            status: Status message.
            details: Optional status details.
            metadata: Optional metadata.
        """
        try:
            # Prepare status data
            data = {
                "status": status,
                "details": details or {}
            }
            
            # Send message
            await self.send_message(
                message_type="status_update",
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send status update: {e}")
            raise DiscoMusicaError(f"Failed to send status update: {e}")
            
    def is_connected(self) -> bool:
        """Check if connected to Logic Pro plugin.
        
        Returns:
            True if connected, False otherwise.
        """
        return self.connected 