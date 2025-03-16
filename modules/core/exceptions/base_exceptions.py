"""Base exceptions for the Disco-Musica system."""
from typing import Optional


class DiscoMusicaError(Exception):
    """Base exception for all Disco-Musica errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Optional error code.
            details: Optional error details.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        
    def to_dict(self) -> dict:
        """Convert error to dictionary.
        
        Returns:
            Dictionary representation of error.
        """
        return {
            "message": self.message,
            "code": self.code,
            "details": self.details
        }


class ResourceNotFoundError(DiscoMusicaError):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self,
        message: str,
        code: str = "RESOURCE_NOT_FOUND",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class ModelNotFoundError(DiscoMusicaError):
    """Raised when a requested model is not found."""
    
    def __init__(
        self,
        message: str,
        code: str = "MODEL_NOT_FOUND",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class ProcessingError(DiscoMusicaError):
    """Raised when processing fails."""
    
    def __init__(
        self,
        message: str,
        code: str = "PROCESSING_ERROR",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class ValidationError(DiscoMusicaError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        code: str = "VALIDATION_ERROR",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class ConfigurationError(DiscoMusicaError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        code: str = "CONFIGURATION_ERROR",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class ResourceAccessError(DiscoMusicaError):
    """Raised when access to a resource is denied."""
    pass


class IntegrationError(DiscoMusicaError):
    """Raised when external integration fails."""
    pass


class AuthenticationError(DiscoMusicaError):
    """Error during authentication."""
    
    def __init__(
        self,
        message: str,
        code: str = "AUTHENTICATION_ERROR",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class AuthorizationError(DiscoMusicaError):
    """Error during authorization."""
    
    def __init__(
        self,
        message: str,
        code: str = "AUTHORIZATION_ERROR",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class RateLimitError(DiscoMusicaError):
    """Error when rate limit exceeded."""
    
    def __init__(
        self,
        message: str,
        code: str = "RATE_LIMIT_ERROR",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details)


class ServiceUnavailableError(DiscoMusicaError):
    """Error when service is unavailable."""
    
    def __init__(
        self,
        message: str,
        code: str = "SERVICE_UNAVAILABLE",
        details: Optional[dict] = None
    ):
        """Initialize the error.
        
        Args:
            message: Error message.
            code: Error code.
            details: Optional error details.
        """
        super().__init__(message, code, details) 