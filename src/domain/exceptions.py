"""
Custom exceptions for domain layer.

All business logic exceptions should inherit from DomainException.
"""


class DomainException(Exception):
    """Base exception for all domain-related errors"""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ExtractionError(DomainException):
    """Raised when document extraction fails"""

    def __init__(self, message: str, file_name: str | None = None, **kwargs):
        super().__init__(message, {"file_name": file_name, **kwargs})
        self.file_name = file_name


class UnsupportedFormatError(DomainException):
    """Raised when file format is not supported"""

    def __init__(self, file_name: str, supported_formats: list[str] | None = None):
        message = f"Unsupported file format: {file_name}"
        if supported_formats:
            message += f". Supported: {', '.join(supported_formats)}"
        super().__init__(message, {
            "file_name": file_name,
            "supported_formats": supported_formats or []
        })


class ConfigurationError(DomainException):
    """Raised when configuration is invalid"""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message, {"config_key": config_key})
        self.config_key = config_key


class ValidationError(DomainException):
    """Raised when data validation fails"""

    def __init__(self, message: str, field: str | None = None, value=None):
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class ServiceError(DomainException):
    """Raised when external service (Ollama, Whisper, etc.) fails"""

    def __init__(self, message: str, service_name: str, **kwargs):
        super().__init__(message, {"service_name": service_name, **kwargs})
        self.service_name = service_name


class OCRError(ServiceError):
    """Raised when OCR processing fails"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="Tesseract OCR", **kwargs)


class VisionError(ServiceError):
    """Raised when Vision model fails"""

    def __init__(self, message: str, model_name: str | None = None, **kwargs):
        super().__init__(
            message,
            service_name="Vision LLM",
            model_name=model_name,
            **kwargs
        )


class AudioProcessingError(ServiceError):
    """Raised when audio processing (Whisper/Pyannote) fails"""

    def __init__(self, message: str, service: str = "Audio", **kwargs):
        super().__init__(message, service_name=service, **kwargs)
