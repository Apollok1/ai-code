"""
CAD Estimator Pro - Domain Exceptions

Custom exception hierarchy for structured error handling.
"""
from typing import Any


class CADException(Exception):
    """Base exception for all CAD Estimator errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ValidationError(CADException):
    """Validation error (e.g., invalid input data)."""

    def __init__(self, message: str, field_name: str | None = None, **kwargs):
        super().__init__(message, {"field_name": field_name, **kwargs})


class DatabaseError(CADException):
    """Database operation error."""

    def __init__(self, message: str, query: str | None = None, **kwargs):
        super().__init__(message, {"query": query, **kwargs})


class ConnectionError(DatabaseError):
    """Database connection error."""

    pass


class QueryError(DatabaseError):
    """Database query execution error."""

    pass


class ParsingError(CADException):
    """File parsing error (Excel, PDF, JSON)."""

    def __init__(self, message: str, file_name: str | None = None, **kwargs):
        super().__init__(message, {"file_name": file_name, **kwargs})


class ExcelParsingError(ParsingError):
    """Excel file parsing error."""

    def __init__(self, message: str, file_name: str | None = None, row: int | None = None, **kwargs):
        super().__init__(message, file_name, row=row, **kwargs)


class PDFParsingError(ParsingError):
    """PDF file parsing error."""

    def __init__(self, message: str, file_name: str | None = None, page: int | None = None, **kwargs):
        super().__init__(message, file_name, page=page, **kwargs)


class JSONParsingError(ParsingError):
    """JSON parsing error."""

    pass


class AIGenerationError(CADException):
    """AI response generation error."""

    def __init__(self, message: str, model: str | None = None, prompt_length: int | None = None, **kwargs):
        super().__init__(message, {"model": model, "prompt_length": prompt_length, **kwargs})


class AIResponseParsingError(AIGenerationError):
    """AI response parsing error (e.g., invalid JSON)."""

    def __init__(self, message: str, response_text: str | None = None, **kwargs):
        super().__init__(message, response_text=response_text[:200] if response_text else None, **kwargs)


class EmbeddingError(CADException):
    """Embedding generation error."""

    def __init__(self, message: str, text: str | None = None, model: str | None = None, **kwargs):
        super().__init__(message, {"text": text[:100] if text else None, "model": model, **kwargs})


class PatternLearningError(CADException):
    """Pattern learning error."""

    def __init__(self, message: str, component_name: str | None = None, **kwargs):
        super().__init__(message, {"component_name": component_name, **kwargs})


class StorageError(CADException):
    """Project/pattern storage error."""

    def __init__(self, message: str, entity_type: str | None = None, entity_id: int | None = None, **kwargs):
        super().__init__(message, {"entity_type": entity_type, "entity_id": entity_id, **kwargs})


class NotFoundError(StorageError):
    """Entity not found error."""

    pass


class ConfigurationError(CADException):
    """Configuration error."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        super().__init__(message, {"config_key": config_key, **kwargs})
