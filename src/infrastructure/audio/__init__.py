"""Audio processing clients"""

from .whisper_client import WhisperASRClient
from .pyannote_client import PyannoteClient

__all__ = ["WhisperASRClient", "PyannoteClient"]
