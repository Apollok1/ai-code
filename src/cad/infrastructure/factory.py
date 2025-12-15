"""
CAD Estimator Pro - Infrastructure Factory

Factory functions for dependency injection and easy setup.
"""
import logging
from typing import Any

from cad.domain.models.config import AppConfig
from cad.infrastructure.database.postgres_client import PostgresClient
from cad.infrastructure.ai.ollama_client import OllamaClient
from cad.infrastructure.parsers.excel_parser import CADExcelParser
from cad.infrastructure.parsers.pdf_parser import CADPDFParser
from cad.infrastructure.parsers.component_parser import CADComponentParser
from cad.infrastructure.multi_model import MultiModelOrchestrator

logger = logging.getLogger(__name__)


def create_database_client(config: AppConfig) -> PostgresClient:
    """
    Create PostgreSQL database client.

    Args:
        config: Application configuration

    Returns:
        PostgresClient instance
    """
    return PostgresClient(config.database)


def create_ai_client(config: AppConfig) -> OllamaClient:
    """
    Create Ollama AI client.

    Args:
        config: Application configuration

    Returns:
        OllamaClient instance
    """
    return OllamaClient(config.ollama)


def create_excel_parser(config: AppConfig) -> CADExcelParser:
    """
    Create Excel parser.

    Args:
        config: Application configuration

    Returns:
        CADExcelParser instance
    """
    return CADExcelParser()


def create_pdf_parser(config: AppConfig) -> CADPDFParser:
    """
    Create PDF parser.

    Args:
        config: Application configuration

    Returns:
        CADPDFParser instance
    """
    return CADPDFParser()


def create_component_parser(config: AppConfig) -> CADComponentParser:
    """
    Create component parser.

    Args:
        config: Application configuration

    Returns:
        CADComponentParser instance
    """
    return CADComponentParser()


def create_multi_model_orchestrator(config: AppConfig) -> MultiModelOrchestrator:
    """
    Create multi-model orchestrator.

    Args:
        config: Application configuration

    Returns:
        MultiModelOrchestrator instance
    """
    ai_client = create_ai_client(config)
    db_client = create_database_client(config)
    return MultiModelOrchestrator(ai_client, db_client, config.multi_model)


def quick_setup(
    ollama_url: str = "http://localhost:11434",
    db_host: str = "localhost"
) -> dict[str, Any]:
    """
    Quick setup with default configuration.

    Args:
        ollama_url: Ollama service URL
        db_host: Database host

    Returns:
        Dict with initialized components:
        - config: AppConfig
        - db: PostgresClient
        - ai: OllamaClient
        - excel_parser: CADExcelParser
        - pdf_parser: CADPDFParser
        - component_parser: CADComponentParser
    """
    from cad.domain.models.config import AppConfig, DatabaseConfig, OllamaConfig

    config = AppConfig(
        database=DatabaseConfig(host=db_host),
        ollama=OllamaConfig(url=ollama_url)
    )

    db = create_database_client(config)
    ai = create_ai_client(config)

    components = {
        'config': config,
        'db': db,
        'ai': ai,
        'excel_parser': create_excel_parser(config),
        'pdf_parser': create_pdf_parser(config),
        'component_parser': create_component_parser(config),
        'multi_model': MultiModelOrchestrator(ai, db, config.multi_model)
    }

    logger.info("✅ Quick setup complete (multi-model pipeline ready)")
    return components
