"""
Extraction Pipeline - Main orchestrator for document processing.

Supports:
- Single file processing
- Batch processing with parallelization
- Progress tracking
- Error handling and recovery
"""

import logging
from typing import BinaryIO, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from domain.interfaces.extractor import Extractor
from domain.models.document import ExtractionResult
from domain.models.config import ExtractionConfig
from domain.exceptions import UnsupportedFormatError

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Main orchestrator for document extraction.

    Responsibilities:
    - Route files to appropriate extractors
    - Parallel processing for batch operations
    - Progress tracking
    - Error handling and recovery
    """

    def __init__(
        self,
        extractors: list[Extractor],
        max_workers: int = 4
    ):
        """
        Initialize extraction pipeline.

        Args:
            extractors: List of available extractors
            max_workers: Maximum parallel workers for batch processing
        """
        self.extractors = extractors
        self.max_workers = max_workers
        logger.info(
            f"ExtractionPipeline initialized: "
            f"{len(extractors)} extractors, {max_workers} workers"
        )

    def process_single(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Process a single file.

        Args:
            file: File object
            file_name: Original file name
            config: Extraction configuration

        Returns:
            ExtractionResult

        Raises:
            UnsupportedFormatError: If no extractor supports the file
            ExtractionError: If extraction fails
        """
        # Find appropriate extractor
        extractor = self._find_extractor(file_name)

        if not extractor:
            supported = self._list_supported_extensions()
            raise UnsupportedFormatError(
                file_name,
                supported_formats=supported
            )

        # Extract
        logger.info(f"Processing {file_name} with {extractor.name}")
        result = extractor.extract(file, file_name, config)

        logger.info(
            f"✓ Completed {file_name}: "
            f"{result.metadata.pages_count} pages, "
            f"{result.total_words} words, "
            f"{result.metadata.processing_time_seconds:.2f}s"
        )

        return result

    def process_batch(
        self,
        files: list[tuple[BinaryIO, str]],
        config: ExtractionConfig,
        progress_callback: Callable[[int, int, str], None] | None = None
    ) -> list[ExtractionResult]:
        """
        Process multiple files in parallel.

        Args:
            files: List of (file, file_name) tuples
            config: Extraction configuration
            progress_callback: Optional callback(current, total, file_name)

        Returns:
            List of extraction results (successful and failed)
        """
        total = len(files)
        results = []

        logger.info(f"Starting batch processing: {total} files, {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._process_with_error_handling,
                    file,
                    name,
                    config
                ): name
                for file, name in files
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                file_name = future_to_file[future]

                try:
                    result = future.result()
                    results.append(result)

                    if result.is_successful():
                        logger.info(f"✓ ({i}/{total}) {file_name}")
                    else:
                        logger.warning(f"⚠ ({i}/{total}) {file_name} - has errors")

                except Exception as e:
                    logger.error(f"✗ ({i}/{total}) {file_name} - {e}")
                    # Continue processing other files

                # Progress callback
                if progress_callback:
                    progress_callback(i, total, file_name)

        logger.info(
            f"Batch processing complete: "
            f"{len(results)}/{total} processed, "
            f"{sum(1 for r in results if r.is_successful())} successful"
        )

        return results

    def _process_with_error_handling(
        self,
        file: BinaryIO,
        file_name: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Process file with error handling.

        Returns result even if errors occurred (with error metadata).
        """
        try:
            return self.process_single(file, file_name, config)
        except Exception as e:
            # Return failed result instead of raising
            logger.exception(f"Error processing {file_name}")

            from domain.models.document import (
                ExtractionMetadata,
                DocumentType,
                Page
            )

            metadata = ExtractionMetadata(
                document_type=DocumentType.from_filename(file_name),
                pages_count=0,
                extraction_method="failed",
                processing_time_seconds=0.0,
                file_size_bytes=0
            )
            metadata.add_error(str(e))

            return ExtractionResult(
                file_name=file_name,
                pages=[Page(number=1, text=f"[ERROR: {e}]")],
                metadata=metadata
            )

    def _find_extractor(self, file_name: str) -> Extractor | None:
        """Find extractor that can handle the file"""
        for extractor in self.extractors:
            if extractor.can_handle(file_name):
                return extractor
        return None

    def _list_supported_extensions(self) -> list[str]:
        """List all supported file extensions"""
        extensions = set()
        for extractor in self.extractors:
            extensions.update(extractor.supported_extensions)
        return sorted(extensions)

    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        return {
            "extractors_count": len(self.extractors),
            "max_workers": self.max_workers,
            "supported_extensions": self._list_supported_extensions(),
            "extractors": [
                {
                    "name": ext.name,
                    "extensions": list(ext.supported_extensions)
                }
                for ext in self.extractors
            ]
        }
