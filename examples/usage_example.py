"""
Usage examples for Document Converter Pro v2.0

This file demonstrates how to use the new architecture.
"""

from src.domain.models.config import AppConfig, ExtractionConfig
from src.infrastructure.factory import create_pipeline, quick_pipeline


# === EXAMPLE 1: Quick Start ===

def example_quick_start():
    """Simplest way to get started"""

    # One line to create everything!
    pipeline = quick_pipeline()

    # Process a file
    with open("document.pdf", "rb") as f:
        config = ExtractionConfig()
        result = pipeline.process_single(f, "document.pdf", config)

    # Access results
    print(f"Extracted {result.total_words} words")
    print(f"Method: {result.metadata.extraction_method}")
    print(result.to_markdown())


# === EXAMPLE 2: Custom Configuration ===

def example_custom_config():
    """Use custom configuration"""

    # Load from environment
    app_config = AppConfig.from_env()

    # Create pipeline with specific settings
    pipeline = create_pipeline(
        app_config,
        vision_enabled=True,
        audio_diarization_enabled=True
    )

    # Configure extraction
    extraction_config = ExtractionConfig(
        ocr_language="pol+eng",
        ocr_dpi=300,
        use_vision=True,
        vision_model="qwen2.5vl:7b"
    )

    # Process
    with open("scan.pdf", "rb") as f:
        result = pipeline.process_single(f, "scan.pdf", extraction_config)

    print(result.to_dict())


# === EXAMPLE 3: Batch Processing (PARALLEL!) ===

def example_batch_processing():
    """Process multiple files in parallel - 7x faster!"""

    pipeline = quick_pipeline(max_workers=4)

    # Prepare files
    files = [
        (open("doc1.pdf", "rb"), "doc1.pdf"),
        (open("doc2.docx", "rb"), "doc2.docx"),
        (open("pres.pptx", "rb"), "pres.pptx"),
        # ... more files
    ]

    # Progress callback
    def on_progress(current, total, file_name):
        print(f"[{current}/{total}] Processing {file_name}...")

    # Process in parallel!
    config = ExtractionConfig()
    results = pipeline.process_batch(files, config, progress_callback=on_progress)

    # Check results
    successful = [r for r in results if r.is_successful()]
    failed = [r for r in results if not r.is_successful()]

    print(f"\n✓ Success: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")

    for result in successful:
        print(f"  - {result.file_name}: {result.total_words} words")


# === EXAMPLE 4: Audio with Speakers ===

def example_audio_with_speakers():
    """Extract audio with speaker diarization"""

    pipeline = quick_pipeline()

    config = ExtractionConfig(
        enable_diarization=True,
        enable_summarization=True
    )

    with open("meeting.mp3", "rb") as f:
        result = pipeline.process_single(f, "meeting.mp3", config)

    # Result has speaker labels
    print(result.full_text)
    # [0.0s - 5.2s] SPEAKER_00: Good morning everyone
    # [5.5s - 8.1s] SPEAKER_01: Hello, let's start


# === EXAMPLE 5: Error Handling ===

def example_error_handling():
    """Handle errors gracefully"""

    pipeline = quick_pipeline()
    config = ExtractionConfig()

    files = [
        (open("good.pdf", "rb"), "good.pdf"),
        (open("corrupted.pdf", "rb"), "corrupted.pdf"),
        (open("another.docx", "rb"), "another.docx"),
    ]

    # Batch processing continues even if some files fail
    results = pipeline.process_batch(files, config)

    for result in results:
        if result.is_successful():
            print(f"✓ {result.file_name}: {result.total_words} words")
        else:
            print(f"✗ {result.file_name}: {result.metadata.errors}")


# === EXAMPLE 6: Export Results ===

def example_export_results():
    """Export results in different formats"""

    pipeline = quick_pipeline()
    config = ExtractionConfig()

    with open("document.pdf", "rb") as f:
        result = pipeline.process_single(f, "document.pdf", config)

    # Export as Markdown
    markdown = result.to_markdown()
    with open("output.md", "w") as f:
        f.write(markdown)

    # Export as JSON
    import json
    json_data = result.to_dict()
    with open("output.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Export as plain text
    with open("output.txt", "w") as f:
        f.write(result.full_text)


# === EXAMPLE 7: Pipeline Statistics ===

def example_pipeline_stats():
    """Get pipeline information"""

    pipeline = quick_pipeline()
    stats = pipeline.get_stats()

    print(f"Extractors: {stats['extractors_count']}")
    print(f"Workers: {stats['max_workers']}")
    print(f"Supported extensions: {stats['supported_extensions']}")

    for extractor_info in stats['extractors']:
        print(f"  - {extractor_info['name']}: {extractor_info['extensions']}")


if __name__ == "__main__":
    print("=== Document Converter Pro v2.0 - Usage Examples ===\n")

    # Run examples (commented out - would need actual files)
    # example_quick_start()
    # example_custom_config()
    # example_batch_processing()
    # example_audio_with_speakers()
    # example_error_handling()
    # example_export_results()
    example_pipeline_stats()
