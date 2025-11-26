"""
Document Converter Pro v2.0 - Main Application

Clean, maintainable Streamlit app using the new architecture.
"""

import streamlit as st
import logging

# Domain & Config
from domain.models.config import AppConfig

# Infrastructure
from infrastructure.factory import create_pipeline

# Presentation
from presentation.state.session_manager import SessionManager
from presentation.components import (
    render_sidebar,
    render_file_uploader,
    render_results,
    ProgressTracker
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Document Converter Pro v2.0",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# === INITIALIZATION ===

@st.cache_resource
def init_pipeline():
    """
    Initialize extraction pipeline (cached across reruns).

    Returns:
        Configured extraction pipeline
    """
    config = AppConfig.from_env()
    logger.info("Initializing pipeline...")

    pipeline = create_pipeline(
        config,
        vision_enabled=True,
        audio_diarization_enabled=True
    )

    logger.info("Pipeline ready!")
    return pipeline, config


# Initialize
pipeline, app_config = init_pipeline()
session = SessionManager()

# === MAIN UI ===

def main():
    """Main application UI"""

    # Title
    st.title("📄 Document Converter Pro v2.0")
    st.caption(
        "🚀 High-performance document processing with AI | "
        "Offline-first | Parallel processing"
    )

    # Sidebar - configuration
    extraction_config = render_sidebar(session, app_config)

    # File uploader
    uploaded_files = render_file_uploader(session)

    if not uploaded_files:
        # Show stats when no files
        st.info("👋 Welcome! Upload documents to get started.")

        with st.expander("ℹ️ Pipeline Information"):
            stats = pipeline.get_stats()

            st.metric("Extractors", stats['extractors_count'])
            st.metric("Parallel Workers", stats['max_workers'])

            st.subheader("Supported Formats")
            st.write(", ".join(stats['supported_extensions']))

            st.subheader("Extractors")
            for ext_info in stats['extractors']:
                st.text(f"• {ext_info['name']}: {ext_info['extensions']}")

        return

    # === CONVERSION BUTTON ===
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button(
            "🚀 Convert All Files",
            type="primary",
            disabled=session.is_converting(),
            use_container_width=True
        ):
            session.start_conversion()
            st.rerun()

    with col2:
        if session.is_converting():
            if st.button("🛑 Cancel", type="secondary", use_container_width=True):
                session.request_cancel()
                st.rerun()

    # === CONVERSION PROCESS ===
    if session.is_converting():
        process_files(uploaded_files, extraction_config)

    # === RESULTS ===
    if session.has_results():
        render_results(session)


def process_files(files, config):
    """
    Process uploaded files with progress tracking.

    Args:
        files: List of uploaded files
        config: Extraction configuration
    """
    st.divider()
    st.subheader("🔄 Processing...")

    # Progress tracker
    tracker = ProgressTracker()
    tracker.start(len(files))

    # Progress callback
    def on_progress(current, total, file_name):
        tracker.update(current, f"Processing {file_name}...")

        # Check for cancellation
        if session.is_cancel_requested():
            raise InterruptedError("User cancelled")

    try:
        # Prepare file tuples
        file_tuples = [(f, f.name) for f in files]

        # Process in parallel!
        logger.info(f"Starting batch processing: {len(files)} files")
        results = pipeline.process_batch(
            file_tuples,
            config,
            progress_callback=on_progress
        )

        # Save results to session
        for result in results:
            session.add_result(result)

        # Complete
        tracker.complete(f"✅ Processed {len(results)} files!")

        # Show summary
        successful = sum(1 for r in results if r.is_successful())
        failed = len(results) - successful

        if failed > 0:
            st.warning(f"⚠️ {successful} successful, {failed} failed")
        else:
            st.success(f"🎉 All {successful} files processed successfully!")

    except InterruptedError:
        tracker.error("⛔ Cancelled by user")
        st.warning("Processing cancelled. Partial results saved.")

    except Exception as e:
        tracker.error(f"❌ Error: {e}")
        st.error(f"Processing failed: {e}")
        logger.exception("Processing error")

    finally:
        session.end_conversion()
        st.rerun()


# === FOOTER ===
def render_footer():
    """Render footer with info"""
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("📦 Document Converter Pro v2.0")

    with col2:
        st.caption(f"⚡ {app_config.max_workers} parallel workers")

    with col3:
        st.caption("🔒 Offline-first design")


# === ENTRYPOINT ===
if __name__ == "__main__":
    try:
        main()
        render_footer()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.exception("Application error")

        if st.button("🔄 Restart"):
            st.rerun()
