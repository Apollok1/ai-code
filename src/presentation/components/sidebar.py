"""
Sidebar component - Configuration and settings.
"""

import streamlit as st
from domain.models.config import AppConfig, ExtractionConfig
from presentation.state.session_manager import SessionManager


def render_sidebar(session: SessionManager, app_config: AppConfig) -> ExtractionConfig:
    """
    Render configuration sidebar.

    Args:
        session: Session manager
        app_config: Application configuration

    Returns:
        ExtractionConfig built from user selections
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # === OCR Settings ===
        st.subheader("📝 OCR")
        ocr_language = st.text_input(
            "OCR Language",
            value="pol+eng",
            disabled=session.is_converting()
        )
        ocr_dpi = st.slider(
            "OCR DPI",
            min_value=72,
            max_value=600,
            value=150,
            disabled=session.is_converting()
        )
        max_pages = st.slider(
            "Max Pages to Process",
            min_value=5,
            max_value=100,
            value=20,
            disabled=session.is_converting()
        )

        # === Vision Settings ===
        st.subheader("👁️ Vision")
        use_vision = st.checkbox(
            "Enable Vision Models",
            value=True,
            disabled=session.is_converting()
        )

        if use_vision:
            vision_model = st.text_input(
                "Vision Model",
                value="qwen2.5vl:7b",
                disabled=session.is_converting()
            )

            vision_mode = st.selectbox(
                "Image Processing Mode",
                options=["ocr", "transcribe", "describe", "ocr_plus_desc"],
                index=2,  # describe
                disabled=session.is_converting()
            )
        else:
            vision_model = "qwen2.5vl:7b"
            vision_mode = "ocr"

        # === Audio Settings ===
        st.subheader("🎤 Audio")
        enable_diarization = st.checkbox(
            "Enable Speaker Diarization",
            value=True,
            disabled=session.is_converting()
        )

        enable_summarization = st.checkbox(
            "Enable Meeting Summaries",
            value=True,
            disabled=session.is_converting()
        )

        if enable_summarization:
            summary_model = st.text_input(
                "Summary Model",
                value="qwen2.5:7b",
                disabled=session.is_converting()
            )
            chunk_size = st.slider(
                "Summary Chunk Size",
                min_value=2000,
                max_value=10000,
                value=6000,
                disabled=session.is_converting()
            )
        else:
            summary_model = "qwen2.5:7b"
            chunk_size = 6000

        # === Performance ===
        st.subheader("⚡ Performance")
        max_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=8,
            value=app_config.max_workers,
            help="More workers = faster processing (uses more resources)",
            disabled=session.is_converting()
        )

        # === Info ===
        st.divider()
        st.caption(f"Ollama: {app_config.ollama_url}")
        st.caption(f"Workers: {max_workers}")

    # Build ExtractionConfig
    config = ExtractionConfig(
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        min_text_length_for_ocr=100,
        max_pages=max_pages,
        use_vision=use_vision,
        vision_model=vision_model,
        vision_mode=vision_mode,
        enable_diarization=enable_diarization,
        enable_summarization=enable_summarization,
        summary_model=summary_model,
        chunk_size=chunk_size
    )

    # Save to session for later use
    session.save_config(config)

    return config
