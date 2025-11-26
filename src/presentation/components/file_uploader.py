"""
File uploader component.
"""

import streamlit as st
from presentation.state.session_manager import SessionManager


def render_file_uploader(session: SessionManager):
    """
    Render file uploader widget.

    Args:
        session: Session manager

    Returns:
        List of uploaded files or None
    """
    uploaded_files = st.file_uploader(
        "📁 Upload Documents",
        type=[
            'pdf', 'docx', 'pptx', 'ppt',
            'jpg', 'jpeg', 'png',
            'txt', 'mp3', 'wav', 'm4a', 'ogg', 'flac',
            'eml', 'msg'
        ],
        accept_multiple_files=True,
        disabled=session.is_converting(),
        help="Supported: PDF, DOCX, PPTX, Images, Audio, Email"
    )

    if uploaded_files:
        # Show file info
        total_size = sum(getattr(f, 'size', 0) for f in uploaded_files)
        size_mb = total_size / (1024 * 1024)

        st.info(
            f"📊 {len(uploaded_files)} files selected "
            f"({size_mb:.1f} MB total)"
        )

        # Check if files changed
        if session.files_changed(uploaded_files):
            st.warning("⚠️ Files changed - previous results cleared")
            session.reset()

    return uploaded_files
