"""
Results display component.
"""

import streamlit as st
from datetime import datetime
from presentation.state.session_manager import SessionManager


def render_results(session: SessionManager):
    """
    Render extraction results.

    Args:
        session: Session manager
    """
    if not session.has_results():
        st.info("👆 Upload files and click Convert to get started")
        return

    st.success(
        f"✅ Processed: {session.state.stats.processed} files | "
        f"Pages: {session.state.stats.total_pages} | "
        f"Duration: {session.state.stats.duration_seconds:.1f}s"
        if session.state.stats.duration_seconds
        else f"✅ Processed: {session.state.stats.processed} files"
    )

    # === Download All Button ===
    col1, col2, col3 = st.columns(3)

    with col1:
        # Combined TXT
        combined_text = session.state.combined_text
        st.download_button(
            "⬇️ Download All (TXT)",
            data=combined_text.encode("utf-8"),
            file_name=f"converted_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            disabled=session.is_converting()
        )

    with col2:
        # Combined Markdown
        markdown_lines = []
        for result in session.get_results():
            markdown_lines.append(result.to_markdown())
            markdown_lines.append("\n---\n")

        st.download_button(
            "⬇️ Download All (MD)",
            data="\n".join(markdown_lines).encode("utf-8"),
            file_name=f"converted_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            disabled=session.is_converting()
        )

    with col3:
        # Combined JSON
        import json
        json_data = [r.to_dict() for r in session.get_results()]

        st.download_button(
            "⬇️ Download All (JSON)",
            data=json.dumps(json_data, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"converted_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            disabled=session.is_converting()
        )

    st.divider()

    # === Individual Results ===
    for result in session.get_results():
        with st.expander(
            f"📄 {result.file_name} "
            f"({result.metadata.pages_count} pages, "
            f"{result.total_words:,} words)"
        ):
            # Metadata
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Method", result.metadata.extraction_method)

            with col2:
                st.metric("Processing Time", f"{result.metadata.processing_time_seconds:.2f}s")

            with col3:
                status = "✓ Success" if result.is_successful() else "✗ Failed"
                st.metric("Status", status)

            # Errors/Warnings
            if result.metadata.errors:
                st.error("Errors:")
                for error in result.metadata.errors:
                    st.text(f"  - {error}")

            if result.metadata.warnings:
                st.warning("Warnings:")
                for warning in result.metadata.warnings:
                    st.text(f"  - {warning}")

            # Text Preview
            st.subheader("Text Preview")
            preview_text = result.full_text[:2000]
            if len(result.full_text) > 2000:
                preview_text += "\n\n[... truncated ...]"

            st.text_area(
                "Content",
                value=preview_text,
                height=300,
                key=f"preview_{result.file_name}",
                disabled=True
            )

            # Download individual
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "⬇️ Download TXT",
                    data=result.full_text.encode("utf-8"),
                    file_name=f"{result.file_name}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{result.file_name}"
                )

            with col2:
                st.download_button(
                    "⬇️ Download MD",
                    data=result.to_markdown().encode("utf-8"),
                    file_name=f"{result.file_name}.md",
                    mime="text/markdown",
                    key=f"dl_md_{result.file_name}"
                )

    # === Reset Button ===
    st.divider()
    if st.button("🔄 Reset Session", type="secondary"):
        session.reset()
        st.rerun()
