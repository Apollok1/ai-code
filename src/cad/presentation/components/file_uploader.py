"""
CAD Estimator Pro - File Uploader Component

File upload UI for Excel, PDF, JSON, and images.
"""
import streamlit as st
from typing import Any


def render_file_uploader() -> dict[str, Any]:
    """
    Render file uploader section.

    Returns:
        Dict with uploaded files:
        {
            'excel': UploadedFile | None,
            'pdfs': list[UploadedFile],
            'jsons': list[UploadedFile],
            'images': list[UploadedFile]
        }
    """
    st.subheader("📎 Pliki wejściowe")

    col1, col2 = st.columns(2)

    with col1:
        excel_file = st.file_uploader(
            "Excel (komponenty)",
            type=['xlsx', 'xls'],
            help="Plik Excel z komponentami CAD (opcjonalnie)"
        )

        pdf_files = st.file_uploader(
            "PDF (specyfikacje)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Specyfikacje techniczne, normy, rysunki PDF"
        )

    with col2:
        json_files = st.file_uploader(
            "JSON (doc-converter/AI)",
            type=['json'],
            accept_multiple_files=True,
            help="Wyniki z doc-converter lub innych źródeł AI"
        )

        image_files = st.file_uploader(
            "Zdjęcia/Rysunki",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Zdjęcia komponentów, schematy, rysunki techniczne"
        )

    # Summary
    files_count = (
        (1 if excel_file else 0) +
        len(pdf_files or []) +
        len(json_files or []) +
        len(image_files or [])
    )

    if files_count > 0:
        st.success(f"✅ Wgrano {files_count} plików")

    return {
        'excel': excel_file,
        'pdfs': pdf_files or [],
        'jsons': json_files or [],
        'images': image_files or []
    }


def render_text_input() -> str:
    """
    Render text input section.

    Returns:
        Pasted text/specification
    """
    st.subheader("📝 Opis projektu")

    description = st.text_area(
        "Szczegółowy opis*",
        height=200,
        placeholder="Opisz projekt CAD: cel, komponenty, wymagania techniczne, normy, ilości...",
        help="Im więcej szczegółów, tym lepsza estymacja"
    )

    additional_text = st.text_area(
        "Dodatkowy tekst/specyfikacja (opcjonalnie)",
        height=120,
        placeholder="Wklej dodatkowe informacje: specyfikacje, wymiary, materiały...",
        help="Dodatkowe informacje zostaną dołączone do analizy"
    )

    return description, additional_text
