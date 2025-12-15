"""
CAD Estimator Pro - Sidebar Component

Sidebar configuration UI for model selection and settings.
"""
import streamlit as st
from typing import Any

from cad.domain.models import DepartmentCode, DEPARTMENTS
from cad.domain.models.config import AppConfig
from cad.presentation.state.session_manager import SessionManager


def render_sidebar(
    session: SessionManager,
    app_config: AppConfig,
    available_text_models: list[str],
    available_vision_models: list[str]
) -> dict[str, Any]:
    """
    Render sidebar with configuration options.

    Args:
        session: Session manager
        app_config: Application configuration
        available_text_models: List of available text models
        available_vision_models: List of available vision models

    Returns:
        Dict with selected configuration
    """
    st.sidebar.title("⚙️ Konfiguracja")

    # AI Model selection
    st.sidebar.subheader("🤖 Modele AI")

    # Text model
    current_text = session.get_selected_text_model()
    if current_text not in available_text_models and available_text_models:
        current_text = available_text_models[0]

    selected_text = st.sidebar.selectbox(
        "Model tekstowy (estymacja/JSON)",
        options=available_text_models if available_text_models else ["llama3:latest"],
        index=available_text_models.index(current_text) if current_text in available_text_models else 0,
        help="Model do analizy komponentów i generowania JSON"
    )
    session.set_selected_text_model(selected_text)

    # Vision model
    if available_vision_models:
        current_vision = session.get_selected_vision_model()
        if current_vision not in available_vision_models:
            current_vision = available_vision_models[0]

        selected_vision = st.sidebar.selectbox(
            "Model Vision (obrazy/rysunki)",
            options=available_vision_models,
            index=available_vision_models.index(current_vision) if current_vision in available_vision_models else 0,
            help="Model do analizy zdjęć i schematów technicznych"
        )
        session.set_selected_vision_model(selected_vision)
    else:
        st.sidebar.warning("⚠️ Brak modeli Vision (zainstaluj llava/qwen2-vl)")

    st.sidebar.markdown("---")

    # Multi-Model Pipeline
    st.sidebar.subheader("🎯 Pipeline Estymacji")
    use_multi_model = st.sidebar.checkbox(
        "Multi-Model Pipeline (4 etapy)",
        value=app_config.multi_model.enabled,
        help="""
        Multi-model pipeline używa 4 wyspecjalizowanych modeli:
        1. Analiza techniczna (qwen2.5:14b)
        2. Dekompozycja struktury (qwen2.5:7b)
        3. Estymacja godzin (qwen2.5:7b + wzorce)
        4. Analiza ryzyk (qwen2.5:14b)

        Single-model: Jeden model dla całości (szybsze, mniej dokładne)
        """
    )
    session.set_use_multi_model(use_multi_model)

    if use_multi_model:
        st.sidebar.caption("✅ Multi-model aktywny (4 etapy)")
        with st.sidebar.expander("⚙️ Wybór modeli per etap", expanded=False):
            # Stage 1: Technical Analysis (reasoning model)
            stage1_current = session.get_stage1_model() or app_config.multi_model.stage1_model
            if stage1_current not in available_text_models and available_text_models:
                stage1_current = available_text_models[0]

            stage1_model = st.selectbox(
                "1️⃣ Technical Analysis (reasoning)",
                options=available_text_models if available_text_models else [app_config.multi_model.stage1_model],
                index=available_text_models.index(stage1_current) if stage1_current in available_text_models else 0,
                help="Model do głębokiej analizy technicznej. Zalecane: większy model (14b+)"
            )
            session.set_stage1_model(stage1_model)

            # Stage 2: Structural Decomposition
            stage2_current = session.get_stage2_model() or app_config.multi_model.stage2_model
            if stage2_current not in available_text_models and available_text_models:
                stage2_current = available_text_models[0]

            stage2_model = st.selectbox(
                "2️⃣ Structural Decomposition",
                options=available_text_models if available_text_models else [app_config.multi_model.stage2_model],
                index=available_text_models.index(stage2_current) if stage2_current in available_text_models else 0,
                help="Model do dekompozycji na komponenty. Może być mniejszy (7b)"
            )
            session.set_stage2_model(stage2_model)

            # Stage 3: Hours Estimation
            stage3_current = session.get_stage3_model() or app_config.multi_model.stage3_model
            if stage3_current not in available_text_models and available_text_models:
                stage3_current = available_text_models[0]

            stage3_model = st.selectbox(
                "3️⃣ Hours Estimation",
                options=available_text_models if available_text_models else [app_config.multi_model.stage3_model],
                index=available_text_models.index(stage3_current) if stage3_current in available_text_models else 0,
                help="Model do estymacji godzin. Może być szybki (7b) + wzorce"
            )
            session.set_stage3_model(stage3_model)

            # Stage 4: Risk & Optimization
            stage4_current = session.get_stage4_model() or app_config.multi_model.stage4_model
            if stage4_current not in available_text_models and available_text_models:
                stage4_current = available_text_models[0]

            stage4_model = st.selectbox(
                "4️⃣ Risk Analysis (critical)",
                options=available_text_models if available_text_models else [app_config.multi_model.stage4_model],
                index=available_text_models.index(stage4_current) if stage4_current in available_text_models else 0,
                help="Model do analizy ryzyk. Zalecane: większy model (14b+)"
            )
            session.set_stage4_model(stage4_model)
    else:
        st.sidebar.caption("⚡ Single-model (szybki)")

    st.sidebar.markdown("---")

    # Web lookup
    st.sidebar.subheader("🌐 Web Lookup")
    allow_web = st.sidebar.checkbox(
        "Zezwól na web lookup (normy/benchmarki)",
        value=session.is_web_lookup_enabled(),
        help="Pobiera publiczne dane: normy ISO/EN, benchmarki. NIE wysyła danych projektu!"
    )
    session.set_web_lookup_enabled(allow_web)

    if allow_web:
        st.sidebar.caption("✅ Web lookup aktywny")
    else:
        st.sidebar.caption("🔒 Tryb offline")

    st.sidebar.markdown("---")

    # Pricing
    st.sidebar.subheader("💰 Wycena")
    hourly_rate = st.sidebar.number_input(
        "Stawka PLN/h",
        min_value=1,
        max_value=1000,
        value=session.get_hourly_rate(),
        step=10
    )
    session.set_hourly_rate(hourly_rate)

    st.sidebar.markdown("---")

    # Status
    st.sidebar.subheader("📊 Status Systemu")
    st.sidebar.write(f"✅ Ollama: {len(available_text_models)} modeli")
    st.sidebar.write(f"✅ Vision: {len(available_vision_models)} modeli")

    # Model list expander
    with st.sidebar.expander("📋 Dostępne modele"):
        st.write("**Tekstowe:**")
        for m in available_text_models[:10]:
            st.caption(f"• {m}")
        if len(available_text_models) > 10:
            st.caption(f"... +{len(available_text_models) - 10}")

        if available_vision_models:
            st.write("**Vision:**")
            for m in available_vision_models:
                st.caption(f"• {m}")

    return {
        'text_model': selected_text,
        'vision_model': selected_vision if available_vision_models else None,
        'use_multi_model': use_multi_model,
        'stage1_model': session.get_stage1_model() if use_multi_model else None,
        'stage2_model': session.get_stage2_model() if use_multi_model else None,
        'stage3_model': session.get_stage3_model() if use_multi_model else None,
        'stage4_model': session.get_stage4_model() if use_multi_model else None,
        'allow_web_lookup': allow_web,
        'hourly_rate': hourly_rate
    }


def render_department_selector() -> DepartmentCode:
    """
    Render department selector.

    Returns:
        Selected DepartmentCode
    """
    selected = st.selectbox(
        "Wybierz dział*",
        options=list(DEPARTMENTS.keys()),
        format_func=lambda dept: dept.display_name,
        help="Wybierz dział do którego należy projekt"
    )

    # Show department context
    st.info(f"📋 {selected.context[:200]}...")

    return selected.code
