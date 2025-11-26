"""
CAD Estimator Pro - Multi-Model Results Display

Enhanced results display showing outputs from all pipeline stages.
"""
import streamlit as st
from typing import Any
from ...domain.models import Estimate


def render_multi_model_results(estimate: Estimate, hourly_rate: int) -> None:
    """
    Render enhanced results from multi-model pipeline.

    Shows outputs from all 4 stages plus final estimate.

    Args:
        estimate: Complete estimate with metadata
        hourly_rate: Hourly rate for cost calculation
    """
    metadata = estimate.generation_metadata or {}

    # Check if this is multi-model estimate
    is_multi_model = metadata.get('multi_model', False)

    if not is_multi_model:
        st.info("ℹ️ To jest estymacja single-model. Dla szczegółowych wyników użyj Multi-Model Pipeline.")
        return

    st.markdown("## 🎯 Wyniki Multi-Model Pipeline")
    st.markdown("Szczegółowe wyniki z każdego etapu estymacji:")

    # Stage 1: Technical Analysis
    render_technical_analysis(metadata)

    # Stage 2: Component Structure
    render_component_structure(metadata)

    # Stage 3: Hours Estimation (standard component display)
    st.markdown("---")
    st.markdown("### 3️⃣ Estymacja Godzin")
    st.markdown(f"**Całkowita liczba godzin:** {estimate.total_hours:.1f}h")
    st.markdown(f"**Liczba komponentów:** {estimate.component_count}")
    st.markdown(f"**Średnia pewność:** {estimate.overall_confidence:.0%}")

    # Brief summary
    col1, col2, col3 = st.columns(3)
    col1.metric("3D Layout", f"{estimate.phases.hours_3d_layout:.1f}h")
    col2.metric("3D Detail", f"{estimate.phases.hours_3d_detail:.1f}h")
    col3.metric("2D Dokumentacja", f"{estimate.phases.hours_2d:.1f}h")

    # Stage 4: Risks & Optimization
    render_risks_and_suggestions(estimate, metadata)

    # Cost Summary
    st.markdown("---")
    st.markdown("### 💰 Podsumowanie Kosztów")
    total_cost = estimate.total_hours * hourly_rate
    col1, col2 = st.columns(2)
    col1.metric("Łączny czas", f"{estimate.total_hours:.1f}h")
    col2.metric("Łączny koszt", f"{total_cost:,.0f} PLN", delta=f"{hourly_rate} PLN/h")


def render_technical_analysis(metadata: dict) -> None:
    """Render Stage 1: Technical Analysis results."""
    st.markdown("---")
    st.markdown("### 1️⃣ Analiza Techniczna")

    complexity = metadata.get('stage1_complexity')
    if complexity:
        # Complexity badge
        complexity_colors = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'very_high': '🔴'
        }
        icon = complexity_colors.get(complexity, '⚪')
        st.markdown(f"**Złożoność projektu:** {icon} `{complexity.upper()}`")

    # Technical details in expander
    with st.expander("🔬 Szczegóły techniczne", expanded=False):
        # Note: We'd need to pass more metadata from the orchestrator
        # For now, show what we have
        if 'stage1_materials' in metadata:
            st.markdown("**Materiały:**")
            for material in metadata['stage1_materials']:
                st.markdown(f"- {material}")

        if 'stage1_standards' in metadata:
            st.markdown("**Standardy:**")
            for standard in metadata['stage1_standards']:
                st.markdown(f"- {standard}")

        if 'stage1_challenges' in metadata:
            st.markdown("**Kluczowe wyzwania:**")
            for challenge in metadata['stage1_challenges']:
                st.markdown(f"- {challenge}")


def render_component_structure(metadata: dict) -> None:
    """Render Stage 2: Component Structure results."""
    st.markdown("---")
    st.markdown("### 2️⃣ Struktura Komponentów")

    component_count = metadata.get('stage2_component_count')
    if component_count:
        st.markdown(f"**Liczba komponentów w hierarchii:** {component_count}")

    # Structure details in expander
    with st.expander("🏗️ Hierarchia komponentów", expanded=False):
        st.info("💡 Pełna wizualizacja hierarchii będzie dostępna w następnej wersji. "
                "Na razie komponent count pokazuje głębokość struktury.")

        # TODO: Add tree visualization using st.graphviz_chart or similar
        # For now just show count
        if component_count:
            st.metric("Całkowita liczba węzłów", component_count)


def render_risks_and_suggestions(estimate: Estimate, metadata: dict) -> None:
    """Render Stage 4: Risks & Optimization results."""
    st.markdown("---")
    st.markdown("### 4️⃣ Analiza Ryzyk i Optymalizacja")

    # Risks
    if estimate.risks:
        st.markdown("#### ⚠️ Zidentyfikowane Ryzyka")
        for i, risk in enumerate(estimate.risks, 1):
            impact_icons = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }
            impact_icon = impact_icons.get(risk.impact, '⚪')

            with st.expander(f"{impact_icon} Ryzyko {i}: {risk.description}", expanded=False):
                st.markdown(f"**Kategoria:** `{risk.category}`")
                st.markdown(f"**Impact:** `{risk.impact}`")
                if risk.mitigation:
                    st.markdown(f"**Mitigacja:** {risk.mitigation}")
    else:
        st.success("✅ Nie zidentyfikowano krytycznych ryzyk")

    # Suggestions
    suggestions = metadata.get('suggestions', [])
    if suggestions:
        st.markdown("#### 💡 Sugestie Optymalizacji")
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"{i}. {suggestion}")

    # Assumptions
    assumptions = metadata.get('assumptions', [])
    if assumptions:
        with st.expander("📋 Założenia", expanded=False):
            for assumption in assumptions:
                st.markdown(f"- {assumption}")

    # Warnings
    warnings = metadata.get('warnings', [])
    if warnings:
        with st.expander("⚠️ Ostrzeżenia", expanded=False):
            for warning in warnings:
                st.warning(warning)
