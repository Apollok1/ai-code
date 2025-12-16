"""
CAD Estimator Pro - Multi-Model Results Display

Enhanced results display showing outputs from all pipeline stages.
"""
import streamlit as st
from typing import Any
from cad.domain.models import Estimate, Component
from cad.domain.models.estimate import Risk


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
    is_multi_model = metadata.get("multi_model", False)

    if not is_multi_model:
        st.info(
            "ℹ️ To jest estymacja single-model. Dla szczegółowych wyników użyj Multi-Model Pipeline."
        )
        return

    st.markdown("## 🎯 Wyniki Multi-Model Pipeline")
    st.markdown("Szczegółowe wyniki z każdego etapu estymacji:")

    # ====== 0. SZYBKIE PODSUMOWANIE PROJEKTU ======
    render_quick_summary(estimate, metadata, hourly_rate)

    # ====== 1. ANALIZA TECHNICZNA ======
    render_technical_analysis(metadata)

    # ====== 2. STRUKTURA KOMPONENTÓW ======
    render_component_structure(metadata)

    # ====== 3. ESTYMACJA GODZIN (krótkie metryki) ======
    st.markdown("---")
    st.markdown("### 3️⃣ Estymacja Godzin (podsumowanie liczbowe)")
    st.markdown(f"**Całkowita liczba godzin:** {estimate.total_hours:.1f}h")
    st.markdown(f"**Liczba komponentów:** {estimate.component_count}")
    st.markdown(f"**Średnia pewność (overall):** {estimate.overall_confidence:.0%}")

    col1, col2, col3 = st.columns(3)
    col1.metric("3D Layout", f"{estimate.phases.hours_3d_layout:.1f}h")
    col2.metric("3D Detail", f"{estimate.phases.hours_3d_detail:.1f}h")
    col3.metric("2D Dokumentacja", f"{estimate.phases.hours_2d:.1f}h")

    # ====== 4. RYZYKA I OPTYMALIZACJE ======
    render_risks_and_suggestions(estimate, metadata)

    # ====== 5. KOSZT ======
    st.markdown("---")
    st.markdown("### 💰 Podsumowanie Kosztów")
    total_cost = estimate.total_hours * hourly_rate
    col1, col2 = st.columns(2)
    col1.metric("Łączny czas", f"{estimate.total_hours:.1f}h")
    col2.metric("Łączny koszt", f"{total_cost:,.0f} PLN", delta=f"{hourly_rate} PLN/h")


# === SZYBKIE PODSUMOWANIE ===


def render_quick_summary(estimate: Estimate, metadata: dict, hourly_rate: int) -> None:
    """Szybkie podsumowanie: opis, complexity, total, top komponenty, skrót ryzyk."""
    st.markdown("---")
    st.subheader("📝 Szybkie podsumowanie projektu")

    # Opis projektu (jeśli dostępny)
    description = _get_description_from_estimate(estimate, metadata)

    # Complexity z Stage 1 (metadata)
    complexity = metadata.get("stage1_complexity") or "brak danych"

    # Średni confidence komponentów
    components: list[Component] = list(estimate.components or [])
    if components:
        avg_conf = sum(c.confidence for c in components) / len(components)
    else:
        avg_conf = 0.0

    # Ryzyka – ile i jaki najwyższy poziom
    risks: list[Risk] = getattr(estimate, "risks", []) or []
    highest_impact = _get_highest_risk_impact(risks)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⏱️ Łącznie godzin", f"{estimate.total_hours:.1f} h")
        st.metric("🧩 Komponenty", estimate.component_count)
    with col2:
        cost = estimate.total_hours * float(hourly_rate or 0)
        st.metric("💰 Koszt (szacunek)", f"{cost:,.0f} PLN")
        st.metric("📈 Complexity", str(complexity).upper())
    with col3:
        st.metric("✅ Średni confidence", f"{avg_conf*100:.1f}%")
        if risks:
            st.metric(
                "⚠️ Ryzyka (liczba / max)",
                f"{len(risks)} / {highest_impact}",
            )
        else:
            st.metric("⚠️ Ryzyka", "0 / brak")

    if description:
        with st.expander("📝 Opis projektu (zestawienie)", expanded=False):
            st.write(description)

    # Top komponenty wg godzin
    st.markdown("### 🏗️ Najbardziej czasochłonne komponenty")

    if not components:
        st.caption("Brak komponentów do wyświetlenia.")
    else:
        sorted_components = sorted(
            components, key=lambda c: c.total_hours, reverse=True
        )
        top_n = sorted_components[:5]

        for comp in top_n:
            with st.expander(
                f"{comp.name} — {comp.total_hours:.1f}h "
                f"(Layout: {comp.hours_3d_layout:.1f}h, Detail: {comp.hours_3d_detail:.1f}h, 2D: {comp.hours_2d:.1f}h)",
                expanded=False,
            ):
                st.write(f"**Łącznie:** {comp.total_hours:.1f} h")
                st.write(
                    f"- 3D Layout: **{comp.hours_3d_layout:.1f} h**\n"
                    f"- 3D Detail: **{comp.hours_3d_detail:.1f} h**\n"
                    f"- 2D dokumentacja: **{comp.hours_2d:.1f} h**"
                )
                st.write(f"**Confidence:** {comp.confidence*100:.1f}%")
                if getattr(comp, "confidence_reason", None):
                    st.caption(f"Powód: {comp.confidence_reason}")
                if getattr(comp, "category", None):
                    st.write(f"**Kategoria:** {comp.category}")
                if getattr(comp, "comment", None):
                    st.write(f"**Komentarz:** {comp.comment}")


def _get_description_from_estimate(estimate: Estimate, metadata: dict) -> str:
    """Spróbuj pobrać opis projektu z Estimate lub metadata."""
    desc = getattr(estimate, "description", None)
    if desc:
        return desc

    meta = metadata or getattr(estimate, "generation_metadata", {}) or {}
    if isinstance(meta, dict):
        return meta.get("description", "") or meta.get("project_description", "") or ""

    return ""


def _get_highest_risk_impact(risks: list[Risk]) -> str:
    """Zwraca najwyższy poziom impact z listy ryzyk."""
    if not risks:
        return "brak"

    order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    max_val = 0
    max_label = "brak"
    for r in risks:
        v = order.get(r.impact, 0)
        if v > max_val:
            max_val = v
            max_label = r.impact
    return max_label


# === STAGE 1 ===


def render_technical_analysis(metadata: dict) -> None:
    """Render Stage 1: Technical Analysis results."""
    st.markdown("---")
    st.markdown("### 1️⃣ Analiza Techniczna")

    complexity = metadata.get("stage1_complexity")
    if complexity:
        complexity_colors = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "very_high": "🔴",
        }
        icon = complexity_colors.get(complexity, "⚪")
        st.markdown(f"**Złożoność projektu:** {icon} `{complexity.upper()}`")

    with st.expander("🔬 Szczegóły techniczne", expanded=False):
        if "stage1_materials" in metadata:
            st.markdown("**Materiały:**")
            for material in metadata["stage1_materials"]:
                st.markdown(f"- {material}")

        if "stage1_standards" in metadata:
            st.markdown("**Standardy:**")
            for standard in metadata["stage1_standards"]:
                st.markdown(f"- {standard}")

        if "stage1_challenges" in metadata:
            st.markdown("**Kluczowe wyzwania:**")
            for challenge in metadata["stage1_challenges"]:
                st.markdown(f"- {challenge}")


# === STAGE 2 ===


def render_component_structure(metadata: dict) -> None:
    """Render Stage 2: Component Structure results."""
    st.markdown("---")
    st.markdown("### 2️⃣ Struktura Komponentów")

    component_count = metadata.get("stage2_component_count")
    if component_count:
        st.markdown(f"**Liczba komponentów w hierarchii:** {component_count}")

    with st.expander("🏗️ Hierarchia komponentów", expanded=False):
        st.info(
            "💡 Pełna wizualizacja hierarchii będzie dostępna w następnej wersji. "
            "Na razie liczba węzłów pokazuje rozmiar struktury."
        )
        if component_count:
            st.metric("Całkowita liczba węzłów", component_count)


# === STAGE 4 ===


def render_risks_and_suggestions(estimate: Estimate, metadata: dict) -> None:
    """Render Stage 4: Risks & Optimization results."""
    st.markdown("---")
    st.markdown("### 4️⃣ Analiza Ryzyk i Optymalizacja")

    # Risks
    if estimate.risks:
        st.markdown("#### ⚠️ Zidentyfikowane Ryzyka")
        for i, risk in enumerate(estimate.risks, 1):
            impact_icons = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🟠",
                "critical": "🔴",
            }
            impact_icon = impact_icons.get(risk.impact, "⚪")

            with st.expander(
                f"{impact_icon} Ryzyko {i}: {risk.description}", expanded=False
            ):
                st.markdown(f"**Kategoria:** `{risk.category}`")
                st.markdown(f"**Impact:** `{risk.impact}`")
                if risk.mitigation:
                    st.markdown(f"**Mitigacja:** {risk.mitigation}")
    else:
        st.success("✅ Nie zidentyfikowano krytycznych ryzyk")

    # Suggestions
    suggestions = metadata.get("suggestions", [])
    if suggestions:
        st.markdown("#### 💡 Sugestie Optymalizacji")
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"{i}. {suggestion}")

    # Assumptions
    assumptions = metadata.get("assumptions", [])
    if assumptions:
        with st.expander("📋 Założenia", expanded=False):
            for assumption in assumptions:
                st.markdown(f"- {assumption}")

    # Warnings
    warnings_list = metadata.get("warnings", [])
    if warnings_list:
        with st.expander("⚠️ Ostrzeżenia", expanded=False):
            for warning in warnings_list:
                st.warning(warning)
