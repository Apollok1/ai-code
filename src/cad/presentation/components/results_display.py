"""
CAD Estimator Pro - Results Display Component

Display estimation results with components breakdown.
"""
import streamlit as st
from cad.domain.models import Estimate


def render_estimate_summary(estimate: Estimate, hourly_rate: int) -> None:
    """
    Render estimate summary (metrics).

    Args:
        estimate: Estimate object
        hourly_rate: Hourly rate PLN
    """
    phases = estimate.phases

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "📐 Layout",
        f"{phases.layout:.1f}h",
        help="3D Layout hours"
    )

    col2.metric(
        "🔧 Detail",
        f"{phases.detail:.1f}h",
        help="3D Detail hours"
    )

    col3.metric(
        "📄 2D",
        f"{phases.documentation:.1f}h",
        help="2D Documentation hours"
    )

    total_cost = estimate.total_hours * hourly_rate
    col4.metric(
        "💰 TOTAL",
        f"{estimate.total_hours:.1f}h",
        delta=f"{int(total_cost):,} PLN".replace(',', ' ')
    )

    # Confidence
    conf_emoji = "🟢" if estimate.overall_confidence > 0.7 else ("🟡" if estimate.overall_confidence > 0.4 else "🔴")
    col5.metric(
        f"{conf_emoji} Confidence",
        f"{estimate.overall_confidence*100:.0f}%",
        delta=f"{estimate.confidence_level} ({estimate.accuracy_estimate})"
    )


def render_components_list(estimate: Estimate) -> None:
    """
    Render components list.

    Args:
        estimate: Estimate object
    """
    st.subheader(f"🔍 Komponenty ({estimate.component_count} pozycji)")

    # Filter non-summary
    components = estimate.non_summary_components

    # Group by confidence level
    high_conf = [c for c in components if c.confidence > 0.7]
    med_conf = [c for c in components if 0.4 <= c.confidence <= 0.7]
    low_conf = [c for c in components if c.confidence < 0.4]

    # Tabs
    tab_all, tab_high, tab_med, tab_low = st.tabs([
        f"📋 Wszystkie ({len(components)})",
        f"🟢 High ({len(high_conf)})",
        f"🟡 Medium ({len(med_conf)})",
        f"🔴 Low ({len(low_conf)})"
    ])

    with tab_all:
        _render_component_table(components)

    with tab_high:
        if high_conf:
            _render_component_table(high_conf)
        else:
            st.info("Brak komponentów z wysoką pewnością")

    with tab_med:
        if med_conf:
            _render_component_table(med_conf)
        else:
            st.info("Brak komponentów ze średnią pewnością")

    with tab_low:
        if low_conf:
            _render_component_table(low_conf)
            st.warning("⚠️ Te komponenty wymagają weryfikacji")
        else:
            st.info("Brak komponentów z niską pewnością")


def _render_component_table(components: list) -> None:
    """Render components as expandable list."""
    for i, comp in enumerate(components):
        conf_emoji = "🟢" if comp.confidence > 0.7 else ("🟡" if comp.confidence > 0.4 else "🔴")

        with st.expander(
            f"{conf_emoji} {comp.name} — {comp.total_hours:.1f}h",
            expanded=(i < 5)  # Expand first 5
        ):
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Layout", f"{comp.hours_3d_layout:.1f}h")
            col2.metric("Detail", f"{comp.hours_3d_detail:.1f}h")
            col3.metric("2D", f"{comp.hours_2d:.1f}h")
            col4.metric("Total", f"{comp.total_hours:.1f}h")

            # Confidence
            st.progress(comp.confidence)
            st.caption(f"**Confidence:** {comp.confidence_level} ({comp.accuracy_estimate})")

            if comp.confidence_reason:
                st.caption(f"**Powód:** {comp.confidence_reason}")

            # Category
            if comp.category:
                st.caption(f"**Kategoria:** {comp.category}")

            # Sub-components
            if comp.subcomponents:
                st.markdown("**Zawiera:**")
                for sub in comp.subcomponents:
                    qty_str = f"{sub.quantity}x " if sub.quantity > 1 else ""
                    st.text(f"  • {qty_str}{sub.name}")

            # Comment
            if comp.comment:
                st.markdown("**Uwagi:**")
                st.info(comp.comment)


def render_risks_and_suggestions(estimate: Estimate) -> None:
    """
    Render risks and suggestions.

    Args:
        estimate: Estimate object
    """
    col1, col2 = st.columns(2)

    with col1:
        if estimate.risks:
            st.subheader(f"⚠️ Ryzyka ({len(estimate.risks)})")
            for risk in estimate.risks:
                impact_emoji = "🔴" if risk.impact.value == "wysoki" else ("🟡" if risk.impact.value == "średni" else "🟢")
                with st.expander(f"{impact_emoji} {risk.risk[:50]}..."):
                    st.markdown(f"**Impact:** {risk.impact.value}")
                    st.markdown(f"**Mitigation:** {risk.mitigation}")

    with col2:
        if estimate.suggestions:
            st.subheader(f"💡 Sugestie ({len(estimate.suggestions)})")
            for sug in estimate.suggestions:
                priority_emoji = "🔴" if sug.priority.value == "high" else ("🟡" if sug.priority.value == "medium" else "🟢")
                with st.expander(f"{priority_emoji} {sug.title}"):
                    st.markdown(sug.description)
                    if sug.impact.hours_delta != 0:
                        st.metric("Zmiana godzin", f"{sug.impact.hours_delta:+.1f}h")


def render_assumptions_and_warnings(estimate: Estimate) -> None:
    """
    Render assumptions and warnings.

    Args:
        estimate: Estimate object
    """
    if estimate.assumptions:
        st.subheader("📌 Założenia")
        for assumption in estimate.assumptions:
            st.write(f"• {assumption}")

    if estimate.warnings:
        st.subheader("⚠️ Ostrzeżenia")
        for warning in estimate.warnings:
            st.warning(warning)
