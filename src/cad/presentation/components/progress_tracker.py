"""
CAD Estimator Pro - Progress Tracker Component

Real-time progress tracking for multi-model pipeline.
"""
import streamlit as st
from typing import Any
from cad.domain.models.multi_model import PipelineProgress, PipelineStage


STAGE_NAMES = {
    PipelineStage.TECHNICAL_ANALYSIS: "Analiza Techniczna",
    PipelineStage.STRUCTURAL_DECOMPOSITION: "Dekompozycja Struktury",
    PipelineStage.HOURS_ESTIMATION: "Estymacja Godzin",
    PipelineStage.RISK_OPTIMIZATION: "Analiza Ryzyk"
}

STAGE_DESCRIPTIONS = {
    PipelineStage.TECHNICAL_ANALYSIS: "Analizuję materiały, standardy i złożoność techniczne...",
    PipelineStage.STRUCTURAL_DECOMPOSITION: "Tworzę hierarchię komponentów i relacje...",
    PipelineStage.HOURS_ESTIMATION: "Szacuję godziny z użyciem wzorców historycznych...",
    PipelineStage.RISK_OPTIMIZATION: "Identyfikuję ryzyka i sugestie optymalizacji..."
}

STAGE_ICONS = {
    PipelineStage.TECHNICAL_ANALYSIS: "🔬",
    PipelineStage.STRUCTURAL_DECOMPOSITION: "🏗️",
    PipelineStage.HOURS_ESTIMATION: "⏱️",
    PipelineStage.RISK_OPTIMIZATION: "⚠️"
}


class ProgressTracker:
    """
    Progress tracker for multi-model pipeline.

    Displays real-time progress with stage info and progress bar.
    """

    def __init__(self, placeholder: Any):
        """
        Initialize progress tracker.

        Args:
            placeholder: Streamlit placeholder for progress updates
        """
        self.placeholder = placeholder
        self.current_stage = None
        self.total_stages = 4

    def update(self, progress: PipelineProgress) -> None:
        """
        Update progress display.

        Args:
            progress: PipelineProgress object
        """
        self.current_stage = progress.current_stage
        completed = len(progress.completed_stages)

        # Calculate progress percentage
        progress_pct = progress.progress_percent / 100

        # Build stage status list
        stage_status = []
        all_stages = [
            PipelineStage.TECHNICAL_ANALYSIS,
            PipelineStage.STRUCTURAL_DECOMPOSITION,
            PipelineStage.HOURS_ESTIMATION,
            PipelineStage.RISK_OPTIMIZATION
        ]

        for stage in all_stages:
            icon = STAGE_ICONS[stage]
            name = STAGE_NAMES[stage]

            if stage in progress.completed_stages:
                status = "✅"
                stage_status.append(f"{icon} {status} **{name}**")
            elif stage == progress.current_stage:
                status = "🔄"
                desc = STAGE_DESCRIPTIONS[stage]
                stage_status.append(f"{icon} {status} **{name}** - *{desc}*")
            else:
                status = "⏳"
                stage_status.append(f"{icon} {status} {name}")

        # Update placeholder with progress
        with self.placeholder.container():
            st.markdown(f"### 🚀 Multi-Model Pipeline: Etap {completed + 1}/{self.total_stages}")

            # Progress bar
            st.progress(progress_pct)

            # Stage list
            st.markdown("---")
            for status_line in stage_status:
                st.markdown(status_line)


def render_progress_placeholder() -> Any:
    """
    Create placeholder for progress tracker.

    Returns:
        Streamlit placeholder
    """
    return st.empty()
