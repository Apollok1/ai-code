"""
CAD Estimator Pro - SessionManager

Typed wrapper around Streamlit's session_state for CAD estimation workflow.
"""
from dataclasses import dataclass, field
from typing import Any
from cad.domain.models import Estimate, Component


@dataclass
class EstimationState:
    """State for estimation workflow."""

    estimate: Estimate | None = None
    base_components: list[Component] = field(default_factory=list)
    ai_adjustments: list[dict] = field(default_factory=list)
    rule_adjustments: list[dict] = field(default_factory=list)
    excel_multipliers: dict[str, float] = field(default_factory=dict)
    clarifying_questions: list[dict] = field(default_factory=list)
    clarifying_answers: dict[str, str] = field(default_factory=dict)
    questions_answered: bool = False
    ai_brief: dict[str, Any] = field(default_factory=dict)
    precheck_results: dict[str, Any] | None = None  # Brain module pre-check results


@dataclass
class UIState:
    """State for UI configuration."""

    hourly_rate: int = 150
    selected_text_model: str = "qwen2.5:7b"
    selected_vision_model: str | None = "qwen2.5vl:7b"
    use_multi_model: bool = True
    stage1_model: str | None = None  # Technical Analysis
    stage2_model: str | None = None  # Structural Decomposition
    stage3_model: str | None = None  # Hours Estimation
    stage4_model: str | None = None  # Risk & Optimization
    allow_web_lookup: bool = False
    admin_authenticated: bool = False


class SessionManager:
    """
    Typed wrapper around st.session_state.

    Provides type-safe access to CAD Estimator session state.
    """

    def __init__(self, session_state: Any):
        """
        Initialize SessionManager.

        Args:
            session_state: Streamlit session_state object
        """
        self._state = session_state

    # Estimation workflow

    def get_estimation_state(self) -> EstimationState:
        """Get estimation workflow state."""
        if "estimation_state" not in self._state:
            self._state["estimation_state"] = EstimationState()
        return self._state["estimation_state"]

    def set_estimation_state(self, state: EstimationState) -> None:
        """Set estimation workflow state."""
        self._state["estimation_state"] = state

    def get_estimate(self) -> Estimate | None:
        """Get current estimate."""
        return self.get_estimation_state().estimate

    def set_estimate(self, estimate: Estimate) -> None:
        """Set current estimate."""
        state = self.get_estimation_state()
        state.estimate = estimate
        self.set_estimation_state(state)

    def get_base_components(self) -> list[Component]:
        """Get base components (from AI/Excel)."""
        return self.get_estimation_state().base_components

    def set_base_components(self, components: list[Component]) -> None:
        """Set base components."""
        state = self.get_estimation_state()
        state.base_components = components
        self.set_estimation_state(state)

    def get_ai_adjustments(self) -> list[dict]:
        """Get AI adjustment proposals."""
        return self.get_estimation_state().ai_adjustments

    def set_ai_adjustments(self, adjustments: list[dict]) -> None:
        """Set AI adjustment proposals."""
        state = self.get_estimation_state()
        state.ai_adjustments = adjustments
        self.set_estimation_state(state)

    def get_rule_adjustments(self) -> list[dict]:
        """Get rule-based adjustment proposals."""
        return self.get_estimation_state().rule_adjustments

    def set_rule_adjustments(self, adjustments: list[dict]) -> None:
        """Set rule-based adjustment proposals."""
        state = self.get_estimation_state()
        state.rule_adjustments = adjustments
        self.set_estimation_state(state)

    def get_excel_multipliers(self) -> dict[str, float]:
        """Get Excel multipliers (layout, detail, documentation)."""
        return self.get_estimation_state().excel_multipliers

    def set_excel_multipliers(self, multipliers: dict[str, float]) -> None:
        """Set Excel multipliers."""
        state = self.get_estimation_state()
        state.excel_multipliers = multipliers
        self.set_estimation_state(state)

    def get_clarifying_questions(self) -> list[dict]:
        """Get clarifying questions for user."""
        return self.get_estimation_state().clarifying_questions

    def set_clarifying_questions(self, questions: list[dict]) -> None:
        """Set clarifying questions."""
        state = self.get_estimation_state()
        state.clarifying_questions = questions
        self.set_estimation_state(state)

    def get_clarifying_answers(self) -> dict[str, str]:
        """Get user's answers to clarifying questions."""
        return self.get_estimation_state().clarifying_answers

    def set_clarifying_answers(self, answers: dict[str, str]) -> None:
        """Set user's answers."""
        state = self.get_estimation_state()
        state.clarifying_answers = answers
        self.set_estimation_state(state)

    def are_questions_answered(self) -> bool:
        """Check if clarifying questions were answered."""
        return self.get_estimation_state().questions_answered

    def set_questions_answered(self, answered: bool) -> None:
        """Set questions answered flag."""
        state = self.get_estimation_state()
        state.questions_answered = answered
        self.set_estimation_state(state)

    def get_ai_brief(self) -> dict[str, Any]:
        """Get AI-generated project brief."""
        return self.get_estimation_state().ai_brief

    def set_ai_brief(self, brief: dict[str, Any]) -> None:
        """Set AI-generated project brief."""
        state = self.get_estimation_state()
        state.ai_brief = brief
        self.set_estimation_state(state)

    def get_precheck_results(self) -> dict[str, Any] | None:
        """Get brain module pre-check results."""
        return self.get_estimation_state().precheck_results

    def set_precheck_results(self, precheck: dict[str, Any]) -> None:
        """Set brain module pre-check results."""
        state = self.get_estimation_state()
        state.precheck_results = precheck
        self.set_estimation_state(state)

    # UI configuration

    def get_ui_state(self) -> UIState:
        """Get UI configuration state."""
        if "ui_state" not in self._state:
            self._state["ui_state"] = UIState()
        return self._state["ui_state"]

    def set_ui_state(self, state: UIState) -> None:
        """Set UI configuration state."""
        self._state["ui_state"] = state

    def get_hourly_rate(self) -> int:
        """Get hourly rate PLN."""
        return self.get_ui_state().hourly_rate

    def set_hourly_rate(self, rate: int) -> None:
        """Set hourly rate PLN."""
        state = self.get_ui_state()
        state.hourly_rate = rate
        self.set_ui_state(state)

    def get_selected_text_model(self) -> str:
        """Get selected text AI model."""
        return self.get_ui_state().selected_text_model

    def set_selected_text_model(self, model: str) -> None:
        """Set selected text AI model."""
        state = self.get_ui_state()
        state.selected_text_model = model
        self.set_ui_state(state)

    def get_selected_vision_model(self) -> str | None:
        """Get selected vision AI model."""
        return self.get_ui_state().selected_vision_model

    def set_selected_vision_model(self, model: str | None) -> None:
        """Set selected vision AI model."""
        state = self.get_ui_state()
        state.selected_vision_model = model
        self.set_ui_state(state)

    def get_use_multi_model(self) -> bool:
        """Check if multi-model pipeline is enabled."""
        return self.get_ui_state().use_multi_model

    def set_use_multi_model(self, enabled: bool) -> None:
        """Set multi-model pipeline enabled flag."""
        state = self.get_ui_state()
        state.use_multi_model = enabled
        self.set_ui_state(state)

    def get_stage1_model(self) -> str | None:
        """Get Stage 1 model (Technical Analysis)."""
        return self.get_ui_state().stage1_model

    def set_stage1_model(self, model: str) -> None:
        """Set Stage 1 model (Technical Analysis)."""
        state = self.get_ui_state()
        state.stage1_model = model
        self.set_ui_state(state)

    def get_stage2_model(self) -> str | None:
        """Get Stage 2 model (Structural Decomposition)."""
        return self.get_ui_state().stage2_model

    def set_stage2_model(self, model: str) -> None:
        """Set Stage 2 model (Structural Decomposition)."""
        state = self.get_ui_state()
        state.stage2_model = model
        self.set_ui_state(state)

    def get_stage3_model(self) -> str | None:
        """Get Stage 3 model (Hours Estimation)."""
        return self.get_ui_state().stage3_model

    def set_stage3_model(self, model: str) -> None:
        """Set Stage 3 model (Hours Estimation)."""
        state = self.get_ui_state()
        state.stage3_model = model
        self.set_ui_state(state)

    def get_stage4_model(self) -> str | None:
        """Get Stage 4 model (Risk & Optimization)."""
        return self.get_ui_state().stage4_model

    def set_stage4_model(self, model: str) -> None:
        """Set Stage 4 model (Risk & Optimization)."""
        state = self.get_ui_state()
        state.stage4_model = model
        self.set_ui_state(state)

    def is_web_lookup_enabled(self) -> bool:
        """Check if web lookup is enabled."""
        return self.get_ui_state().allow_web_lookup

    def set_web_lookup_enabled(self, enabled: bool) -> None:
        """Set web lookup enabled flag."""
        state = self.get_ui_state()
        state.allow_web_lookup = enabled
        self.set_ui_state(state)

    def is_admin_authenticated(self) -> bool:
        """Check if admin is authenticated."""
        return self.get_ui_state().admin_authenticated

    def set_admin_authenticated(self, authenticated: bool) -> None:
        """Set admin authenticated flag."""
        state = self.get_ui_state()
        state.admin_authenticated = authenticated
        self.set_ui_state(state)

    # Utility methods

    def clear_estimation(self) -> None:
        """Clear estimation workflow state."""
        self._state["estimation_state"] = EstimationState()

    def clear_all(self) -> None:
        """Clear all session state."""
        self._state.clear()
