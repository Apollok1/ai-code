"""CAD Estimator Pro - Domain Models."""

from .department import Department, DepartmentCode, DEPARTMENTS, get_department
from .risk import Risk, RiskLevel
from .suggestion import Suggestion, SuggestionType, SuggestionPriority, SuggestionImpact
from .component import Component, SubComponent, ComponentPattern
from .estimate import Estimate, EstimatePhases
from .project import Project, ProjectVersion

__all__ = [
    # Department
    "Department",
    "DepartmentCode",
    "DEPARTMENTS",
    "get_department",
    # Risk
    "Risk",
    "RiskLevel",
    # Suggestion
    "Suggestion",
    "SuggestionType",
    "SuggestionPriority",
    "SuggestionImpact",
    # Component
    "Component",
    "SubComponent",
    "ComponentPattern",
    # Estimate
    "Estimate",
    "EstimatePhases",
    # Project
    "Project",
    "ProjectVersion",
]
