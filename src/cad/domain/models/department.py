"""
CAD Estimator Pro - Department Domain Model

Defines departments (131-135) with their contexts and aliases.
"""
from dataclasses import dataclass
from enum import Enum


class DepartmentCode(str, Enum):
    """Department codes (131-135)."""

    AUTOMOTIVE = "131"
    INDUSTRIAL_MACHINERY = "132"
    TRANSPORTATION = "133"
    HEAVY_EQUIPMENT = "134"
    SPECIAL_PURPOSE = "135"


@dataclass(frozen=True)
class Department:
    """
    Department with code, name, and industry context.

    Immutable value object representing a CAD estimation department.
    """

    code: DepartmentCode
    name: str
    context: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("Department name cannot be empty")
        if not self.context:
            raise ValueError("Department context cannot be empty")

    @property
    def display_name(self) -> str:
        """Returns formatted display name: '131 - Automotive'."""
        return f"{self.code.value} - {self.name}"


# Predefined departments (from original DEPARTMENTS and DEPARTMENT_CONTEXT)
DEPARTMENTS: dict[DepartmentCode, Department] = {
    DepartmentCode.AUTOMOTIVE: Department(
        code=DepartmentCode.AUTOMOTIVE,
        name="Automotive",
        context="""Branża: AUTOMOTIVE (Faurecia, VW, Merit, Sitech, Joyson)
Specyfika: Komponenty samochodowe, wysokie wymagania jakościowe, spawanie precyzyjne, duże serie produkcyjne, normy automotive (IATF 16949)."""
    ),
    DepartmentCode.INDUSTRIAL_MACHINERY: Department(
        code=DepartmentCode.INDUSTRIAL_MACHINERY,
        name="Industrial Machinery",
        context="""Branża: INDUSTRIAL MACHINERY (PMP, ITM, Amazon)
Specyfika: Maszyny przemysłowe, automatyka, systemy pakowania, linie produkcyjne, robotyka przemysłowa, PLC."""
    ),
    DepartmentCode.TRANSPORTATION: Department(
        code=DepartmentCode.TRANSPORTATION,
        name="Transportation",
        context="""Branża: TRANSPORTATION (Volvo, Scania)
Specyfika: Pojazdy ciężarowe, autobusy, systemy transportowe, wytrzymałość strukturalna, normy transportowe."""
    ),
    DepartmentCode.HEAVY_EQUIPMENT: Department(
        code=DepartmentCode.HEAVY_EQUIPMENT,
        name="Heavy Equipment",
        context="""Branża: HEAVY EQUIPMENT (Volvo CE, Mine Master)
Specyfika: Maszyny budowlane, koparki, ładowarki, ekstremalne obciążenia, odporność na warunki terenowe."""
    ),
    DepartmentCode.SPECIAL_PURPOSE: Department(
        code=DepartmentCode.SPECIAL_PURPOSE,
        name="Special Purpose Machinery",
        context="""Branża: SPECIAL PURPOSE MACHINERY (Bosch, Chassis Brakes, BWI, Besta)
Specyfika: Maszyny specjalne, niestandardowe rozwiązania, prototypy, unikalne wymagania klienta."""
    ),
}


def get_department(code: str | DepartmentCode) -> Department:
    """
    Get department by code string or enum.

    Args:
        code: Department code ('131'-'135' or DepartmentCode enum)

    Returns:
        Department object

    Raises:
        ValueError: If code is invalid
    """
    if isinstance(code, str):
        try:
            code = DepartmentCode(code)
        except ValueError:
            raise ValueError(f"Invalid department code: {code}. Valid codes: 131-135")

    return DEPARTMENTS[code]
