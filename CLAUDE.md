# CAD Estimation Pipeline - Claude Code Configuration

## 📋 Project Overview
Multi-model AI pipeline for CAD drawing time estimation with 4 processing stages.

## 🎯 Project Structure
```
src/cad/
├── domain/models/        # Core dataclasses (EstimatePhases, Risk, Component)
├── infrastructure/
│   ├── multi_model/     # 4-stage pipeline orchestrator
│   └── database/        # PostgreSQL client
└── application/         # Estimation pipeline
```

## ⚠️ Critical Dataclasses (READ BEFORE USING!)

### EstimatePhases
```python
@dataclass(frozen=True)
class EstimatePhases:
    layout: float          # 3D Layout hours
    detail: float          # 3D Detail hours
    documentation: float   # 2D Documentation hours
```
**CORRECT:** `EstimatePhases(layout=x, detail=y, documentation=z)`
**WRONG:** `EstimatePhases(hours_3d_layout=x, ...)` ❌

### Risk
```python
@dataclass(frozen=True)
class Risk:
    category: str
    impact: str
    mitigation: str
    severity: str
```

### Component
See `src/cad/domain/models/estimate.py:48-78` for full definition.

---

## 🛠️ Development Workflow

### Before Making Changes
1. **Read the dataclass definition** - Never assume parameter names
2. **Run existing tests** - `pytest tests/validate_multi_model.py -v`
3. **Check type hints** - `mypy src/cad/`

### After Making Changes
1. **Run full test suite** - `pytest tests/ -v --cov=src`
2. **Verify coverage ≥ 80%**
3. **Check types** - `mypy src/`
4. **Commit with clear message**

---

## 🧪 Testing Commands

### Full Pipeline Test
```bash
pytest tests/validate_multi_model.py -v
pytest tests/integration/test_pipeline.py -v
```

### Coverage Report
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
# View: htmlcov/index.html
```

### Type Checking
```bash
mypy src/ --show-error-codes
```

---

## 🚨 Common Pitfalls & Solutions

### ❌ Problem: Wrong dataclass parameters
```python
# WRONG
phases = EstimatePhases(
    hours_3d_layout=10,      # ❌ Wrong parameter name
    hours_3d_detail=20,      # ❌ Wrong parameter name
    hours_2d=5               # ❌ Wrong parameter name
)
```

✅ **Solution:** Always read the dataclass definition first
```python
# CORRECT
phases = EstimatePhases(
    layout=10,          # ✅ Correct
    detail=20,          # ✅ Correct
    documentation=5     # ✅ Correct
)
```

### ❌ Problem: Negative hours values
```python
phases = EstimatePhases(layout=-5, detail=10, documentation=5)  # ValueError!
```

✅ **Solution:** Validate before instantiation
```python
total_layout = max(0, sum(c.hours_3d_layout for c in components))
```

### ❌ Problem: Missing test updates
When changing pipeline code, tests break silently.

✅ **Solution:** Always run tests after changes
```bash
pytest tests/validate_multi_model.py -v --tb=short
```

---

## 📚 Key Files Reference

### Pipeline Orchestrator
`src/cad/infrastructure/multi_model/orchestrator.py`
- 4-stage pipeline execution
- Context management
- Phase calculation

### Pipeline Stages
- `stage1_technical_analysis.py` - Document analysis
- `stage2_structural_decomposition.py` - Component breakdown
- `stage3_estimation.py` - Time estimation
- `stage4_risk_optimization.py` - Risk assessment

### Domain Models
`src/cad/domain/models/estimate.py`
- All dataclass definitions
- Validation logic
- from_components() factory methods

---

## 🎓 Development Rules

1. **NEVER** modify dataclass instantiations without reading the definition first
2. **ALWAYS** validate numeric inputs (hours ≥ 0, confidence 0-1)
3. **REQUIRED** test coverage ≥ 80%
4. **USE** frozen dataclasses for immutability
5. **ADD** validation in `__post_init__` methods
6. **LOG** errors with full context, never silent failures

---

## 🔧 Quick Commands

```bash
# Run all tests
pytest tests/ -v

# Test specific module
pytest tests/unit/domain/ -v

# Coverage with missing lines
pytest --cov=src --cov-report=term-missing

# Type check
mypy src/

# Format code
black src/ tests/
ruff check src/ tests/

# Full quality check
black src/ tests/ && ruff check src/ tests/ && mypy src/ && pytest tests/ -v --cov=src
```

---

## 💡 Tips for Claude Code

- Use `/ask` to query this file: `/ask What are EstimatePhases parameters?`
- Before editing pipeline: Read all 4 stage files
- Before using dataclass: Search for its definition
- After pipeline changes: Run `/pipeline-test` (if configured)
- Keep this file updated as project evolves

---

**Last Updated:** 2026-01-22
**Project Version:** 2.0.0
**Python Version:** 3.10+
