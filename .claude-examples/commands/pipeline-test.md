---
name: pipeline-test
description: Run complete CAD estimation pipeline test suite
---

# Pipeline Test Command

Run the full multi-model pipeline test suite and report results.

## Execution Steps

1. **Validate Multi-Model Pipeline**
   ```bash
   pytest tests/validate_multi_model.py -v --tb=short
   ```

2. **Integration Tests**
   ```bash
   pytest tests/integration/test_pipeline.py -v --tb=short
   ```

3. **Unit Tests (Domain Models)**
   ```bash
   pytest tests/unit/domain/ -v --tb=short
   ```

4. **Coverage Report**
   ```bash
   pytest tests/ --cov=src --cov-report=term-missing
   ```

## Success Criteria

- ✅ All 4 pipeline stages execute without errors
- ✅ EstimatePhases, Risk, Component objects created correctly
- ✅ All validation rules pass (hours ≥ 0, confidence 0-1)
- ✅ Integration tests pass
- ✅ Coverage ≥ 80%

## Report

Provide summary:
- Total tests: X passed, Y failed
- Coverage: Z%
- Failures: List each with error message
- Recommendations: If any tests fail
