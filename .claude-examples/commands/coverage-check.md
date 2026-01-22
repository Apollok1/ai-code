---
name: coverage-check
description: Check test coverage and suggest improvements
---

# Coverage Check Command

Analyze test coverage and provide specific recommendations for improvement.

## Execution

```bash
pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=json
```

## Analysis

1. **Overall Coverage**
   - Report total coverage percentage
   - Target: ≥ 80%

2. **Files Below Target**
   - List files with < 80% coverage
   - Show missing line numbers

3. **Uncovered Code**
   - Identify critical uncovered paths:
     - Error handling branches
     - Edge cases in validation
     - Pipeline stage error scenarios

## Recommendations

For each file below target, suggest:
1. **Specific test cases to add**
   - Input scenarios not covered
   - Error conditions to test
   - Edge cases to handle

2. **Priority Level**
   - 🔴 Critical: Core pipeline logic, data validation
   - 🟡 Medium: Utility functions, helpers
   - 🟢 Low: Simple getters, constants

## Example Output

```
📊 Coverage Report
==================
Overall: 75.2% (Target: 80%)

🔴 Files Below Target:
- src/cad/infrastructure/multi_model/orchestrator.py: 68%
  Missing lines: 142-156, 203-210
  Suggested tests:
    ✓ Test error handling when stage fails
    ✓ Test empty component list scenario
    ✓ Test confidence calculation edge cases

- src/cad/domain/models/estimate.py: 72%
  Missing lines: 89-94
  Suggested tests:
    ✓ Test negative hours validation
    ✓ Test from_components() with empty list
```

## View HTML Report

Open detailed report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```
