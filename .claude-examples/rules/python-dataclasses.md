# Python Dataclass Development Rules

## ⚠️ CRITICAL RULES - ALWAYS FOLLOW

### 1. NEVER Assume Dataclass Parameter Names

**WRONG Approach:**
```python
# DON'T guess parameter names
phases = EstimatePhases(hours_3d_layout=10, ...)  # ❌ Might be wrong!
```

**CORRECT Approach:**
```python
# ALWAYS read the dataclass definition first
# Read: src/cad/domain/models/estimate.py
# See: EstimatePhases has fields: layout, detail, documentation
phases = EstimatePhases(layout=10, detail=20, documentation=5)  # ✅
```

### 2. Read Before Write

**Before instantiating ANY dataclass:**
1. ✅ Read the class definition file
2. ✅ Note exact field names and types
3. ✅ Check for validation in `__post_init__`
4. ✅ Use correct parameter names

### 3. Validate Input Values

**All numeric fields must be validated:**
```python
# ❌ WRONG - Can cause ValueError
total_hours = sum(c.hours for c in components)  # Might be negative!
phases = EstimatePhases(layout=total_hours, ...)

# ✅ CORRECT - Validate first
total_hours = max(0, sum(c.hours for c in components))
phases = EstimatePhases(layout=total_hours, ...)
```

### 4. Use Frozen Dataclasses for Immutability

**When creating dataclasses:**
```python
@dataclass(frozen=True)  # ✅ Immutable - prevents accidental modification
class MyData:
    value: float
```

### 5. Add Validation in `__post_init__`

**Validate all constraints:**
```python
@dataclass(frozen=True)
class EstimatePhases:
    layout: float
    detail: float
    documentation: float

    def __post_init__(self):
        # ✅ Validate all fields
        if self.layout < 0:
            raise ValueError(f"Layout hours cannot be negative: {self.layout}")
        if self.detail < 0:
            raise ValueError(f"Detail hours cannot be negative: {self.detail}")
        if self.documentation < 0:
            raise ValueError(f"Documentation hours cannot be negative: {self.documentation}")
```

---

## 📋 Project-Specific Dataclasses

### EstimatePhases
**Location:** `src/cad/domain/models/estimate.py:12-45`

**Correct Usage:**
```python
phases = EstimatePhases(
    layout=10.0,        # 3D Layout hours
    detail=20.0,        # 3D Detail hours
    documentation=5.0   # 2D Documentation hours
)
```

**Validation:**
- All values must be ≥ 0
- Type: float

### Risk
**Location:** `src/cad/domain/models/estimate.py:48-78`

**Correct Usage:**
```python
risk = Risk(
    category="Technical",
    impact="High",
    mitigation="Add buffer time",
    severity="Medium"
)
```

### Component
**Location:** `src/cad/domain/models/estimate.py:81-120`

**Read full definition before using** - many fields!

---

## 🚨 Common Mistakes to Avoid

### Mistake #1: Wrong Parameter Names
```python
# ❌ WRONG
EstimatePhases(hours_3d_layout=10, hours_3d_detail=20, hours_2d=5)

# ✅ CORRECT
EstimatePhases(layout=10, detail=20, documentation=5)
```

### Mistake #2: No Validation
```python
# ❌ WRONG - Can raise ValueError if negative
total = sum(values)
phases = EstimatePhases(layout=total, ...)

# ✅ CORRECT
total = max(0, sum(values))
phases = EstimatePhases(layout=total, ...)
```

### Mistake #3: Mutable Dataclasses
```python
# ❌ WRONG - Can be modified accidentally
@dataclass
class Data:
    value: float

# ✅ CORRECT - Immutable
@dataclass(frozen=True)
class Data:
    value: float
```

### Mistake #4: Missing Type Hints
```python
# ❌ WRONG - No type safety
@dataclass
class Data:
    value  # No type hint!

# ✅ CORRECT
@dataclass(frozen=True)
class Data:
    value: float
```

---

## ✅ Checklist Before Using Dataclass

- [ ] Read the dataclass definition file
- [ ] Note exact field names (not assumed names!)
- [ ] Check field types (float, str, int, etc.)
- [ ] Review `__post_init__` validation rules
- [ ] Validate input values before instantiation
- [ ] Use correct parameter names in constructor
- [ ] Handle potential ValueError from validation

---

## 🔧 Tools to Verify Correctness

### Find Dataclass Definition
```bash
grep -n "class EstimatePhases" src/cad/domain/models/estimate.py
```

### Find All Usages
```bash
grep -rn "EstimatePhases(" src/
```

### Validate Parameters Match
```bash
# Check if using old wrong names
grep -rn "hours_3d_layout\|hours_3d_detail\|hours_2d" src/
```

---

**Remember:** 5 minutes reading the definition saves 30 minutes debugging parameter errors! 🎯
