# 🚀 CAD Estimator Pro - Plan Usprawnień i Dopracowania

**Data analizy:** 2026-01-23
**Wersja aplikacji:** v2.0
**Analiza:** Architektura, funkcjonalność, dokładność estymacji

---

## 📊 Obecny Stan Aplikacji

### ✅ Mocne Strony
1. **Solidna architektura** - Hexagonal + DDD
2. **4-stage multi-model pipeline** - Dobry podział odpowiedzialności
3. **Pattern matching** - Uczenie się z historii
4. **Validation** - Sanity checks w każdym stage
5. **Immutable dataclasses** - Thread-safe, bez side effects

### ⚠️ Obszary do Poprawy
1. **Dokładność estymacji** - Brak feedbacku i fine-tuningu
2. **Complexity multiplier** - Prosta formuła, brak ML
3. **Pattern matching** - Keyword search, słabe dla podobnych komponentów
4. **Confidence scores** - Nie są kalibrowane
5. **Brak A/B testing** - Nie wiadomo, który model jest lepszy
6. **Single-model fallback** - Nie zaimplementowany (orchestrator.py:248)

---

## 🎯 Priorytetowe Usprawnienia

---

## 1️⃣ **FINE-TUNING MODELI AI** (Największy wpływ!)

### Problem
Modele (Llama, GPT, Claude) używane "out-of-the-box" bez dostosowania do CAD.

### Rozwiązanie: Continuous Learning Loop

#### A) Zbieranie danych treningowych
```python
# Nowa tabela: estimation_feedback
CREATE TABLE estimation_feedback (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    component_name TEXT,

    -- Estymacja AI
    estimated_hours_3d_layout FLOAT,
    estimated_hours_3d_detail FLOAT,
    estimated_hours_2d FLOAT,
    estimated_confidence FLOAT,

    -- Rzeczywiste wartości (po ukończeniu projektu)
    actual_hours_3d_layout FLOAT,
    actual_hours_3d_detail FLOAT,
    actual_hours_2d FLOAT,

    -- Metryki
    error_percentage FLOAT,
    model_used TEXT,
    complexity_level TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);
```

**Implementacja:**
```python
# src/cad/domain/models/feedback.py
@dataclass(frozen=True)
class EstimationFeedback:
    """Feedback loop for model improvement."""
    component_name: str
    estimated_hours: EstimatePhases
    actual_hours: EstimatePhases
    error_percentage: float
    model_used: str

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return (
            abs(self.estimated_hours.layout - self.actual_hours.layout) +
            abs(self.estimated_hours.detail - self.actual_hours.detail) +
            abs(self.estimated_hours.documentation - self.actual_hours.documentation)
        ) / 3

    @property
    def should_retrain(self) -> bool:
        """Should this trigger model retraining?"""
        return self.error_percentage > 25.0  # >25% error
```

#### B) Prompt fine-tuning (bez trenowania modelu)

**Zamiast fine-tuningu całego modelu, użyj Few-Shot Learning:**

```python
# src/cad/infrastructure/multi_model/stage3_hours_estimation.py

def _build_estimation_prompt_with_examples(
    self,
    context: StageContext,
    all_nodes: list[ComponentNode],
    complexity_multiplier: float,
) -> str:
    """Dodaj najlepsze przykłady z historii."""

    # Pobierz 5 najbardziej dokładnych estymacji z bazy
    best_examples = self.db_client.get_best_estimation_examples(
        department=context.department_code,
        limit=5,
        min_accuracy=0.9  # >90% dokładności
    )

    examples_text = ""
    if best_examples:
        examples_text = "\n\nEXAMPLES OF ACCURATE ESTIMATES (learn from these):\n"
        for ex in best_examples:
            examples_text += f"""
Component: {ex['component_name']}
Category: {ex['category']}
Complexity: {ex['complexity']}
Estimated: layout={ex['est_layout']}h, detail={ex['est_detail']}h, 2d={ex['est_2d']}h
Actual: layout={ex['actual_layout']}h, detail={ex['actual_detail']}h, 2d={ex['actual_2d']}h
Accuracy: {ex['accuracy']}%
Reasoning: {ex['reasoning']}
---
"""

    # Dodaj examples_text do promptu
    return f"""
{base_prompt}

{examples_text}

Now estimate the components below...
"""
```

**ROI:** ⭐⭐⭐⭐⭐ (High) - Może poprawić dokładność o 20-40%

---

## 2️⃣ **INTELIGENTNY COMPLEXITY MULTIPLIER**

### Problem
Obecnie: `hours = baseline * complexity_multiplier`
To zbyt proste - nie uwzględnia:
- Typu komponentu (prostokąt vs krzywizna)
- Materiału (stal vs plastik)
- Tolerancji (luźna vs GD&T)

### Rozwiązanie: ML-based Complexity Scoring

```python
# src/cad/domain/models/complexity.py

@dataclass(frozen=True)
class ComplexityFactors:
    """Detailed complexity breakdown."""

    # Geometric complexity (0.5 - 2.0)
    geometric_factor: float = 1.0  # Cylinders=0.8, Freeforms=1.8

    # Material complexity (0.8 - 1.5)
    material_factor: float = 1.0   # Steel=1.0, Titanium=1.4, Plastic=0.9

    # Tolerance complexity (1.0 - 2.5)
    tolerance_factor: float = 1.0  # ±0.5mm=1.0, GD&T=2.0

    # Assembly complexity (1.0 - 1.8)
    assembly_factor: float = 1.0   # Standalone=1.0, Mated=1.5

    # Safety/regulatory (1.0 - 2.0)
    safety_factor: float = 1.0     # Standard=1.0, Safety-critical=1.8

    @property
    def total_multiplier(self) -> float:
        """Compound multiplier (not additive!)."""
        return (
            self.geometric_factor *
            self.material_factor *
            self.tolerance_factor *
            self.assembly_factor *
            self.safety_factor
        )
```

**Implementacja w Stage 3:**

```python
# Zamiast prostego complexity_multiplier
def _calculate_complexity_factors(
    self,
    component: ComponentNode,
    tech_analysis: TechnicalAnalysis
) -> ComplexityFactors:
    """Calculate detailed complexity factors."""

    # 1. Geometric complexity (ML model lub rules)
    geometric = self._assess_geometric_complexity(component)

    # 2. Material complexity (lookup table)
    material = self._assess_material_complexity(tech_analysis.materials)

    # 3. Tolerance (z technical analysis)
    tolerance = self._assess_tolerance_complexity(tech_analysis)

    # 4. Assembly (z structural decomposition)
    assembly = self._assess_assembly_complexity(component)

    # 5. Safety (keyword detection)
    safety = self._assess_safety_requirements(tech_analysis)

    return ComplexityFactors(
        geometric_factor=geometric,
        material_factor=material,
        tolerance_factor=tolerance,
        assembly_factor=assembly,
        safety_factor=safety
    )
```

**ROI:** ⭐⭐⭐⭐ (High) - Dokładność +15-25%

---

## 3️⃣ **SEMANTIC PATTERN MATCHING** (pgvector)

### Problem
Obecnie: Keyword search (`LIKE '%bearing%'`)
Nie znajdzie: "łożysko" ≈ "bearing" ≈ "roller support"

### Rozwiązanie: Embedding-based Search

```python
# src/cad/infrastructure/embeddings/component_embeddings.py

class ComponentEmbeddingsService:
    """Semantic search for similar components."""

    def __init__(self, db_client: DatabaseClient, ai_client: AIClient):
        self.db = db_client
        self.ai = ai_client

    def find_similar_components(
        self,
        component_name: str,
        category: str | None = None,
        department: str | None = None,
        limit: int = 10,
        similarity_threshold: float = 0.8
    ) -> list[dict]:
        """Find semantically similar components."""

        # 1. Generate embedding dla component_name
        embedding = self.ai.generate_embedding(component_name)

        # 2. pgvector cosine similarity search
        query = """
        SELECT
            c.name,
            c.avg_hours_3d_layout,
            c.avg_hours_3d_detail,
            c.avg_hours_2d,
            c.confidence,
            c.occurrence_count,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM component_patterns c
        WHERE 1 - (c.embedding <=> %s::vector) > %s
        {category_filter}
        {department_filter}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """

        # 3. Execute with filters
        results = self.db.execute(
            query,
            (embedding, embedding, similarity_threshold, embedding, limit)
        )

        return results
```

**Dodaj embedding column do component_patterns:**

```sql
-- Migration
ALTER TABLE component_patterns
ADD COLUMN embedding vector(768);  -- Llama 3.2 embeddings = 768 dims

CREATE INDEX ON component_patterns
USING ivfflat (embedding vector_cosine_ops);
```

**Użycie w Stage 3:**

```python
# Zamiast _find_pattern_for_component (keyword search)
def _find_pattern_for_component_semantic(
    self,
    component_name: str,
    category: str,
    department: str
) -> dict | None:
    """Find best match using embeddings."""

    similar = self.embeddings_service.find_similar_components(
        component_name=component_name,
        category=category,
        department=department,
        limit=3,
        similarity_threshold=0.85  # 85%+ similarity
    )

    if similar:
        # Weighted average z top 3 matches
        weights = [s['similarity'] for s in similar[:3]]
        total_weight = sum(weights)

        avg_layout = sum(s['avg_hours_3d_layout'] * s['similarity'] for s in similar[:3]) / total_weight
        avg_detail = sum(s['avg_hours_3d_detail'] * s['similarity'] for s in similar[:3]) / total_weight
        avg_2d = sum(s['avg_hours_2d'] * s['similarity'] for s in similar[:3]) / total_weight

        return {
            'avg_hours_3d_layout': avg_layout,
            'avg_hours_3d_detail': avg_detail,
            'avg_hours_2d': avg_2d,
            'confidence': min(similar[0]['similarity'], 0.95),
            'source': f"Semantic match: {similar[0]['name']} (similarity={similar[0]['similarity']:.2f})"
        }

    return None
```

**ROI:** ⭐⭐⭐⭐⭐ (Very High) - Może znaleźć 3-5x więcej dopasowań

---

## 4️⃣ **CONFIDENCE CALIBRATION**

### Problem
AI podaje `confidence=0.7` ale nie wiadomo, czy to znaczy 70% dokładności.

### Rozwiązanie: Calibration Curve

```python
# src/cad/infrastructure/learning/confidence_calibrator.py

class ConfidenceCalibrator:
    """Calibrate AI confidence scores to real accuracy."""

    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        self.calibration_curve = self._build_calibration_curve()

    def _build_calibration_curve(self) -> dict[float, float]:
        """
        Build mapping: AI confidence → Real accuracy

        Example:
        AI says 0.7 → Real accuracy is 0.55 (overcalibrated)
        AI says 0.9 → Real accuracy is 0.85 (slightly over)
        """

        # Fetch historical data
        query = """
        SELECT
            ROUND(estimated_confidence, 1) AS conf_bucket,
            AVG(CASE
                WHEN ABS(actual_hours - estimated_hours) / NULLIF(actual_hours, 0) < 0.1
                THEN 1.0 ELSE 0.0
            END) AS real_accuracy,
            COUNT(*) AS sample_count
        FROM estimation_feedback
        WHERE actual_hours IS NOT NULL
        GROUP BY ROUND(estimated_confidence, 1)
        HAVING COUNT(*) >= 10  -- Min 10 samples per bucket
        ORDER BY conf_bucket
        """

        results = self.db.execute(query)

        # Build lookup: {0.5: 0.42, 0.6: 0.51, 0.7: 0.68, ...}
        return {r['conf_bucket']: r['real_accuracy'] for r in results}

    def calibrate(self, ai_confidence: float) -> float:
        """Convert AI confidence to calibrated real accuracy."""

        # Round to nearest bucket
        bucket = round(ai_confidence, 1)

        if bucket in self.calibration_curve:
            return self.calibration_curve[bucket]

        # Fallback: linear interpolation
        return ai_confidence * 0.85  # Assume slight overcalibration
```

**Użycie:**

```python
# W Stage 3 po otrzymaniu confidence od AI
calibrator = ConfidenceCalibrator(self.db_client)

raw_confidence = float(est.get("confidence", 0.5))
calibrated_confidence = calibrator.calibrate(raw_confidence)

component = Component(
    ...
    confidence=calibrated_confidence,  # Use calibrated!
    confidence_reason=f"AI: {raw_confidence:.2f}, Calibrated: {calibrated_confidence:.2f}"
)
```

**ROI:** ⭐⭐⭐ (Medium-High) - Lepsze zaufanie do wyników

---

## 5️⃣ **A/B TESTING FRAMEWORK**

### Problem
Nie wiadomo, który model (Llama 3.2, GPT-4, Claude Sonnet) jest najlepszy dla CAD.

### Rozwiązanie: Automatyczne A/B Testing

```python
# src/cad/application/ab_testing.py

class ABTestingOrchestrator:
    """Run A/B tests on different models."""

    def __init__(self, orchestrator: MultiModelOrchestrator):
        self.orch = orchestrator

    def run_ab_test(
        self,
        context: StageContext,
        models_to_test: dict[str, dict[str, str]],
        metrics: list[str] = ["mae", "total_hours", "confidence"]
    ) -> dict:
        """
        Run same estimation with different model configs.

        Args:
            context: Pipeline context
            models_to_test: {"variant_a": {"stage3": "llama3.2"}, "variant_b": {"stage3": "gpt-4"}}
            metrics: Metrics to compare

        Returns:
            Results for each variant
        """

        results = {}

        for variant_name, model_config in models_to_test.items():
            logger.info(f"Running variant: {variant_name}")

            # Run pipeline with this config
            estimate = self.orch.execute_pipeline(
                context=context,
                stage1_model=model_config.get("stage1"),
                stage2_model=model_config.get("stage2"),
                stage3_model=model_config.get("stage3"),
                stage4_model=model_config.get("stage4")
            )

            # Collect metrics
            results[variant_name] = {
                "total_hours": estimate.total_hours,
                "component_count": len(estimate.components),
                "avg_confidence": estimate.overall_confidence,
                "risks_count": len(estimate.risks),
                "estimate": estimate
            }

        return results
```

**Przykład użycia:**

```python
# Test 3 models for Stage 3
ab_test = ABTestingOrchestrator(orchestrator)

variants = {
    "llama32": {"stage3": "llama3.2:latest"},
    "gpt4": {"stage3": "gpt-4"},
    "claude": {"stage3": "claude-sonnet-4"}
}

results = ab_test.run_ab_test(
    context=pipeline_context,
    models_to_test=variants
)

# Compare
for variant, res in results.items():
    print(f"{variant}: {res['total_hours']}h, confidence={res['avg_confidence']:.2f}")

# Later: compare with actual hours to see which was most accurate
```

**ROI:** ⭐⭐⭐⭐ (High) - Find best model for your use case

---

## 6️⃣ **ENSEMBLE METHODS**

### Problem
Jeden model może się mylić. Co jeśli użyć 3 modeli i uśrednić?

### Rozwiązanie: Multi-Model Ensemble

```python
# src/cad/infrastructure/multi_model/ensemble_stage3.py

class EnsembleHoursEstimation:
    """Use multiple models and combine their estimates."""

    def __init__(
        self,
        ai_client: AIClient,
        db_client: DatabaseClient,
        config: MultiModelConfig
    ):
        self.ai = ai_client
        self.db = db_client
        self.config = config

        # Models to ensemble
        self.models = ["llama3.2:latest", "gpt-4-turbo", "claude-sonnet-4"]

    def estimate_hours_ensemble(
        self,
        context: StageContext
    ) -> StageContext:
        """Run estimation with 3 models and combine."""

        all_estimates: list[list[Component]] = []

        # Run each model
        for model in self.models:
            stage3 = HoursEstimationStage(self.ai, self.db, self.config)
            ctx_result = stage3.estimate_hours(context, model=model)
            all_estimates.append(ctx_result.estimated_components)

        # Combine (weighted average by confidence)
        combined = self._combine_estimates(all_estimates)

        return context.with_estimated_components(combined)

    def _combine_estimates(
        self,
        estimates_list: list[list[Component]]
    ) -> list[Component]:
        """Combine multiple estimates via weighted average."""

        # Assume all lists have same component order
        num_components = len(estimates_list[0])
        combined_components = []

        for i in range(num_components):
            # Get i-th component from each model
            components = [estimates[i] for estimates in estimates_list]

            # Weighted average by confidence
            total_conf = sum(c.confidence for c in components)

            avg_layout = sum(c.hours_3d_layout * c.confidence for c in components) / total_conf
            avg_detail = sum(c.hours_3d_detail * c.confidence for c in components) / total_conf
            avg_2d = sum(c.hours_2d * c.confidence for c in components) / total_conf
            avg_confidence = total_conf / len(components)  # Average confidence

            combined = Component(
                name=components[0].name,
                hours_3d_layout=avg_layout,
                hours_3d_detail=avg_detail,
                hours_2d=avg_2d,
                confidence=avg_confidence,
                confidence_reason=f"Ensemble of {len(self.models)} models",
                is_summary=False,
                subcomponents=()
            )
            combined_components.append(combined)

        return combined_components
```

**ROI:** ⭐⭐⭐⭐ (High) - Może poprawić o 10-20%, ale 3x koszt API

---

## 7️⃣ **REAL-TIME FEEDBACK LOOP**

### Problem
Feedback zbierany tylko na końcu projektu (po miesiącach).

### Rozwiązanie: Progressive Feedback

```python
# src/cad/presentation/components/feedback_widget.py

import streamlit as st

def render_feedback_widget(estimate: Estimate, project_id: int):
    """Widget do zbierania feedbacku w trakcie realizacji."""

    st.subheader("📊 Aktualizuj estymację podczas pracy")

    with st.expander("Podaj rzeczywiste godziny (w trakcie pracy)"):
        for comp in estimate.components:
            st.write(f"**{comp.name}**")
            col1, col2, col3 = st.columns(3)

            with col1:
                actual_layout = st.number_input(
                    f"Layout (est: {comp.hours_3d_layout:.1f}h)",
                    min_value=0.0,
                    key=f"actual_layout_{comp.name}"
                )

            with col2:
                actual_detail = st.number_input(
                    f"Detail (est: {comp.hours_3d_detail:.1f}h)",
                    min_value=0.0,
                    key=f"actual_detail_{comp.name}"
                )

            with col3:
                actual_2d = st.number_input(
                    f"2D (est: {comp.hours_2d:.1f}h)",
                    min_value=0.0,
                    key=f"actual_2d_{comp.name}"
                )

            if st.button(f"Zapisz feedback dla {comp.name}"):
                # Save to estimation_feedback table
                feedback = EstimationFeedback(
                    component_name=comp.name,
                    estimated_hours=EstimatePhases(
                        layout=comp.hours_3d_layout,
                        detail=comp.hours_3d_detail,
                        documentation=comp.hours_2d
                    ),
                    actual_hours=EstimatePhases(
                        layout=actual_layout,
                        detail=actual_detail,
                        documentation=actual_2d
                    ),
                    error_percentage=calculate_error(comp, actual_layout, actual_detail, actual_2d),
                    model_used=estimate.generation_metadata.get('stage3_model', 'unknown')
                )

                db.save_feedback(feedback)
                st.success(f"✅ Feedback zapisany dla {comp.name}")
```

**ROI:** ⭐⭐⭐⭐⭐ (Critical) - Kluczowe dla continuous learning!

---

## 8️⃣ **VISUALIZATION & ANALYTICS**

### Dodaj dashboard z metrykami:

```python
# src/cad/presentation/pages/analytics.py

import streamlit as st
import plotly.express as px

def render_analytics_dashboard(db: DatabaseClient):
    """Dashboard z metrykami dokładności."""

    st.title("📊 Model Performance Analytics")

    # 1. Accuracy over time
    accuracy_data = db.execute("""
        SELECT
            DATE_TRUNC('week', created_at) AS week,
            AVG(100 - ABS(error_percentage)) AS avg_accuracy,
            COUNT(*) AS sample_count
        FROM estimation_feedback
        GROUP BY week
        ORDER BY week DESC
        LIMIT 12
    """)

    fig = px.line(
        accuracy_data,
        x='week',
        y='avg_accuracy',
        title='Model Accuracy Trend (Last 12 Weeks)',
        labels={'avg_accuracy': 'Accuracy %', 'week': 'Week'}
    )
    st.plotly_chart(fig)

    # 2. Best/Worst components
    component_accuracy = db.execute("""
        SELECT
            component_name,
            AVG(100 - ABS(error_percentage)) AS avg_accuracy,
            COUNT(*) AS estimate_count
        FROM estimation_feedback
        GROUP BY component_name
        HAVING COUNT(*) >= 3
        ORDER BY avg_accuracy DESC
        LIMIT 20
    """)

    st.subheader("🏆 Best Estimated Components")
    st.dataframe(component_accuracy[:10])

    st.subheader("⚠️ Worst Estimated Components (needs improvement)")
    st.dataframe(component_accuracy[-10:])

    # 3. Model comparison
    model_perf = db.execute("""
        SELECT
            model_used,
            AVG(100 - ABS(error_percentage)) AS avg_accuracy,
            COUNT(*) AS samples
        FROM estimation_feedback
        GROUP BY model_used
        ORDER BY avg_accuracy DESC
    """)

    st.subheader("🤖 Model Performance Comparison")
    fig2 = px.bar(
        model_perf,
        x='model_used',
        y='avg_accuracy',
        title='Accuracy by Model',
        text='samples'
    )
    st.plotly_chart(fig2)
```

**ROI:** ⭐⭐⭐ (Medium) - Better visibility into model performance

---

## 🎯 **ROADMAP IMPLEMENTACJI**

### **Faza 1: Foundation (Tydzień 1-2)** 🟢

**Priorytet: KRYTYCZNY**

1. ✅ Dodaj tabelę `estimation_feedback` do bazy
2. ✅ Dodaj `EstimationFeedback` model do domain
3. ✅ Zaimplementuj feedback widget w UI
4. ✅ Zbieraj feedback przez 2-4 tygodnie (minimum 50-100 samples)

**Pliki do stworzenia/zmodyfikowania:**
```
src/cad/domain/models/feedback.py          # NEW
src/cad/presentation/components/feedback_widget.py  # NEW
migrations/add_estimation_feedback.sql     # NEW
```

**Kod do implementacji:**
- [Zobacz sekcję 7️⃣ Real-Time Feedback Loop](#7️⃣-real-time-feedback-loop)

---

### **Faza 2: Semantic Search (Tydzień 3)** 🟡

**Priorytet: WYSOKI**

1. ✅ Dodaj `embedding` column do `component_patterns`
2. ✅ Generate embeddings dla existing patterns (batch job)
3. ✅ Zaimplementuj `ComponentEmbeddingsService`
4. ✅ Zamień keyword search na semantic w Stage 3

**Pliki:**
```
src/cad/infrastructure/embeddings/component_embeddings.py  # NEW
migrations/add_component_embeddings.sql                    # NEW
src/cad/infrastructure/multi_model/stage3_hours_estimation.py  # MODIFY
```

**Kod:**
- [Zobacz sekcję 3️⃣ Semantic Pattern Matching](#3️⃣-semantic-pattern-matching-pgvector)

**Expected Impact:** +30-50% więcej dopasowań

---

### **Faza 3: Confidence Calibration (Tydzień 4)** 🟡

**Priorytet: ŚREDNI-WYSOKI**

1. ✅ Zaimplementuj `ConfidenceCalibrator`
2. ✅ Build calibration curve z zebranych feedbacks
3. ✅ Użyj calibrated confidence w Stage 3

**Pliki:**
```
src/cad/infrastructure/learning/confidence_calibrator.py  # NEW
src/cad/infrastructure/multi_model/stage3_hours_estimation.py  # MODIFY
```

**Kod:**
- [Zobacz sekcję 4️⃣ Confidence Calibration](#4️⃣-confidence-calibration)

**Expected Impact:** Lepsze zaufanie do confidence scores

---

### **Faza 4: Complexity Factors (Tydzień 5-6)** 🟡

**Priorytet: WYSOKI**

1. ✅ Dodaj `ComplexityFactors` model
2. ✅ Zaimplementuj rules dla każdego faktora
3. ✅ Zamień prosty `complexity_multiplier` na multi-factor

**Pliki:**
```
src/cad/domain/models/complexity.py  # NEW
src/cad/infrastructure/multi_model/stage3_hours_estimation.py  # MODIFY
```

**Kod:**
- [Zobacz sekcję 2️⃣ Inteligentny Complexity Multiplier](#2️⃣-inteligentny-complexity-multiplier)

**Expected Impact:** +15-25% dokładności

---

### **Faza 5: Few-Shot Learning (Tydzień 7)** 🟢

**Priorytet: KRYTYCZNY**

1. ✅ Query `estimation_feedback` for best examples (accuracy > 90%)
2. ✅ Dodaj examples do promptu w Stage 3
3. ✅ A/B test: z examples vs bez

**Pliki:**
```
src/cad/infrastructure/multi_model/stage3_hours_estimation.py  # MODIFY
```

**Kod:**
- [Zobacz sekcję 1️⃣ Fine-Tuning Modeli AI (część B)](#b-prompt-fine-tuning-bez-trenowania-modelu)

**Expected Impact:** +20-40% dokładności (BIGGEST IMPACT!)

---

### **Faza 6: A/B Testing & Ensemble (Tydzień 8-10)** 🔵

**Priorytet: NICE-TO-HAVE**

1. ✅ Zaimplementuj `ABTestingOrchestrator`
2. ✅ Test różnych modeli (Llama vs GPT vs Claude)
3. ✅ Opcjonalnie: Ensemble (jeśli budget pozwala)

**Pliki:**
```
src/cad/application/ab_testing.py  # NEW
src/cad/infrastructure/multi_model/ensemble_stage3.py  # NEW (optional)
```

**Kod:**
- [Zobacz sekcję 5️⃣ A/B Testing Framework](#5️⃣-ab-testing-framework)
- [Zobacz sekcję 6️⃣ Ensemble Methods](#6️⃣-ensemble-methods)

**Expected Impact:** Find best model + optional +10-20% from ensemble

---

### **Faza 7: Analytics Dashboard (Tydzień 11)** 🔵

**Priorytet: NICE-TO-HAVE**

1. ✅ Dodaj analytics page do Streamlit
2. ✅ Wykresy: accuracy over time, best/worst components, model comparison

**Pliki:**
```
src/cad/presentation/pages/analytics.py  # NEW
```

**Kod:**
- [Zobacz sekcję 8️⃣ Visualization & Analytics](#8️⃣-visualization--analytics)

---

## 📊 **EXPECTED RESULTS**

| Usprawnienie | Impact na Dokładność | Effort | ROI | Priorytet |
|--------------|---------------------|--------|-----|-----------|
| Few-Shot Learning (Best Examples) | **+20-40%** | Niski | ⭐⭐⭐⭐⭐ | 1 🔥 |
| Semantic Pattern Matching | **+30-50% więcej dopasowań** | Średni | ⭐⭐⭐⭐⭐ | 2 🔥 |
| Complexity Factors | **+15-25%** | Średni-Wysoki | ⭐⭐⭐⭐ | 3 |
| Confidence Calibration | Lepsze zaufanie | Niski | ⭐⭐⭐ | 4 |
| A/B Testing | Find best model | Niski | ⭐⭐⭐⭐ | 5 |
| Ensemble Methods | **+10-20%** (ale 3x koszt) | Średni | ⭐⭐⭐ | 6 |
| Real-Time Feedback | **Kluczowe dla learning** | Niski | ⭐⭐⭐⭐⭐ | 1 🔥 |
| Analytics Dashboard | Visibility | Niski | ⭐⭐⭐ | 7 |

**Łączny potencjał poprawy dokładności: +50-80%** 🎯

---

## 🚨 **QUICK WINS** (Zacznij od tego!)

### **Quick Win #1: Feedback Widget** (1 dzień)
```bash
# 1. Dodaj migration
psql -d cad_estimator -f migrations/add_estimation_feedback.sql

# 2. Dodaj feedback widget do UI
# (kod w sekcji 7)

# 3. Zbieraj dane przez 2 tygodnie
```

### **Quick Win #2: Few-Shot Examples** (2 dni)
```python
# Po zebraniu 50+ feedbacks:
# Dodaj best examples do prompta Stage 3
# (kod w sekcji 1B)
```

### **Quick Win #3: Semantic Search** (3-4 dni)
```bash
# 1. Dodaj embedding column
ALTER TABLE component_patterns ADD COLUMN embedding vector(768);

# 2. Generate embeddings (batch job)
python scripts/generate_component_embeddings.py

# 3. Użyj w Stage 3
# (kod w sekcji 3)
```

---

## 📈 **METRYKI DO ŚLEDZENIA**

1. **MAE (Mean Absolute Error)**
   - Target: < 15% średniego błędu

2. **Accuracy @ 10%**
   - % estymacji w zakresie ±10% od actual
   - Target: > 60%

3. **Confidence Calibration**
   - Czy confidence=0.7 rzeczywiście daje 70% accuracy?

4. **Pattern Match Rate**
   - % komponentów z pattern match (semantic search)
   - Target: > 40% (obecnie ~15-20% z keyword)

5. **Model Comparison**
   - Który model (Llama/GPT/Claude) jest najbardziej accurate?

---

## 🎓 **DODATKOWE USPRAWNIENIA** (Long-term)

### 1. **Component Images Recognition** (CV + Vision Models)
```python
# Use vision models to detect component type from images
# "This looks like a bearing" → auto-suggest category
```

### 2. **BOM Parser Integration**
```python
# Import BOM (Bill of Materials) from PDFs
# Auto-populate components
```

### 3. **Time Series Forecasting**
```python
# Predict project completion date based on:
# - Estimated hours
# - Team velocity
# - Historical delays
```

### 4. **Risk Prediction Model**
```python
# ML model to predict which risks will actually materialize
# Based on historical project outcomes
```

---

## 🛠️ **IMPLEMENTACJA - NASTĘPNE KROKI**

Chcesz, żebym:

1. **Zaimplementował feedback widget + migration?** (Quick Win #1)
2. **Stworzył kod dla semantic search?** (Quick Win #3)
3. **Dodał few-shot learning do Stage 3?** (Quick Win #2)
4. **Wszystkie powyższe w kolejności priorytetów?**

Powiedz, od czego zaczynamy! 🚀

---

**Dokument stworzony:** 2026-01-23
**Autor:** Claude (Sonnet 4.5)
**Wersja:** 1.0
