# 🚀 Quick Start - Implementacja Usprawnień CAD Estimator

**Czas na pierwsze usprawnienia: 2-4 godziny**

---

## 📋 Przygotowanie (5 minut)

### 1. Przeczytaj Plan Usprawnień
```bash
cat docs/IMPROVEMENT_PLAN.md
```

**Kluczowe sekcje:**
- Quick Wins - zacznij tutaj!
- Expected Results - jaki zysk z każdego usprawnienia
- Implementation Roadmap - kolejność wdrażania

---

## 🎯 Quick Win #1: Feedback System (2h)

### Krok 1: Uruchom Migrację (2 min)

```bash
# Sprawdź połączenie z bazą
psql -U postgres -d cad_estimator -c "SELECT version();"

# Uruchom migrację
psql -U postgres -d cad_estimator -f migrations/001_add_estimation_feedback.sql

# Weryfikacja
psql -U postgres -d cad_estimator -c "SELECT COUNT(*) FROM estimation_feedback;"
```

**Expected output:**
```
count
-------
     0
(1 row)
```

### Krok 2: Testuj Funkcje SQL (5 min)

```bash
# Test: Sprawdź view
psql -U postgres -d cad_estimator -c "SELECT * FROM estimation_accuracy_summary;"

# Test: Funkcja get_best_estimation_examples
psql -U postgres -d cad_estimator -c "SELECT * FROM get_best_estimation_examples('131', 5, 0.9);"
```

### Krok 3: Dodaj Feedback Widget do UI (1.5h)

**Plik:** `src/cad/presentation/components/feedback_widget.py`

Skopiuj kod z `docs/IMPROVEMENT_PLAN.md`, sekcja "7️⃣ Real-Time Feedback Loop"

**Lub użyj tego szablonu:**

```python
# src/cad/presentation/components/feedback_widget.py
import streamlit as st
from cad.domain.models.feedback import EstimationFeedback
from cad.domain.models.estimate import Estimate, EstimatePhases

def render_feedback_widget(estimate: Estimate, project_id: int, db_client):
    """Widget do zbierania feedbacku podczas realizacji projektu."""

    st.subheader("📊 Podaj rzeczywiste godziny (w trakcie pracy)")

    st.info("💡 Im więcej feedbacku, tym dokładniejsze estymacje w przyszłości!")

    for i, comp in enumerate(estimate.components):
        with st.expander(f"📝 {comp.name}", expanded=(i == 0)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Estymacja: Layout", f"{comp.hours_3d_layout:.1f}h")
                actual_layout = st.number_input(
                    "Rzeczywiste (Layout)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.5,
                    key=f"feedback_layout_{i}"
                )

            with col2:
                st.metric("Estymacja: Detail", f"{comp.hours_3d_detail:.1f}h")
                actual_detail = st.number_input(
                    "Rzeczywiste (Detail)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.5,
                    key=f"feedback_detail_{i}"
                )

            with col3:
                st.metric("Estymacja: 2D", f"{comp.hours_2d:.1f}h")
                actual_2d = st.number_input(
                    "Rzeczywiste (2D)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.5,
                    key=f"feedback_2d_{i}"
                )

            notes = st.text_area(
                "Notatki (opcjonalne)",
                placeholder="Np. 'Komplikacje z tolerancjami', 'Prostsze niż myślałem'",
                key=f"feedback_notes_{i}"
            )

            if st.button(f"💾 Zapisz feedback", key=f"save_feedback_{i}"):
                if actual_layout == 0 and actual_detail == 0 and actual_2d == 0:
                    st.warning("⚠️ Podaj przynajmniej jedną wartość!")
                else:
                    # Create feedback object
                    feedback = EstimationFeedback(
                        component_name=comp.name,
                        component_category=None,  # TODO: extract from metadata
                        department_code=None,     # TODO: extract from project
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
                        model_used=estimate.generation_metadata.get('stage3_model', 'unknown'),
                        complexity_level=estimate.generation_metadata.get('stage1_complexity', 'unknown'),
                        estimated_confidence=comp.confidence,
                        notes=notes
                    )

                    # Save to database
                    save_feedback(db_client, project_id, feedback)

                    # Show accuracy
                    if feedback.error_percentage:
                        accuracy_color = "green" if feedback.accuracy > 80 else "orange" if feedback.accuracy > 60 else "red"
                        st.markdown(f"**Dokładność: :{accuracy_color}[{feedback.accuracy:.1f}%]**")

                    st.success(f"✅ Feedback zapisany dla: {comp.name}")
                    st.balloons()

def save_feedback(db_client, project_id: int, feedback: EstimationFeedback):
    """Save feedback to database."""
    query = """
    INSERT INTO estimation_feedback (
        project_id, component_name, component_category, department_code,
        estimated_hours_3d_layout, estimated_hours_3d_detail, estimated_hours_2d,
        estimated_confidence,
        actual_hours_3d_layout, actual_hours_3d_detail, actual_hours_2d,
        model_used, complexity_level, notes
    ) VALUES (
        %s, %s, %s, %s,
        %s, %s, %s,
        %s,
        %s, %s, %s,
        %s, %s, %s
    )
    """

    with db_client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (
                project_id,
                feedback.component_name,
                feedback.component_category,
                feedback.department_code,
                feedback.estimated_hours.layout,
                feedback.estimated_hours.detail,
                feedback.estimated_hours.documentation,
                feedback.estimated_confidence,
                feedback.actual_hours.layout if feedback.actual_hours else None,
                feedback.actual_hours.detail if feedback.actual_hours else None,
                feedback.actual_hours.documentation if feedback.actual_hours else None,
                feedback.model_used,
                feedback.complexity_level,
                feedback.notes
            ))
        conn.commit()
```

### Krok 4: Dodaj Widget do Main App (10 min)

**Plik:** `src/cad/presentation/app.py`

Znajdź sekcję wyświetlania wyników estymacji i dodaj:

```python
# After displaying estimate results
if st.session_state.get('last_estimate'):
    from src.cad.presentation.components.feedback_widget import render_feedback_widget

    st.markdown("---")
    render_feedback_widget(
        estimate=st.session_state['last_estimate'],
        project_id=st.session_state.get('last_project_id', 0),
        db_client=db
    )
```

### Krok 5: Testuj! (10 min)

```bash
# Uruchom aplikację
streamlit run src/cad/presentation/app.py

# 1. Stwórz nową estymację
# 2. Przewiń w dół do feedback widget
# 3. Wprowadź rzeczywiste godziny
# 4. Kliknij "Zapisz feedback"
# 5. Sprawdź bazę:

psql -U postgres -d cad_estimator -c "SELECT * FROM estimation_feedback ORDER BY created_at DESC LIMIT 5;"
```

**✅ Quick Win #1 Done! Teraz zbieraj dane przez 2-4 tygodnie.**

---

## 🎯 Quick Win #2: Semantic Search (4h)

### Krok 1: Sprawdź pgvector (5 min)

```bash
# Sprawdź czy pgvector jest zainstalowane
psql -U postgres -d cad_estimator -c "SELECT * FROM pg_extension WHERE extname='vector';"

# Jeśli nie ma, zainstaluj:
# Ubuntu/Debian:
sudo apt install postgresql-16-pgvector

# macOS:
brew install pgvector

# Włącz extension:
psql -U postgres -d cad_estimator -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Krok 2: Uruchom Migrację (2 min)

```bash
psql -U postgres -d cad_estimator -f migrations/002_add_component_embeddings.sql

# Weryfikacja
psql -U postgres -d cad_estimator -c "\d component_patterns" | grep embedding
```

**Expected output:**
```
 embedding                  | vector(768)          |
```

### Krok 3: Wygeneruj Embeddings (1-2h, zależnie od liczby komponentów)

```bash
# Sprawdź ile komponentów jest w bazie
psql -U postgres -d cad_estimator -c "SELECT COUNT(*) FROM component_patterns;"

# Uruchom batch job
python scripts/generate_component_embeddings.py

# Powinieneś zobaczyć:
# Found X components needing embeddings
# Processing batch 1/Y
# ✅ Embedding generation complete!
```

**Uwaga:** Jeśli masz dużo komponentów (>1000), może to zająć godzinę.

### Krok 4: Testuj Semantic Search (10 min)

```bash
# Wygeneruj embedding dla test query
psql -U postgres -d cad_estimator

# W psql:
-- Test 1: Znajdź podobne komponenty do "bearing"
-- (Musisz najpierw wygenerować embedding przez Ollama API)

-- Test 2: Sprawdź funkcję search_similar_components
-- (Przykład w migrations/002_add_component_embeddings.sql)
```

### Krok 5: Zintegruj z Stage 3 (1.5h)

**TODO:** Zmodyfikuj `src/cad/infrastructure/multi_model/stage3_hours_estimation.py`

Zobacz sekcję "3️⃣ Semantic Pattern Matching" w `docs/IMPROVEMENT_PLAN.md`

**✅ Quick Win #2 Done! Semantic search działa.**

---

## 🎯 Quick Win #3: Few-Shot Learning (2h)

**Wymagania:** Min. 50 feedbacks zebranych przez Quick Win #1

### Krok 1: Sprawdź Dane (5 min)

```bash
# Ile mamy feedbacks?
psql -U postgres -d cad_estimator -c "
SELECT COUNT(*) AS total_feedbacks,
       COUNT(*) FILTER (WHERE actual_hours_3d_layout IS NOT NULL) AS completed_feedbacks,
       COUNT(*) FILTER (WHERE error_percentage < 10) AS high_quality_examples
FROM estimation_feedback;
"
```

**Potrzebujesz:** ≥ 50 completed_feedbacks, ≥ 5 high_quality_examples

### Krok 2: Pobierz Best Examples (10 min)

```bash
# Test funkcji SQL
psql -U postgres -d cad_estimator -c "
SELECT * FROM get_best_estimation_examples('131', 5, 0.9);
"
```

Powinieneś zobaczyć 5 najlepszych przykładów (accuracy > 90%)

### Krok 3: Dodaj Examples do Promptu (1.5h)

**Plik:** `src/cad/infrastructure/multi_model/stage3_hours_estimation.py`

Zobacz kod w sekcji "1️⃣ Fine-Tuning - Few-Shot Learning" w `docs/IMPROVEMENT_PLAN.md`

**Funkcja do zmodyfikowania:**
```python
def _build_estimation_prompt(self, context, all_nodes, complexity_multiplier):
    # Dodaj wywołanie get_best_estimation_examples
    # Dodaj examples do promptu
```

### Krok 4: A/B Test (30 min)

Uruchom estymację 2 razy:
1. Bez examples (stary prompt)
2. Z examples (nowy prompt)

Porównaj wyniki.

**✅ Quick Win #3 Done! Few-shot learning aktywny.**

---

## 📊 Metryki Sukcesu

Po wdrożeniu Quick Wins, śledź:

### 1. Feedback Collection Rate
```sql
SELECT
    DATE_TRUNC('week', created_at) AS week,
    COUNT(*) AS feedbacks_collected,
    COUNT(*) FILTER (WHERE actual_hours_3d_layout IS NOT NULL) AS completed
FROM estimation_feedback
GROUP BY week
ORDER BY week DESC;
```

**Target:** ≥ 10 completed feedbacks/tydzień

### 2. Model Accuracy Trend
```sql
SELECT
    model_used,
    AVG(100 - error_percentage) AS avg_accuracy,
    COUNT(*) AS samples
FROM estimation_feedback
WHERE actual_hours_3d_layout IS NOT NULL
GROUP BY model_used
ORDER BY avg_accuracy DESC;
```

**Target:** Accuracy > 75% (baseline), później > 85%

### 3. Semantic Match Rate
```sql
-- TODO: Add tracking w kodzie stage3
-- % komponentów które znalazły semantic match
```

**Target:** > 40% (vs ~15% z keyword search)

---

## 🐛 Troubleshooting

### Problem: pgvector nie instaluje się
```bash
# Ubuntu 22.04+
sudo apt-get update
sudo apt install -y postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### Problem: Ollama nie generuje embeddings
```bash
# Sprawdź czy model jest pobrany
ollama list | grep nomic-embed-text

# Jeśli nie ma, pobierz:
ollama pull nomic-embed-text

# Test API:
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test"
}'
```

### Problem: Migration fails
```bash
# Rollback:
psql -U postgres -d cad_estimator -c "
DROP TABLE IF EXISTS estimation_feedback CASCADE;
DROP FUNCTION IF EXISTS get_best_estimation_examples CASCADE;
"

# Spróbuj ponownie
```

---

## 📚 Następne Kroki

Po ukończeniu Quick Wins (2-4 tyg):

1. **Przejrzyj zebrane dane**
   ```bash
   psql -U postgres -d cad_estimator -c "SELECT * FROM estimation_accuracy_summary;"
   ```

2. **Implementuj Complexity Factors** (Faza 4)
3. **Dodaj Analytics Dashboard** (Faza 7)
4. **Rozważ Ensemble Methods** (jeśli budget pozwala)

---

## 💡 Tips

- **Zbieraj feedback konsekwentnie** - Im więcej danych, tym lepiej
- **Komunikuj zespołowi** - Wyjaśnij, dlaczego feedback jest ważny
- **Monitoruj accuracy co tydzień** - Zobacz poprawę w czasie
- **Eksperymentuj** - Testuj różne modele, prompty, thresholdy

---

**Pytania? Sprawdź:** `docs/IMPROVEMENT_PLAN.md` (pełna dokumentacja)

Powodzenia! 🚀
