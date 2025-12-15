"""
CAD Estimator Pro - Learning Components

UI components for learning system (pattern learning from feedback).
"""
import streamlit as st
import pandas as pd
from typing import Any

from cad.domain.exceptions import PatternLearningError


def render_add_actual_hours(app: dict, project_id: int):
    """
    Render form to add actual hours to project.

    Args:
        app: App context
        project_id: Project ID
    """
    try:
        project = app["db"].get_project(project_id)
        if not project:
            st.error(f"❌ Projekt {project_id} nie znaleziony")
            return

        st.subheader(f"⏱️ Dodaj actual hours - {project.name}")

        # Show estimated hours
        st.info(f"💡 Estymacja: **{project.estimated_hours:.1f}h**")

        # Form
        with st.form(f"actual_hours_form_{project_id}"):
            actual_hours = st.number_input(
                "Rzeczywiste godziny (actual)",
                min_value=0.0,
                max_value=10000.0,
                value=project.actual_hours or project.estimated_hours,
                step=0.5,
                help="Podaj rzeczywistą liczbę godzin, które zajął projekt"
            )

            notes = st.text_area(
                "Notatki (opcjonalne)",
                placeholder="Np. Większy zakres niż zakładano, dodatkowe wymagania...",
                help="Opcjonalne notatki dotyczące różnicy między estymacją a rzeczywistością"
            )

            submitted = st.form_submit_button("💾 Zapisz i naucz wzorce", type="primary")

            if submitted:
                try:
                    # Calculate accuracy (symetryczna miara dopasowania, bez dzielenia przez zero)
                    if project.estimated_hours > 0 and actual_hours > 0:
                        smaller = min(project.estimated_hours, actual_hours)
                        larger = max(project.estimated_hours, actual_hours)
                        accuracy = smaller / larger  # 1.0 = idealnie, 0.5 = 2x różnica itd.
                    else:
                        accuracy = 0.0

                    # Update project
                    with app["db"].get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE projects
                                SET actual_hours = %s,
                                    accuracy = %s,
                                    updated_at = NOW()
                                WHERE id = %s
                            """, (actual_hours, accuracy, project_id))
                            conn.commit()

                    # Learn from feedback
                    st.info("🧠 Uczę wzorce z feedbacku...")
                    updated_count = app["pattern_learner"].learn_from_project_feedback(
                        project_id=project_id,
                        actual_hours=actual_hours
                    )

                    st.success(
                        f"✅ Zapisano! Zaktualizowano **{updated_count}** wzorców.\n\n"
                        f"📊 Dokładność predykcji: **{accuracy*100:.1f}%**"
                    )

                    # Show difference
                    diff = actual_hours - project.estimated_hours
                    diff_pct = (diff / project.estimated_hours * 100) if project.estimated_hours > 0 else 0

                    if abs(diff_pct) > 20:
                        st.warning(
                            f"⚠️ Duża różnica: {diff:+.1f}h ({diff_pct:+.1f}%)\n\n"
                            f"System nauczył się z tego projektu i poprawi przyszłe predykcje!"
                        )

                    st.rerun()

                except PatternLearningError as e:
                    st.error(f"❌ Błąd uczenia wzorców: {e}")
                except Exception as e:
                    st.error(f"❌ Błąd zapisu: {e}")

    except Exception as e:
        st.error(f"❌ Błąd: {e}")


def render_learning_stats(app: dict):
    """
    Render learning statistics.

    Args:
        app: App context
    """
    st.subheader("📊 Statystyki uczenia")

    try:
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                # Total patterns
                cur.execute("SELECT COUNT(*) FROM component_patterns")
                total_patterns = cur.fetchone()[0]

                # Patterns learned from actual data
                cur.execute("SELECT COUNT(*) FROM component_patterns WHERE source = 'actual'")
                actual_patterns = cur.fetchone()[0]

                # High confidence patterns
                cur.execute("SELECT COUNT(*) FROM component_patterns WHERE confidence > 0.8")
                high_conf = cur.fetchone()[0]

                # Low confidence patterns (need more data)
                cur.execute("SELECT COUNT(*) FROM component_patterns WHERE confidence < 0.5")
                low_conf = cur.fetchone()[0]

                # Average occurrences
                cur.execute("SELECT AVG(occurrences) FROM component_patterns WHERE occurrences > 0")
                avg_occurrences = cur.fetchone()[0] or 0.0

                # Projects with actual hours
                cur.execute("SELECT COUNT(*) FROM projects WHERE actual_hours IS NOT NULL")
                projects_with_actual = cur.fetchone()[0]

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🧩 Wzorce ogółem", total_patterns)
        col2.metric("✅ Z actual data", actual_patterns)
        col3.metric("📁 Projekty z actual", projects_with_actual)

        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Wysoki confidence", high_conf, help="Confidence > 80%")
        col2.metric("🟡 Niski confidence", low_conf, help="Confidence < 50%, potrzeba więcej danych")
        col3.metric("📊 Śr. obserwacje", f"{avg_occurrences:.1f}")

        # Progress bar
        if total_patterns > 0:
            high_conf_pct = (high_conf / total_patterns) * 100
            st.progress(high_conf_pct / 100, text=f"Wzorce wysokiej jakości: {high_conf_pct:.1f}%")

    except Exception as e:
        st.error(f"❌ Błąd pobierania statystyk: {e}")


def render_pattern_improvements(app: dict, limit: int = 10):
    """
    Render recent pattern improvements.

    Args:
        app: App context
        limit: Max number of patterns to show
    """
    st.subheader("🔄 Ostatnio zaktualizowane wzorce")

    try:
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        name,
                        department,
                        avg_hours_total,
                        occurrences,
                        confidence,
                        source,
                        last_updated
                    FROM component_patterns
                    WHERE last_updated IS NOT NULL
                    ORDER BY last_updated DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

        if not rows:
            st.info("📊 Brak ostatnich aktualizacji wzorców")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            "Nazwa", "Dział", "Śr. godziny", "Obserwacje", "Confidence", "Źródło", "Ostatnia aktualizacja"
        ])

        # Format columns
        df["Śr. godziny"] = df["Śr. godziny"].apply(lambda x: f"{x:.1f}h" if pd.notna(x) else "-")
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "-")
        df["Ostatnia aktualizacja"] = pd.to_datetime(df["Ostatnia aktualizacja"]).dt.strftime("%Y-%m-%d %H:%M")

        # Display
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Nazwa": st.column_config.TextColumn(width="large"),
                "Dział": st.column_config.TextColumn(width="small"),
                "Śr. godziny": st.column_config.TextColumn(width="small"),
                "Obserwacje": st.column_config.NumberColumn(width="small"),
                "Confidence": st.column_config.TextColumn(width="small"),
                "Źródło": st.column_config.TextColumn(width="small"),
                "Ostatnia aktualizacja": st.column_config.TextColumn(width="medium"),
            }
        )

    except Exception as e:
        st.error(f"❌ Błąd wyświetlania wzorców: {e}")


def render_batch_import(app: dict):
    """
    Render batch import form for historical data.

    Args:
        app: App context
    """
    st.subheader("📥 Import danych historycznych")

    st.info(
        "💡 **Jak to działa:**\n\n"
        "1. Wybierz dział dla danych historycznych\n"
        "2. Wgraj plik Excel z historycznymi projektami\n"
        "3. System automatycznie wyekstrahuje komponenty i godziny\n"
        "4. Wzorce zostaną nauczone z danych historycznych\n"
        "5. Przyszłe predykcje będą bardziej dokładne"
    )

    # Department selection
    from cad.domain.models.department import DepartmentCode, DEPARTMENTS

    dept_options = list(DEPARTMENTS.keys())
    selected_dept = st.selectbox(
        "Dział dla danych historycznych",
        options=dept_options,
        format_func=lambda d: f"{d.value} - {DEPARTMENTS[d].name}",
        index=0,  # Default: 131 Automotive
        help="Wybierz dział, dla którego importujesz dane historyczne"
    )
    dept_code_str = selected_dept.value  # '131', '132', etc.

    uploaded_file = st.file_uploader(
        "Wybierz plik Excel",
        type=["xlsx", "xls"],
        help="Plik Excel z danymi historycznych projektów (arkusz 'Zestawienie')"
    )

    if uploaded_file:
        try:
            # Show preview
            import pandas as pd
            df = pd.read_excel(uploaded_file, sheet_name="Zestawienie", nrows=5)
            st.write("👀 Podgląd danych:")
            st.dataframe(df.head(), use_container_width=True)

            # Import button
            if st.button("🚀 Importuj i naucz wzorce", type="primary"):
                with st.spinner("Importuję dane historyczne..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)

                        # Use BatchImporter with selected department
                        results = app["batch_importer"].import_from_excel(
                            excel_file=uploaded_file,
                            department=dept_code_str,
                            mark_as_historical=True
                        )

                        st.success(
                            f"✅ Import zakończony!\n\n"
                            f"📁 Zaimportowano: **{results['projects_imported']}** projektów\n"
                            f"🧩 Nauczono: **{results['patterns_learned']}** wzorców\n"
                            f"🔗 Bundle: **{results['bundles_learned']}** relacji"
                        )

                        # Show some stats
                        if results.get('errors'):
                            with st.expander("⚠️ Błędy podczas importu"):
                                for error in results['errors']:
                                    st.warning(error)

                    except Exception as e:
                        st.error(f"❌ Błąd importu: {e}")

        except Exception as e:
            st.error(f"❌ Nie można odczytać pliku: {e}")
