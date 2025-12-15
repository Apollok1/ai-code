"""
CAD Estimator Pro - Project History Components

UI components for project history display and management.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from io import BytesIO

from cad.domain.models.department import DepartmentCode, DEPARTMENTS


def render_project_filters() -> dict:
    """
    Render project filters.

    Returns:
        Dict with filter values
    """
    st.subheader("🔍 Filtry")

    col1, col2, col3 = st.columns(3)

    with col1:
        department_filter = st.selectbox(
            "Dział",
            options=["Wszystkie"] + [f"{d.value} - {DEPARTMENTS[d].name}" for d in DepartmentCode],
            index=0
        )

    with col2:
        days_back = st.selectbox(
            "Okres",
            options=[7, 14, 30, 60, 90, 180, 365, -1],
            format_func=lambda x: f"Ostatnie {x} dni" if x > 0 else "Wszystkie",
            index=4  # Default 90 days
        )

    with col3:
        has_actual = st.selectbox(
            "Status",
            options=["Wszystkie", "Z actual hours", "Bez actual hours"],
            index=0
        )

    return {
        "department": None if department_filter == "Wszystkie" else department_filter.split(" - ")[0],
        "days_back": days_back if days_back > 0 else None,
        "has_actual": None if has_actual == "Wszystkie" else (has_actual == "Z actual hours")
    }


def render_projects_table(app: dict, filters: dict):
    """
    Render projects table with filtering.

    Args:
        app: App context
        filters: Filter values from render_project_filters
    """
    try:
        # Build query
        where_clauses = []
        params = []

        if filters["department"]:
            where_clauses.append("department = %s")
            params.append(filters["department"])

        if filters["days_back"]:
            cutoff_date = datetime.now() - timedelta(days=filters["days_back"])
            where_clauses.append("created_at >= %s")
            params.append(cutoff_date)

        if filters["has_actual"] is not None:
            if filters["has_actual"]:
                where_clauses.append("actual_hours IS NOT NULL")
            else:
                where_clauses.append("actual_hours IS NULL")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Query projects
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        id,
                        name,
                        client,
                        department,
                        estimated_hours,
                        actual_hours,
                        accuracy,
                        created_at,
                        is_historical
                    FROM projects
                    WHERE {where_sql}
                    ORDER BY created_at DESC
                    LIMIT 100
                """, params)
                rows = cur.fetchall()

        if not rows:
            st.info("🔍 Brak projektów spełniających kryteria")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            "ID", "Nazwa", "Klient", "Dział", "Estymacja [h]",
            "Actual [h]", "Dokładność", "Data", "Historyczny"
        ])

        # Format columns
        df["Estymacja [h]"] = df["Estymacja [h]"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        df["Actual [h]"] = df["Actual [h]"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        df["Dokładność"] = df["Dokładność"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
        )
        df["Data"] = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m-%d %H:%M")
        df["Historyczny"] = df["Historyczny"].apply(lambda x: "✅" if x else "")

        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn(width="small"),
                "Nazwa": st.column_config.TextColumn(width="large"),
                "Klient": st.column_config.TextColumn(width="medium"),
                "Dział": st.column_config.TextColumn(width="small"),
                "Estymacja [h]": st.column_config.TextColumn(width="small"),
                "Actual [h]": st.column_config.TextColumn(width="small"),
                "Dokładność": st.column_config.TextColumn(width="small"),
                "Data": st.column_config.TextColumn(width="medium"),
                "Historyczny": st.column_config.TextColumn(width="small"),
            }
        )

        # Stats summary
        total = len(df)
        with_actual = df["Actual [h]"].apply(lambda x: x != "-").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("📁 Wszystkie projekty", total)
        col2.metric("✅ Z actual hours", with_actual)
        col3.metric("⏳ Bez actual hours", total - with_actual)

    except Exception as e:
        st.error(f"❌ Błąd pobierania projektów: {e}")


def render_project_details(app: dict, project_id: int):
    """
    Render detailed project view.

    Args:
        app: App context
        project_id: Project ID to display
    """
    try:
        project = app["db"].get_project(project_id)
        if not project:
            st.error(f"❌ Projekt {project_id} nie znaleziony")
            return

        st.subheader(f"📋 {project.name}")

        # Project info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Klient:** {project.client or '-'}")
            st.write(f"**Dział:** {project.department.value}")
        with col2:
            st.write(f"**Estymacja:** {project.estimated_hours:.1f}h")
            if project.actual_hours:
                st.write(f"**Actual:** {project.actual_hours:.1f}h")
        with col3:
            st.write(f"**Data:** {project.created_at.strftime('%Y-%m-%d')}")
            if project.accuracy:
                accuracy_pct = project.accuracy * 100
                color = "green" if accuracy_pct >= 80 else "orange" if accuracy_pct >= 60 else "red"
                st.markdown(f"**Dokładność:** :{color}[{accuracy_pct:.1f}%]")

        # Description
        if project.description:
            with st.expander("📝 Opis projektu"):
                st.write(project.description)

        # Components
        if project.estimate and project.estimate.components:
            st.markdown("---")
            st.subheader("🧩 Komponenty")

            components_data = []
            for comp in project.estimate.non_summary_components:
                components_data.append({
                    "Nazwa": comp.name,
                    "Ilość": comp.quantity,
                    "Layout [h]": f"{comp.hours_3d_layout:.1f}",
                    "Detail [h]": f"{comp.hours_3d_detail:.1f}",
                    "2D [h]": f"{comp.hours_2d:.1f}",
                    "Razem [h]": f"{comp.total_hours:.1f}"
                })

            df = pd.DataFrame(components_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # AI Analysis
        if project.ai_analysis:
            with st.expander("🤖 Analiza AI"):
                st.write(project.ai_analysis)

    except Exception as e:
        st.error(f"❌ Błąd wyświetlania projektu: {e}")


def render_accuracy_chart(app: dict, filters: dict):
    """
    Render accuracy chart for projects with actual hours.

    Args:
        app: App context
        filters: Filter values
    """
    try:
        # Build query
        where_clauses = []
        params = []

        where_clauses.append("actual_hours IS NOT NULL")
        where_clauses.append("accuracy IS NOT NULL")

        if filters["department"]:
            where_clauses.append("department = %s")
            params.append(filters["department"])

        if filters["days_back"]:
            cutoff_date = datetime.now() - timedelta(days=filters["days_back"])
            where_clauses.append("created_at >= %s")
            params.append(cutoff_date)

        where_sql = " AND ".join(where_clauses)

        # Query
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        name,
                        created_at,
                        estimated_hours,
                        actual_hours,
                        accuracy
                    FROM projects
                    WHERE {where_sql}
                    ORDER BY created_at ASC
                """, params)
                rows = cur.fetchall()

        if not rows:
            st.info("📊 Brak projektów z danymi actual hours do wykresu")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            "Projekt", "Data", "Estymacja", "Actual", "Dokładność"
        ])
        df["Dokładność %"] = df["Dokładność"] * 100
        df["Data"] = pd.to_datetime(df["Data"])

        # Chart
        st.subheader("📊 Dokładność predykcji w czasie")
        st.line_chart(
            df,
            x="Data",
            y="Dokładność %",
            use_container_width=True
        )

        # Stats
        avg_accuracy = df["Dokładność %"].mean()
        median_accuracy = df["Dokładność %"].median()

        col1, col2, col3 = st.columns(3)
        col1.metric("Średnia dokładność", f"{avg_accuracy:.1f}%")
        col2.metric("Mediana dokładności", f"{median_accuracy:.1f}%")
        col3.metric("Liczba projektów", len(df))

    except Exception as e:
        st.error(f"❌ Błąd generowania wykresu: {e}")


def render_export_projects(app: dict, filters: dict):
    """
    Render export projects to CSV/Excel.

    Args:
        app: App context
        filters: Filter values
    """
    st.subheader("📥 Export danych")

    try:
        # Build query
        where_clauses = []
        params = []

        if filters["department"]:
            where_clauses.append("department = %s")
            params.append(filters["department"])

        if filters["days_back"]:
            cutoff_date = datetime.now() - timedelta(days=filters["days_back"])
            where_clauses.append("created_at >= %s")
            params.append(cutoff_date)

        if filters["has_actual"] is not None:
            if filters["has_actual"]:
                where_clauses.append("actual_hours IS NOT NULL")
            else:
                where_clauses.append("actual_hours IS NULL")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Query projects
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        id,
                        name,
                        client,
                        department,
                        description,
                        estimated_hours,
                        estimated_hours_3d_layout,
                        estimated_hours_3d_detail,
                        estimated_hours_2d,
                        actual_hours,
                        accuracy,
                        created_at,
                        is_historical
                    FROM projects
                    WHERE {where_sql}
                    ORDER BY created_at DESC
                """, params)
                rows = cur.fetchall()

        if not rows:
            st.info("🔍 Brak projektów do exportu")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            "ID", "Nazwa", "Klient", "Dział", "Opis",
            "Estymacja [h]", "Layout [h]", "Detail [h]", "2D [h]",
            "Actual [h]", "Dokładność", "Data utworzenia", "Historyczny"
        ])

        # Format
        df["Dokładność"] = df["Dokładność"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        df["Data utworzenia"] = pd.to_datetime(df["Data utworzenia"]).dt.strftime("%Y-%m-%d %H:%M")

        st.write(f"**Liczba projektów do exportu:** {len(df)}")

        # Export format
        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Pobierz CSV",
                data=csv,
                file_name=f"projekty_cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Excel export
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Projekty', index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Pobierz Excel",
                data=buffer,
                file_name=f"projekty_cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Błąd exportu: {e}")


def render_export_patterns(app: dict):
    """
    Render export patterns to CSV/Excel.

    Args:
        app: App context
    """
    st.subheader("📥 Export wzorców")

    try:
        # Department filter
        department = st.selectbox(
            "Wybierz dział dla exportu wzorców",
            options=["Wszystkie", "131", "132", "133", "134", "135"],
            index=0,
            key="export_patterns_dept"
        )

        # Query patterns
        where_sql = "1=1" if department == "Wszystkie" else "department = %s"
        params = [] if department == "Wszystkie" else [department]

        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        name,
                        pattern_key,
                        department,
                        avg_hours_3d_layout,
                        avg_hours_3d_detail,
                        avg_hours_2d,
                        avg_hours_total,
                        occurrences,
                        confidence,
                        source,
                        last_updated
                    FROM component_patterns
                    WHERE {where_sql}
                    ORDER BY department, occurrences DESC
                """, params)
                rows = cur.fetchall()

        if not rows:
            st.info("🔍 Brak wzorców do exportu")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=[
            "Nazwa", "Pattern Key", "Dział",
            "Śr. Layout [h]", "Śr. Detail [h]", "Śr. 2D [h]", "Śr. Total [h]",
            "Obserwacje", "Confidence", "Źródło", "Ostatnia aktualizacja"
        ])

        # Format
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        df["Ostatnia aktualizacja"] = pd.to_datetime(df["Ostatnia aktualizacja"], errors='coerce').dt.strftime("%Y-%m-%d %H:%M")

        st.write(f"**Liczba wzorców do exportu:** {len(df)}")

        # Export buttons
        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Pobierz CSV",
                data=csv,
                file_name=f"wzorce_cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_patterns_csv"
            )

        with col2:
            # Excel export
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Wzorce', index=False)
            buffer.seek(0)

            st.download_button(
                label="📥 Pobierz Excel",
                data=buffer,
                file_name=f"wzorce_cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="export_patterns_excel"
            )

    except Exception as e:
        st.error(f"❌ Błąd exportu wzorców: {e}")
