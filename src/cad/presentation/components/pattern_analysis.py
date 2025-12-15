"""
CAD Estimator Pro - Pattern Analysis Components

UI components for pattern and bundle analysis.
"""
import streamlit as st
import pandas as pd
from typing import Any


def render_pattern_search(app: dict):
    """
    Render pattern search and display.

    Args:
        app: App context
    """
    st.subheader("🔍 Wyszukiwanie wzorców")

    # Search input
    search_query = st.text_input(
        "Szukaj wzorca",
        placeholder="Np. wspornik, śruba, łącznik...",
        help="Wpisz nazwę komponentu do wyszukania"
    )

    # Department filter
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox(
            "Dział",
            options=["Wszystkie", "131", "132", "133", "134", "135"],
            index=0
        )

    with col2:
        min_confidence = st.slider(
            "Min. confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Filtruj wzorce według minimalnego poziomu pewności"
        )

    if search_query:
        try:
            # Build query
            where_clauses = ["LOWER(name) LIKE %s"]
            params = [f"%{search_query.lower()}%"]

            if department != "Wszystkie":
                where_clauses.append("department = %s")
                params.append(department)

            where_clauses.append("confidence >= %s")
            params.append(min_confidence)

            where_sql = " AND ".join(where_clauses)

            # Query
            with app["db"].get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT
                            name,
                            department,
                            avg_hours_3d_layout,
                            avg_hours_3d_detail,
                            avg_hours_2d,
                            avg_hours_total,
                            occurrences,
                            confidence,
                            source
                        FROM component_patterns
                        WHERE {where_sql}
                        ORDER BY occurrences DESC, confidence DESC
                        LIMIT 50
                    """, params)
                    rows = cur.fetchall()

            if not rows:
                st.info(f"🔍 Nie znaleziono wzorców dla '{search_query}'")
                return

            # Display results
            st.write(f"**Znaleziono:** {len(rows)} wzorców")

            df = pd.DataFrame(rows, columns=[
                "Nazwa", "Dział", "Layout [h]", "Detail [h]", "2D [h]",
                "Total [h]", "Obserwacje", "Confidence", "Źródło"
            ])

            # Format columns
            df["Layout [h]"] = df["Layout [h]"].apply(lambda x: f"{x:.1f}")
            df["Detail [h]"] = df["Detail [h]"].apply(lambda x: f"{x:.1f}")
            df["2D [h]"] = df["2D [h]"].apply(lambda x: f"{x:.1f}")
            df["Total [h]"] = df["Total [h]"].apply(lambda x: f"{x:.1f}")
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.0f}%")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Nazwa": st.column_config.TextColumn(width="large"),
                    "Dział": st.column_config.TextColumn(width="small"),
                    "Layout [h]": st.column_config.TextColumn(width="small"),
                    "Detail [h]": st.column_config.TextColumn(width="small"),
                    "2D [h]": st.column_config.TextColumn(width="small"),
                    "Total [h]": st.column_config.TextColumn(width="small"),
                    "Obserwacje": st.column_config.NumberColumn(width="small"),
                    "Confidence": st.column_config.TextColumn(width="small"),
                    "Źródło": st.column_config.TextColumn(width="small"),
                }
            )

        except Exception as e:
            st.error(f"❌ Błąd wyszukiwania: {e}")


def render_top_patterns(app: dict, limit: int = 20):
    """
    Render top patterns by occurrences.

    Args:
        app: App context
        limit: Max patterns to show
    """
    st.subheader("🏆 Najczęstsze wzorce")

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
                        source
                    FROM component_patterns
                    WHERE occurrences > 2
                    ORDER BY occurrences DESC, confidence DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

        if not rows:
            st.info("📊 Brak wzorców do wyświetlenia")
            return

        df = pd.DataFrame(rows, columns=[
            "Nazwa", "Dział", "Śr. godziny", "Obserwacje", "Confidence", "Źródło"
        ])

        # Format
        df["Śr. godziny"] = df["Śr. godziny"].apply(lambda x: f"{x:.1f}h")
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.0f}%")

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
            }
        )

    except Exception as e:
        st.error(f"❌ Błąd: {e}")


def render_low_confidence_patterns(app: dict, threshold: float = 0.5, limit: int = 20):
    """
    Render patterns with low confidence (need more data).

    Args:
        app: App context
        threshold: Confidence threshold
        limit: Max patterns to show
    """
    st.subheader("⚠️ Wzorce wymagające więcej danych")

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
                        source
                    FROM component_patterns
                    WHERE confidence < %s
                    ORDER BY occurrences ASC, confidence ASC
                    LIMIT %s
                """, (threshold, limit))
                rows = cur.fetchall()

        if not rows:
            st.success("✅ Wszystkie wzorce mają wystarczającą liczbę obserwacji!")
            return

        st.warning(
            f"⚠️ **{len(rows)} wzorców** ma niski confidence (< {threshold*100:.0f}%).\n\n"
            "System potrzebuje więcej danych historycznych dla tych komponentów, "
            "aby poprawić dokładność predykcji."
        )

        df = pd.DataFrame(rows, columns=[
            "Nazwa", "Dział", "Śr. godziny", "Obserwacje", "Confidence", "Źródło"
        ])

        # Format
        df["Śr. godziny"] = df["Śr. godziny"].apply(lambda x: f"{x:.1f}h")
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.0f}%")

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
            }
        )

    except Exception as e:
        st.error(f"❌ Błąd: {e}")


def render_bundle_analysis(app: dict):
    """
    Render component bundle analysis.

    Args:
        app: App context
    """
    st.subheader("🔗 Analiza relacji komponentów (Bundles)")

    st.info(
        "💡 **Bundles** to typowe relacje parent→sub.\n\n"
        "Np. 'Wspornik' często występuje z 'Śrubą M12' (2-4 szt.)"
    )

    # Search parent
    parent_search = st.text_input(
        "Szukaj komponentu nadrzędnego",
        placeholder="Np. wspornik, rama, obudowa...",
        help="Wpisz nazwę komponentu, dla którego chcesz zobaczyć typowe sub-komponenty"
    )

    if parent_search:
        try:
            with app["db"].get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT
                            parent_name,
                            parent_key,
                            department
                        FROM component_bundles
                        WHERE LOWER(parent_name) LIKE %s
                        ORDER BY parent_name
                        LIMIT 10
                    """, (f"%{parent_search.lower()}%",))
                    parents = cur.fetchall()

            if not parents:
                st.info(f"🔍 Nie znaleziono bundles dla '{parent_search}'")
                return

            # Select parent
            parent_options = [f"{p[0]} ({p[2]})" for p in parents]
            selected = st.selectbox("Wybierz komponent", options=parent_options)

            if selected:
                # Parse selection
                parent_name = selected.rsplit(" (", 1)[0]
                department = selected.rsplit(" (", 1)[1].rstrip(")")

                # Get bundles
                with app["db"].get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT
                                sub_name,
                                total_qty,
                                occurrences,
                                confidence
                            FROM component_bundles
                            WHERE parent_name = %s AND department = %s
                            ORDER BY occurrences DESC, confidence DESC
                        """, (parent_name, department))
                        bundles = cur.fetchall()

                if bundles:
                    st.write(f"**Typowe sub-komponenty dla:** {parent_name}")

                    df = pd.DataFrame(bundles, columns=[
                        "Sub-komponent", "Suma ilości", "Obserwacje", "Confidence"
                    ])

                    # Calculate avg quantity
                    df["Śr. ilość"] = (df["Suma ilości"] / df["Obserwacje"]).apply(lambda x: f"{x:.1f}")
                    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.0f}%")
                    df = df[["Sub-komponent", "Śr. ilość", "Obserwacje", "Confidence"]]

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Sub-komponent": st.column_config.TextColumn(width="large"),
                            "Śr. ilość": st.column_config.TextColumn(width="small"),
                            "Obserwacje": st.column_config.NumberColumn(width="small"),
                            "Confidence": st.column_config.TextColumn(width="small"),
                        }
                    )
                else:
                    st.info("Brak sub-komponentów dla tego elementu")

        except Exception as e:
            st.error(f"❌ Błąd: {e}")


def render_top_bundles(app: dict, limit: int = 20):
    """
    Render most common bundles.

    Args:
        app: App context
        limit: Max bundles to show
    """
    st.subheader("🏆 Najczęstsze relacje (Bundles)")

    try:
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        parent_name,
                        sub_name,
                        department,
                        total_qty,
                        occurrences,
                        confidence
                    FROM component_bundles
                    WHERE occurrences > 2
                    ORDER BY occurrences DESC, confidence DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

        if not rows:
            st.info("📊 Brak bundles do wyświetlenia")
            return

        df = pd.DataFrame(rows, columns=[
            "Parent", "Sub-komponent", "Dział", "Suma ilości", "Obserwacje", "Confidence"
        ])

        # Calculate avg quantity
        df["Śr. ilość"] = (df["Suma ilości"] / df["Obserwacje"]).apply(lambda x: f"{x:.1f}")
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x*100:.0f}%")
        df = df[["Parent", "Sub-komponent", "Dział", "Śr. ilość", "Obserwacje", "Confidence"]]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Parent": st.column_config.TextColumn(width="large"),
                "Sub-komponent": st.column_config.TextColumn(width="large"),
                "Dział": st.column_config.TextColumn(width="small"),
                "Śr. ilość": st.column_config.TextColumn(width="small"),
                "Obserwacje": st.column_config.NumberColumn(width="small"),
                "Confidence": st.column_config.TextColumn(width="small"),
            }
        )

    except Exception as e:
        st.error(f"❌ Błąd: {e}")
