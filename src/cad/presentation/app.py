"""
CAD Estimator Pro - Main Streamlit Application

Main entry point for CAD Estimator Pro Streamlit UI.
"""
import streamlit as st
import logging
from typing import Any

from cad.domain.models.config import AppConfig
from cad.infrastructure.factory import (
    create_database_client,
    create_ai_client,
    create_excel_parser,
    create_pdf_parser,
    create_component_parser,
)
from cad.infrastructure.learning.pattern_learner import PatternLearner
from cad.infrastructure.learning.bundle_learner import BundleLearner
from cad.infrastructure.embeddings.pgvector_service import PgVectorService
from cad.infrastructure.multi_model import MultiModelOrchestrator
from cad.application.estimation_pipeline import EstimationPipeline
from cad.application.batch_importer import BatchImporter
from cad.presentation.state.session_manager import SessionManager
from cad.presentation.components.sidebar import render_sidebar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CAD Estimator Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_app() -> dict[str, Any]:
    """
    Initialize application (cached).

    Returns:
        Dict with initialized components
    """
    logger.info("🚀 Initializing CAD Estimator Pro...")

    # Load configuration
    config = AppConfig.from_env()

    # Create core services
    db = create_database_client(config)
    ai = create_ai_client(config)

    # Initialize schema
    try:
        db.init_schema()
        logger.info("✅ Database schema initialized")
    except Exception as e:
        logger.error(f"❌ Schema initialization failed: {e}")
        st.error(f"Database initialization failed: {e}")

    # Create parsers
    excel_parser = create_excel_parser(config)
    pdf_parser = create_pdf_parser(config)
    component_parser = create_component_parser(config)

    # Create learning components
    pattern_learner = PatternLearner(config.learning, db)
    bundle_learner = BundleLearner(config.learning, db)

    # Create embedding service
    pgvector_service = PgVectorService(db, ai)

    # Create multi-model orchestrator
    multi_model = MultiModelOrchestrator(ai, db, config.multi_model)

    # Create pipeline
    pipeline = EstimationPipeline(
        config=config,
        db_client=db,
        ai_client=ai,
        excel_parser=excel_parser,
        pdf_parser=pdf_parser,
        component_parser=component_parser,
        pattern_learner=pattern_learner,
        bundle_learner=bundle_learner,
        pgvector_service=pgvector_service,
        multi_model_orchestrator=multi_model,
    )

    # Create batch importer
    batch_importer = BatchImporter(
        config=config,
        db_client=db,
        excel_parser=excel_parser,
        pattern_learner=pattern_learner,
        bundle_learner=bundle_learner,
    )

    logger.info("✅ Initialization complete (multi-model pipeline ready)")

    return {
        "config": config,
        "db": db,
        "ai": ai,
        "pipeline": pipeline,
        "batch_importer": batch_importer,
        "pattern_learner": pattern_learner,
        "bundle_learner": bundle_learner,
        "pgvector": pgvector_service,
        "multi_model": multi_model,
    }


def main():
    """Main application entry point."""
    st.title("🚀 CAD Estimator Pro")

    # Initialize app
    try:
        app = init_app()
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        logger.error(f"Initialization failed: {e}", exc_info=True)
        st.stop()

    # Initialize session manager
    session = SessionManager(st.session_state)

    # Get available models
    try:
        all_models = app["ai"].list_available_models()
        text_models = [
            m
            for m in all_models
            if not any(
                m.startswith(p)
                for p in (
                    "llava",
                    "bakllava",
                    "moondream",
                    "qwen2-vl",
                    "qwen2.5vl",
                    "nomic-embed",
                )
            )
        ]
        vision_models = [
            m
            for m in all_models
            if any(
                m.startswith(p)
                for p in ("llava", "bakllava", "moondream", "qwen2-vl", "qwen2.5vl")
            )
        ]
    except Exception as e:
        logger.warning(f"Failed to list models: {e}")
        text_models = ["llama3:latest"]
        vision_models = []

    # Navigation FIRST - menu at the top of sidebar
    st.sidebar.title("📋 Menu")
    page = st.sidebar.radio(
        "Nawigacja",
        ["📊 Dashboard", "🆕 Nowy projekt", "📚 Historia i Uczenie", "🛠️ Admin"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Render sidebar configuration BELOW the menu
    sidebar_config = render_sidebar(
        session=session,
        app_config=app['config'],
        available_text_models=text_models,
        available_vision_models=vision_models
    )

    # Routing
    if "Dashboard" in page:
        render_dashboard_page(app, session)
    elif "Nowy projekt" in page:
        render_new_project_page(app, session, sidebar_config)
    elif "Historia" in page:
        render_history_page(app, session)
    elif "Admin" in page:
        render_admin_page(app, session)


# ==================== PAGES ====================


def render_dashboard_page(app: dict, session: SessionManager):
    """Render Dashboard page."""
    st.header("📊 Dashboard")

    # Quick stats
    try:
        with app["db"].get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM projects")
                project_count = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM component_patterns WHERE occurrences > 2"
                )
                pattern_count = cur.fetchone()[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("📁 Projekty", project_count)
        col2.metric("🧩 Wzorce", pattern_count)
        col3.metric("🤖 Status AI", "✅ Online" if app["ai"] else "❌ Offline")

    except Exception as e:
        st.error(f"Błąd pobierania statystyk: {e}")

    st.info("💡 Dashboard w pełni funkcjonalny będzie dostępny w kolejnej iteracji")


def render_new_project_page(app: dict, session: SessionManager, config: dict):
    """Render New Project page."""
    st.header("🆕 Nowy Projekt")

    from cad.presentation.components.file_uploader import render_file_uploader, render_text_input
    from cad.presentation.components.sidebar import render_department_selector

    # Department selection
    department = render_department_selector()

    # Project details
    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input(
            "Nazwa projektu*", placeholder="np. Stacja dociskania omega"
        )
        client = st.text_input("Klient", placeholder="np. Firma sp. z o.o.")

    # Description and files
    description, additional_text = render_text_input()
    files = render_file_uploader()

    # Estimate button
    if st.button("🤖 Analizuj z AI", use_container_width=True, type="primary"):
        if not description and not files["excel"]:
            st.warning("⚠️ Podaj opis lub wgraj plik Excel")
        else:
            # Check if multi-model is enabled
            use_multi_model = config.get("use_multi_model", False)

            if use_multi_model:
                # Multi-model with progress tracking
                from cad.presentation.components.progress_tracker import (
                    render_progress_placeholder,
                    ProgressTracker,
                )

                progress_placeholder = render_progress_placeholder()
                tracker = ProgressTracker(progress_placeholder)  # na przyszłość, gdy podepniesz callbacki

                try:
                    # Combine texts
                    full_text = description
                    if additional_text:
                        full_text += "\n\n" + additional_text

                    # Pokaż komunikat, że ruszył 4‑etapowy pipeline
                    progress_placeholder.info(
                        "⏳ Uruchomiono Multi‑Model Pipeline (4 etapy: "
                        "analiza techniczna → struktura → godziny → ryzyka)..."
                    )

                    # Spinner na czas pracy całego pipeline'u
                    with st.spinner("Analizuję projekt (4‑etapowy Multi‑Model Pipeline)..."):
                        estimate = app["pipeline"].estimate_from_description(
                            description=full_text,
                            department=department,
                            pdf_files=files["pdfs"],
                            excel_file=files["excel"],
                            use_multi_model=True,
                            stage1_model=config.get("stage1_model"),
                            stage2_model=config.get("stage2_model"),
                            stage3_model=config.get("stage3_model"),
                            stage4_model=config.get("stage4_model"),
                        )

                    # Clear progress, show success
                    progress_placeholder.empty()
                    session.set_estimate(estimate)
                    st.success(
                        f"✅ Multi‑Model Pipeline zakończony: "
                        f"{estimate.total_hours:.1f}h, {estimate.component_count} komponentów"
                    )

                    # Display enhanced results
                    from cad.presentation.components.multi_model_results import (
                        render_multi_model_results,
                    )

                    render_multi_model_results(estimate, config["hourly_rate"])

                    # Also show standard component list
                    st.markdown("---")
                    st.markdown("### 📋 Lista Komponentów (szczegóły)")
                    from cad.presentation.components.results_display import (
                        render_components_list,
                    )

                    render_components_list(estimate)

                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"❌ Multi‑Model Pipeline nie powiódł się: {e}")
                    logger.error(f"Multi-model estimation failed: {e}", exc_info=True)

            else:
                # Single-model with spinner
                with st.spinner("Analizuję projekt (single‑model)..."):
                    try:
                        # Combine texts
                        full_text = description
                        if additional_text:
                            full_text += "\n\n" + additional_text

                        # Estimate (single-model)
                        estimate = app["pipeline"].estimate_from_description(
                            description=full_text,
                            department=department,
                            pdf_files=files["pdfs"],
                            excel_file=files["excel"],
                            use_multi_model=False,
                        )

                        # Save to session
                        session.set_estimate(estimate)

                        st.success(
                            f"✅ Analiza zakończona: "
                            f"{estimate.total_hours:.1f}h, {estimate.component_count} komponentów"
                        )

                        # Display results
                        from cad.presentation.components.results_display import (
                            render_estimate_summary,
                            render_components_list,
                        )

                        render_estimate_summary(estimate, config["hourly_rate"])
                        st.markdown("---")
                        render_components_list(estimate)

                    except Exception as e:
                        st.error(f"❌ Analiza nie powiodła się: {e}")
                        logger.error(f"Estimation failed: {e}", exc_info=True)

def render_history_page(app: dict, session: SessionManager):
    """Render History & Learning page."""
    from cad.presentation.components.project_history import (
        render_project_filters,
        render_projects_table,
        render_project_details,
        render_accuracy_chart,
        render_export_projects,
        render_export_patterns,
    )
    from cad.presentation.components.learning import (
        render_add_actual_hours,
        render_learning_stats,
        render_pattern_improvements,
        render_batch_import,
    )
    from cad.presentation.components.pattern_analysis import (
        render_pattern_search,
        render_top_patterns,
        render_low_confidence_patterns,
        render_bundle_analysis,
        render_top_bundles,
    )

    st.header("📚 Historia i Uczenie")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Historia projektów",
        "🧠 Uczenie",
        "🔍 Wzorce",
        "🔗 Bundles",
        "📥 Export"
    ])

    with tab1:
        st.subheader("📁 Historia projektów")

        project_id_input = st.number_input(
            "ID projektu (szczegóły)",
            min_value=1,
            value=None,
            step=1,
            help="Wpisz ID projektu, aby zobaczyć szczegóły",
            key="history_project_details_id",
        )

        if project_id_input:
            render_project_details(app, project_id_input)
            st.markdown("---")

        # Filters dla tabeli i wykresu
        filters = render_project_filters(key_prefix="history_main")

        render_projects_table(app, filters)
        st.markdown("---")
        render_accuracy_chart(app, filters)

    with tab2:
        st.subheader("🧠 System uczenia")

        # Stats
        render_learning_stats(app)

        st.markdown("---")

        # Add actual hours
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### ⏱️ Dodaj actual hours")
            project_id_for_feedback = st.number_input(
                "ID projektu",
                min_value=1,
                value=None,
                step=1,
                key="feedback_project_id",
                help="Wpisz ID projektu, aby dodać rzeczywiste godziny"
            )

            if project_id_for_feedback:
                render_add_actual_hours(app, project_id_for_feedback)

        with col2:
            st.markdown("### 🔄 Ostatnio zaktualizowane")
            render_pattern_improvements(app, limit=10)

        st.markdown("---")

        # Batch import
        render_batch_import(app)

    with tab3:
        st.subheader("🔍 Analiza wzorców")

        # Search
        render_pattern_search(app)

        st.markdown("---")

        # Top patterns
        col1, col2 = st.columns(2)

        with col1:
            render_top_patterns(app, limit=15)

        with col2:
            render_low_confidence_patterns(app, threshold=0.5, limit=15)

    with tab4:
        st.subheader("🔗 Analiza relacji (Bundles)")

        # Bundle search
        render_bundle_analysis(app)

        st.markdown("---")

        # Top bundles
        render_top_bundles(app, limit=20)

    with tab5:
        st.subheader("📥 Export danych")

        st.info(
            "💡 **Export danych do CSV/Excel**\n\n"
            "Możesz wyeksportować:\n"
            "- Projekty (z filtrami)\n"
            "- Wzorce komponentów (według działu)"
        )

        # Osobne filtry dla exportu (inny prefix kluczy!)
        filters_for_export = render_project_filters(key_prefix="history_export")

        st.markdown("---")
        render_export_projects(app, filters_for_export)

        st.markdown("---")
        render_export_patterns(app)

def render_admin_page(app: dict, session: SessionManager):
    """Render Admin page."""
    st.header("🛠️ Panel Administratora")

    # Simple authentication
    if not session.is_admin_authenticated():
        password = st.text_input("Hasło administratora", type="password")
        if st.button("Zaloguj"):
            if password == "polmic":  # CHANGE IN PRODUCTION!
                session.set_admin_authenticated(True)
                st.rerun()
            else:
                st.error("❌ Błędne hasło")
        st.stop()

    st.success("✅ Zalogowano jako Administrator")

    # Admin actions
    tab1, tab2, tab3 = st.tabs(["📊 Statystyki", "🗑️ Czyszczenie", "🔄 Embeddings"])

    with tab1:
        st.subheader("📊 Statystyki bazy")
        try:
            with app["db"].get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM projects")
                    projects = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM component_patterns")
                    patterns = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM component_bundles")
                    bundles = cur.fetchone()[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Projekty", projects)
            col2.metric("Wzorce", patterns)
            col3.metric("Bundles", bundles)

        except Exception as e:
            st.error(f"Błąd: {e}")

    with tab2:
        st.subheader("🗑️ Czyszczenie danych")
        st.warning("⚠️ Operacje nieodwracalne!")

        if st.button("🗑️ Usuń wzorce z confidence < 0.1"):
            try:
                deleted = app["db"].delete_patterns_with_low_confidence(0.1)
                st.success(f"✅ Usunięto {deleted} wzorców")
            except Exception as e:
                st.error(f"Błąd: {e}")

    with tab3:
        st.subheader("🔄 Przelicz embeddingi")
        if st.button("🔄 Przelicz wszystkie embeddingi"):
            with st.spinner("Przeliczam embeddingi..."):
                try:
                    counts = app["pgvector"].batch_update_embeddings(
                        update_projects=True,
                        update_patterns=True,
                    )
                    st.success(
                        f"✅ Przeliczono {counts['projects']} projektów i {counts['patterns']} wzorców"
                    )
                except Exception as e:
                    st.error(f"Błąd: {e}")


if __name__ == "__main__":
    main()
