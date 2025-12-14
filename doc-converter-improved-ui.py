# ULEPSZONE UI DLA DOC-CONVERTER SIDEBAR
# Wersja z lepszymi opisami, tooltipami i grupowaniem

import streamlit as st

with st.sidebar:
    st.header("⚙️ Ustawienia")

    # ============================================================================
    # STATUS USŁUG - Zwinięty domyślnie
    # ============================================================================
    with st.expander("🔌 Status usług (kliknij aby rozwinąć)", expanded=False):
        st.caption("📊 **Status połączeń z lokalnymi usługami:**")

        def _status_url(name, url, description):
            try:
                host = urlparse(url).hostname or ""
                is_local = is_private_host(host)
                status_icon = "✅" if is_local else "❌"
                status_text = "lokalny" if is_local else "zewnętrzny"
                st.caption(f"{status_icon} **{name}** - {description}")
                st.caption(f"   └─ `{url}` ({status_text})")
            except Exception:
                st.caption(f"⚠️ **{name}** - nie można zweryfikować")

        _status_url("Ollama", OLLAMA_URL, "AI models (LLM)")
        _status_url("Whisper", WHISPER_URL, "Transkrypcja audio")
        _status_url("Pyannote", PYANNOTE_URL, "Rozpoznawanie mówców")

    st.markdown("---")

    # ============================================================================
    # MODELE AI - Z WYJAŚNIENIAMI
    # ============================================================================
    st.subheader("🤖 Modele AI")

    # Help text dla modeli
    with st.expander("ℹ️ Co to są modele AI?", expanded=False):
        st.markdown("""
        **Modele AI** to "mózgi" aplikacji, które przetwarzają tekst i obrazy.

        - **Model tekstowy** - analizuje i podsumowuje dokumenty
        - **Model wizyjny** - rozpoznaje treść na obrazach/zdjęciach

        💡 **Porada:** Większe modele (np. 14b) są dokładniejsze ale wolniejsze.
        Mniejsze (7b) są szybsze ale mniej dokładne.
        """)

    # 1) Model tekstowy
    available_text_models = [
        m for m in list_ollama_models()
        if not any(m.startswith(p) for p in ("llava", "bakllava", "moondream", "qwen2-vl", "qwen2.5vl", "nomic-embed"))
    ]

    if "selected_main_text_model" not in st.session_state:
        default_text = "qwen2.5:14b" if "qwen2.5:14b" in available_text_models else (
            available_text_models[0] if available_text_models else "llama3:latest"
        )
        st.session_state["selected_main_text_model"] = default_text

    try:
        text_idx = available_text_models.index(st.session_state["selected_main_text_model"])
    except (ValueError, IndexError):
        text_idx = 0

    main_text_model = st.selectbox(
        "📝 Model tekstowy",
        options=available_text_models or ["llama3:latest"],
        index=text_idx,
        key="main_text_sel",
        help="""
        **Używany do:**
        • Podsumowań dokumentów
        • Analizy tekstów
        • Web search (jeśli włączony)
        • Project Brain

        **Rekomendacja:** qwen2.5:14b (dokładny) lub llama3 (szybki)
        """,
        disabled=st.session_state.get("converting", False)
    )
    st.session_state["selected_main_text_model"] = main_text_model

    st.markdown("---")

    # ============================================================================
    # TRYB PRACY - Prywatność i Internet
    # ============================================================================
    st.subheader("🔒 Prywatność i Internet")

    with st.expander("ℹ️ Co to znaczy?", expanded=False):
        st.markdown("""
        **Tryb offline** - blokuje wszystkie połączenia internetowe poza lokalnymi usługami.

        **Web lookup** - pozwala aplikacji pobierać publiczne strony WWW dla uzupełnienia informacji.

        ⚠️ **WAŻNE:** Aplikacja NIE wysyła Twoich dokumentów na zewnątrz!
        Web lookup pobiera TYLKO publiczne strony (np. Wikipedia) jako kontekst.
        """)

    OFFLINE_MODE = st.checkbox(
        "🔐 Tryb offline (maksymalna prywatność)",
        value=OFFLINE_MODE,
        help="Blokuje dostęp do internetu. Używa tylko lokalnych usług.",
        disabled=st.session_state.get("converting", False)
    )

    st.session_state["ALLOW_WEB"] = st.checkbox(
        "🌐 Web lookup (pobieranie publicznych stron)",
        value=st.session_state.get("ALLOW_WEB", True),
        help="""
        Pozwala aplikacji pobierać publiczne strony WWW dla weryfikacji informacji.

        ✅ NIE wysyła Twoich dokumentów na zewnątrz
        ✅ Pobiera tylko publiczne dane (Wikipedia, dokumentacja)
        ✅ Używane tylko dla Vision: "opisz obraz"
        """,
        disabled=st.session_state.get("converting", False) or OFFLINE_MODE
    )

    if st.session_state.get("ALLOW_WEB", False):
        st.info("🔍 Web search aktywny - Vision może weryfikować opisy obrazów")
    else:
        st.success("🔒 Web search wyłączony - maksymalna prywatność")

    st.markdown("---")

    # ============================================================================
    # VISION - Analiza obrazów
    # ============================================================================
    st.subheader("👁️ Vision (analiza obrazów)")

    with st.expander("ℹ️ Co to jest Vision?", expanded=False):
        st.markdown("""
        **Vision** to AI który "widzi" obrazy i potrafi je opisać lub przeczytać tekst z nich.

        **Tryby pracy:**
        - **OCR** - tylko rozpoznawanie tekstu (Tesseract)
        - **Vision: przepisz tekst** - AI czyta tekst z obrazu (lepsze od OCR)
        - **Vision: opisz obraz** - AI opisuje CO WIDZI na obrazie
        - **OCR + Vision** - oba razem

        💡 **Użyj Vision gdy:**
        • Masz zdjęcia/schematy/rysunki
        • OCR nie radzi sobie z tekstem
        • Chcesz opis zawartości obrazu
        """)

    vision_models = list_vision_models()
    use_vision = st.checkbox(
        "✨ Włącz Vision (AI dla obrazów)",
        value=True if vision_models else False,
        help="Używa AI do analizy obrazów, zdjęć, schematów, rysunków technicznych",
        disabled=st.session_state.get("converting", False)
    )

    if vision_models and use_vision:
        if "selected_vision_model" not in st.session_state:
            default_vision = "qwen2.5vl:7b"
            st.session_state["selected_vision_model"] = (
                default_vision if default_vision in vision_models else
                next((m for m in vision_models if m.startswith("qwen")), vision_models[0])
            )

        try:
            vision_idx = vision_models.index(st.session_state["selected_vision_model"])
        except (ValueError, IndexError):
            vision_idx = 0
            st.session_state["selected_vision_model"] = vision_models[0]

        selected_vision = st.selectbox(
            "Model Vision",
            vision_models,
            index=vision_idx,
            key="vision_model_sel",
            help="qwen2.5vl:7b - najlepszy do dokumentów technicznych",
            disabled=st.session_state.get("converting", False)
        )
        st.session_state["selected_vision_model"] = selected_vision

        # Tryb dla obrazów
        if "image_mode_idx" not in st.session_state:
            st.session_state["image_mode_idx"] = 2  # "Vision: opisz obraz"

        image_mode_label = st.selectbox(
            "Tryb pracy dla obrazów",
            options=["OCR", "Vision: przepisz tekst", "Vision: opisz obraz", "OCR + Vision opis"],
            index=st.session_state["image_mode_idx"],
            key="img_mode_sel",
            help="""
            • OCR - szybki, tylko tekst
            • Vision: przepisz tekst - AI czyta (lepsze od OCR)
            • Vision: opisz obraz - AI opisuje co widzi (POLECANE)
            • OCR + Vision - oba razem (najdokładniejsze)
            """,
            disabled=st.session_state.get("converting", False)
        )
        st.session_state["image_mode_idx"] = ["OCR", "Vision: przepisz tekst", "Vision: opisz obraz", "OCR + Vision opis"].index(image_mode_label)
        image_mode = IMAGE_MODE_MAP.get(image_mode_label, "ocr")
    else:
        selected_vision = None
        image_mode = "ocr"
        if use_vision:
            st.warning("⚠️ Brak modeli Vision. Zainstaluj: `ollama pull llava:13b`")

    st.markdown("---")

    # ============================================================================
    # OPCJE ZAAWANSOWANE - Zwinięte domyślnie
    # ============================================================================
    with st.expander("🔧 Opcje zaawansowane", expanded=False):
        st.subheader("OCR (rozpoznawanie tekstu)")
        st.caption("Tesseract OCR - dla PDF-ów skanowanych i obrazów z tekstem")

        ocr_pages_limit = st.slider(
            "Limit stron OCR",
            min_value=5,
            max_value=50,
            value=20,
            help="Maksymalna liczba stron do przetworzenia przez OCR (duże PDFy mogą być wolne)",
            disabled=st.session_state.get("converting", False)
        )

        st.markdown("---")

        st.subheader("💾 Zapis lokalny")
        st.caption("Automatycznie zapisuj wyniki do plików na dysku")

        enable_local_save = st.checkbox(
            "Zapisz wyniki lokalnie",
            value=False,
            help="Wyniki będą zapisane w folderze na dysku (txt, json, md)",
            disabled=st.session_state.get("converting", False)
        )

        base_output_dir = st.text_input(
            "Katalog wyjściowy",
            value="outputs",
            help="Ścieżka do folderu gdzie zapisywać wyniki",
            disabled=st.session_state.get("converting", False) or not enable_local_save
        )

        st.markdown("---")

        st.subheader("📚 AnythingLLM")
        st.caption("Integracja z AnythingLLM dla zarządzania dokumentami")

        # ... reszta AnythingLLM config ...

    # ============================================================================
    # POMOC - Zawsze widoczna na dole
    # ============================================================================
    st.markdown("---")
    with st.expander("❓ Pomoc i podpowiedzi", expanded=False):
        st.markdown("""
        ### 🎯 Szybki start

        1. **Upload pliku** - PDF, Word, zdjęcie, audio
        2. **Kliknij "Konwertuj"**
        3. **Gotowe!**

        ### 💡 Wskazówki

        **Dla PDF tekstowych:**
        - Użyj domyślnych ustawień
        - Vision nie jest potrzebny

        **Dla skanów/zdjęć:**
        - Włącz Vision
        - Wybierz "Vision: opisz obraz"

        **Dla audio:**
        - Automatycznie używa Whisper (transkrypcja)
        - Pyannote rozpoznaje mówców (jeśli dostępny)

        ### 🔐 Prywatność

        ✅ Wszystko działa **lokalnie**
        ✅ Dokumenty **NIE są wysyłane** na zewnątrz
        ✅ Web lookup pobiera tylko **publiczne strony**

        ### 🆘 Problemy?

        Sprawdź "Status usług" powyżej - wszystkie powinny być zielone (✅).
        """)

# KONIEC ULEPSZONEGO UI
