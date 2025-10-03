import streamlit as st
import io
import base64
import re
import requests
import logging
import os
from datetime import datetime
from PIL import Image
import numpy as np

# Parsery
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from pptx import Presentation
from docx import Document
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc-converter")

st.set_page_config(page_title="📄 Document Converter", layout="wide", page_icon="📄")

# === CONFIG ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
ANYTHINGLLM_URL = os.getenv("ANYTHINGLLM_URL", "")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000")

# === HELPERY ===

def list_ollama_models():
    """Lista modeli z Ollama."""
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if r.ok:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        logger.error(f"Ollama connection error: {e}")
    return []

def list_vision_models():
    """Filtruj tylko modele wizyjne."""
    all_models = list_ollama_models()
    prefixes = ("llava", "bakllava", "moondream", "llava-phi")
    return [m for m in all_models if any(m.startswith(p) for p in prefixes)]

def query_ollama_vision(prompt: str, image_b64: str, model: str):
    """Zapytaj model wizyjny o obraz."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        logger.error(f"Vision model error: {e}")
        return f"[BŁĄD: {e}]"

def ocr_image_bytes(img_bytes: bytes, lang: str = 'pol+eng') -> str:
    """OCR Tesseract z preprocessingiem."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        np_img = np.array(img)
        _, thr = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_thr = Image.fromarray(thr)
        return pytesseract.image_to_string(img_thr, lang=lang) or ""
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""

def extract_audio_whisper(file) -> tuple[str, int]:
    """Audio → tekst przez Whisper ASR."""
    try:
        file.seek(0)
        files = {'audio_file': (file.name, file.read(), file.type)}
        r = requests.post(
            f"{WHISPER_URL}/asr",
            files=files,
            params={"task": "transcribe", "language": "pl"},
            timeout=120
        )
        r.raise_for_status()
        result = r.json()
        return result.get("text", ""), 1
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return f"[BŁĄD AUDIO: {e}]", 0

def extract_pdf(file, use_vision: bool, vision_model: str, ocr_pages_limit: int = 20):
    """PDF: tekst + opcjonalnie OCR/Vision."""
    texts = []
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= ocr_pages_limit:
                    texts.append(f"\n[... limit {ocr_pages_limit} stron ...]")
                    break
                t = page.extract_text() or ""
                texts.append(t)

        full_text = "\n".join(texts)

        if len(full_text.strip()) < 100:
            file.seek(0)
            images = convert_from_bytes(file.read(), fmt="jpeg", dpi=150, first_page=1, last_page=min(ocr_pages_limit, 10))

            if use_vision and vision_model:
                st.info(f"🖼️ Używam {vision_model} do analizy obrazów...")
                for idx, img in enumerate(images[:5], 1):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buf.getvalue()).decode()

                    prompt = """Przepisz DOKŁADNIE cały tekst z obrazu. Zachowaj pisównię, układ, symbole. Nie tłumacz, nie interpretuj - tylko przepisz. Jeśli nieczytelne - wpisz [NIECZYTELNE]."""

                    response = query_ollama_vision(prompt, img_b64, vision_model)
                    texts.append(f"\n--- Strona {idx} (Vision) ---\n{response}")
            else:
                st.info("📝 OCR Tesseract...")
                for idx, img in enumerate(images[:ocr_pages_limit], 1):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    ocr_text = ocr_image_bytes(buf.getvalue())
                    texts.append(f"\n--- Strona {idx} (OCR) ---\n{ocr_text}")

        return "\n".join(texts), len(texts)
    except Exception as e:
        logger.error(f"PDF extract error: {e}")
        return f"[BŁĄD PDF: {e}]", 0

def extract_pptx(file, use_vision: bool, vision_model: str):
    """PPTX: tekst + obrazy (opcjonalnie Vision)."""
    try:
        prs = Presentation(file)
        slides_text = []

        for i, slide in enumerate(prs.slides, 1):
            parts = [f"=== Slajd {i} ==="]

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)

            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text
                if notes:
                    parts.append(f"Notatki: {notes}")

            if use_vision and vision_model:
                for shape in slide.shapes:
                    if shape.shape_type == 13:
                        try:
                            img_stream = shape.image.blob
                            img_b64 = base64.b64encode(img_stream).decode()
                            prompt = "Opisz ten obraz ze slajdu: co widzisz? Jaki tekst, wykresy, diagramy?"
                            response = query_ollama_vision(prompt, img_b64, vision_model)
                            parts.append(f"[Obraz] {response}")
                        except:
                            pass

            slides_text.append("\n".join(parts))

        return "\n\n".join(slides_text), len(prs.slides)
    except Exception as e:
        logger.error(f"PPTX error: {e}")
        return f"[BŁĄD PPTX: {e}]", 0

def extract_docx(file):
    """DOCX: tekst + tabele."""
    try:
        doc = Document(file)
        paras = [p.text for p in doc.paragraphs if p.text]

        for tbl in doc.tables:
            for row in tbl.rows:
                paras.append(" | ".join(cell.text for cell in row.cells))

        return "\n".join(paras), len(paras)
    except Exception as e:
        logger.error(f"DOCX error: {e}")
        return f"[BŁĄD DOCX: {e}]", 0

def extract_image(file, use_vision: bool, vision_model: str):
    """Obraz: Vision lub OCR."""
    try:
        file.seek(0)
        img_bytes = file.read()

        if use_vision and vision_model:
            img_b64 = base64.b64encode(img_bytes).decode()
            prompt = """Przepisz DOKŁADNIE cały tekst z obrazu. Zachowaj pisównię, układ, symbole. Nie tłumacz, nie interpretuj - tylko przepisz. Jeśli nieczytelne - wpisz [NIECZYTELNE]."""
            text = query_ollama_vision(prompt, img_b64, vision_model)
        else:
            text = ocr_image_bytes(img_bytes)

        return text, 1
    except Exception as e:
        logger.error(f"Image error: {e}")
        return f"[BŁĄD IMG: {e}]", 0

def process_file(file, use_vision: bool, vision_model: str, ocr_limit: int):
    """Router do odpowiedniego ekstraktora."""
    name = file.name.lower()

    if name.endswith('.pdf'):
        return extract_pdf(file, use_vision, vision_model, ocr_limit)
    elif name.endswith(('.pptx', '.ppt')):
        return extract_pptx(file, use_vision, vision_model)
    elif name.endswith('.docx'):
        return extract_docx(file)
    elif name.endswith(('.jpg', '.jpeg', '.png')):
        return extract_image(file, use_vision, vision_model)
    elif name.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        return extract_audio_whisper(file)
    elif name.endswith('.txt'):
        file.seek(0)
        return file.read().decode('utf-8', errors='ignore'), 1
    else:
        return "[Nieobsługiwany format]", 0

def send_to_anythingllm(text: str, filename: str):
    """Wyślij dokument do AnythingLLM."""
    if not ANYTHINGLLM_URL or not ANYTHINGLLM_API_KEY:
        return False, "Brak konfiguracji AnythingLLM"

    try:
        headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}"}
        payload = {"name": filename, "content": text, "type": "text/plain"}

        r = requests.post(
            f"{ANYTHINGLLM_URL}/api/v1/document-upload",
            headers=headers,
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        return True, "✅ Wysłano do AnythingLLM"
    except Exception as e:
        return False, f"Błąd AnythingLLM: {e}"

# === UI ===

st.title("📄 Document Converter Pro")
st.caption("Konwersja PDF/DOCX/PPTX/IMG/AUDIO → TXT z OCR, Vision lub Whisper")

with st.sidebar:
    st.header("⚙️ Ustawienia")

    vision_models = list_vision_models()
    use_vision = st.checkbox("Użyj modelu wizyjnego", value=True if vision_models else False)

    if vision_models:
        selected_vision = st.selectbox("Model wizyjny", vision_models, index=0)
    else:
        selected_vision = None
        st.warning("⚠️ Brak modeli Vision w Ollama\nZainstaluj: `ollama pull llava:13b`")

    st.subheader("OCR")
    ocr_pages_limit = st.slider("Limit stron OCR", 5, 50, 20)

    st.subheader("AnythingLLM")
    has_anythingllm = bool(ANYTHINGLLM_URL and ANYTHINGLLM_API_KEY)
    st.caption(f"Status: {'✅ Skonfigurowane' if has_anythingllm else '❌ Brak config'}")

uploaded_files = st.file_uploader(
    "Wgraj dokumenty",
    type=['pdf', 'docx', 'pptx', 'ppt', 'jpg', 'jpeg', 'png', 'txt', 'mp3', 'wav', 'm4a', 'ogg', 'flac'],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} plików")

    if st.button("🚀 Konwertuj wszystkie", type="primary"):
        all_texts = []
        stats = {'processed': 0, 'errors': 0, 'pages': 0}

        progress = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            progress.progress((idx + 1) / len(uploaded_files), text=f"Przetwarzam: {file.name}")

            st.subheader(f"📄 {file.name}")

            try:
                text, pages = process_file(file, use_vision, selected_vision, ocr_pages_limit)

                all_texts.append(f"\n{'='*80}\n")
                all_texts.append(f"PLIK: {file.name}\n")
                all_texts.append(f"Typ: {file.type}, Rozmiar: {file.size/1024:.1f} KB\n")
                all_texts.append(f"{'='*80}\n")
                all_texts.append(text)
                all_texts.append(f"\n[Stron/sekcji: {pages}]\n")

                stats['processed'] += 1
                stats['pages'] += pages

                with st.expander(f"Preview: {file.name}"):
                    st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

            except Exception as e:
                st.error(f"❌ Błąd: {e}")
                stats['errors'] += 1

        progress.empty()

        st.success(f"✅ Przetworzono: {stats['processed']}/{len(uploaded_files)}")
        st.metric("Strony/sekcje", stats['pages'])

        combined_text = "\n".join(all_texts)

        st.download_button(
            "⬇️ Pobierz TXT",
            combined_text.encode('utf-8'),
            file_name=f"converted_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

        if has_anythingllm:
            if st.button("📤 Wyślij do AnythingLLM"):
                success, msg = send_to_anythingllm(combined_text, "converted_docs.txt")
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
