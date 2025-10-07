import streamlit as st
import io
import base64
import re
import requests
import logging
import os
import json
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

# --- DODATKOWE IMPORTY DO PODSUMOWAŃ AUDIO ---
from typing import List, Dict, Any, Tuple

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc-converter")

st.set_page_config(page_title="📄 Document Converter", layout="wide", page_icon="📄")

# === CONFIG ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
ANYTHINGLLM_URL = os.getenv("ANYTHINGLLM_URL", "http://anythingllm:3001")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "TC9T0P1-XBQ4ATS-QYSXZG8-RMFFVH6")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000")
PYANNOTE_URL = os.getenv("PYANNOTE_URL", "http://pyannote:8000")

# === STAŁE ===
MIN_TEXT_FOR_OCR_SKIP = 100
VISION_TRANSCRIBE_PROMPT = (
    "Przepisz DOKŁADNIE cały tekst z obrazu. Zachowaj pisownię, układ, symbole. "
    "Nie tłumacz, nie interpretuj - tylko przepisz. Jeśli coś nieczytelne - wpisz [NIECZYTELNE]."
)
VISION_DESCRIBE_PROMPT = (
    "Opisz ten obraz: co na nim widać? Wymień kluczowe elementy, teksty, wykresy lub diagramy, "
    "ogólny kontekst i ewentualny przekaz."
)

IMAGE_MODE_MAP = {
    "OCR": "ocr",
    "Vision: przepisz tekst": "vision_transcribe",
    "Vision: opisz obraz": "vision_describe",
    "OCR + Vision opis": "ocr_plus_vision_desc",
}

# === HELPERY ===
def safe_filename(name: str) -> str:
    """Sanityzacja nazwy pliku."""
    base = os.path.basename(name)
    base = re.sub(r'[^A-Za-z0-9.-]+', '', base)
    return base or "plik"

def create_run_dir(base_dir: str) -> str:
    """Katalog dla tego uruchomienia."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_text(path: str, text: str):
    """Zapis tekstu do pliku."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

def format_timestamp(seconds: float) -> str:
    """SRT format: 00:00:00,000"""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h = s // 3600
    s = s % 3600
    m = s // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def segments_to_srt(segments: list) -> str:
    """Whisper segments → SRT."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.get("start", 0.0))
        end = format_timestamp(seg.get("end", 0.0))
        text_seg = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text_seg}\n")
    return "\n".join(lines)

def calculate_timeout(file_size_bytes: int, base: int = 120) -> int:
    """Dynamiczny timeout bazując na rozmiarze pliku."""
    size_mb = file_size_bytes / 1024 / 1024
    return max(base, int(size_mb * 10))  # ~10s/MB

def list_ollama_models():
    """Lista modeli z Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
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
        return f"[BŁĄD VISION: {e}]"

def query_ollama_text(prompt: str, model: str = "llama3:latest", json_mode: bool = False, timeout: int = 120) -> str:
    """Tekstowe zapytanie do Ollama (bez obrazów)."""
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama text error: {e}")
        return f"[BŁĄD OLLAMA: {e}]"

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

def extract_audio_whisper(file):
    """Audio → tekst przez Whisper ASR. Zwraca (text, pages, meta)."""
    try:
        file.seek(0)
        timeout = calculate_timeout(file.size)
        data = file.read()
        files = {'audio_file': (file.name, data, getattr(file, "type", None) or "application/octet-stream")}
        r = requests.post(
            f"{WHISPER_URL}/asr?task=transcribe&language=pl&word_timestamps=true&output=json",
            files=files,
            timeout=timeout
        )
        r.raise_for_status()

        # Walidacja JSON
        try:
            result = r.json()
        except json.JSONDecodeError as je:
            logger.error(f"Whisper JSON decode error: {je}, response: {r.text[:500]}")
            return f"[BŁĄD: Whisper zwrócił nieprawidłowy format]", 0, {"type": "audio", "error": "invalid_json"}

        text_res = result.get("text", "") or ""
        segments = result.get("segments", [])

        # Metadata - tylko istotne dane, nie cały JSON
        meta = {
            "type": "audio",
            "segments_count": len(segments),
            "duration": result.get("duration"),
            "language": result.get("language"),
            "segments": segments  # Potrzebne dla SRT
        }

        # Format z timestampami
        if segments:
            lines = ["=== TRANSKRYPCJA Z TIMESTAMPAMI ===", ""]
            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                txt = seg.get("text", "").strip()
                lines.append(f"[{start:.1f}s - {end:.1f}s] {txt}")
            text_res = "\n".join(lines)

        return text_res, 1, meta

    except requests.exceptions.Timeout:
        logger.error(f"Whisper timeout after {timeout}s")
        return f"[BŁĄD: Timeout - plik zbyt długi]", 0, {"type": "audio", "error": "timeout"}
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return f"[BŁĄD AUDIO: {e}]", 0, {"type": "audio", "error": str(e)}

def diarize_audio(file) -> dict:
    """Pyannote speaker diarization."""
    pyannote_url = os.getenv("PYANNOTE_URL", "http://pyannote:8000")
    try:
        file.seek(0)
        timeout = calculate_timeout(file.size, base=300)
        files = {'file': (file.name, file.read(), getattr(file, "type", None) or "application/octet-stream")}
        r = requests.post(
            f"{pyannote_url}/diarize",
            files=files,
            timeout=timeout
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Pyannote error: {e}")
        return {}

def extract_audio_with_speakers(file):
    """Whisper + Pyannote = transkrypcja z identyfikacją głosów."""
    try:
        # 1. Whisper - transkrypcja z timestampami
        text_only, _, meta = extract_audio_whisper(file)
        segments = meta.get("segments", [])

        if not segments:
            return text_only, 1, meta

        # 2. Pyannote - diarization
        st.info("🎤 Identyfikuję głosy...")
        file.seek(0)
        diarization = diarize_audio(file)

        if not diarization or 'segments' not in diarization:
            st.warning("Nie udało się rozpoznać głosów - zwracam samą transkrypcję")
            return text_only, 1, meta

        # 3. Połącz - mapuj segmenty Whisper → głosy Pyannote
        output_lines = ["=== TRANSKRYPCJA Z IDENTYFIKACJĄ GŁOSÓW ===", ""]

        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            txt = seg.get("text", "").strip()

            # Znajdź kto mówi w tym przedziale
            speaker = "SPEAKER_?"
            for spk_seg in diarization['segments']:
                spk_start = spk_seg.get('start', 0)
                spk_end = spk_seg.get('end', 999999)
                if spk_start <= start <= spk_end:
                    speaker = spk_seg.get('speaker', 'SPEAKER_?')
                    break

            output_lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker}: {txt}")

        result = "\n".join(output_lines)
        meta['has_speakers'] = True

        return result, 1, meta

    except Exception as e:
        logger.error(f"Audio with speakers error: {e}")
        return f"[BŁĄD: {e}]", 0, {"type": "audio", "error": str(e)}

def extract_pdf(file, use_vision: bool, vision_model: str, ocr_pages_limit: int = 20):
    """PDF: tekst + opcjonalnie OCR/Vision."""
    texts = []
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                if i >= ocr_pages_limit:
                    texts.append(f"\n[... limit {ocr_pages_limit} stron ...]")
                    break
                t = page.extract_text() or ""
                texts.append(t)

        full_text = "\n".join(texts)

        if len(full_text.strip()) < MIN_TEXT_FOR_OCR_SKIP:
            file.seek(0)
            images = convert_from_bytes(
                file.read(),
                fmt="jpeg",
                dpi=150,
                first_page=1,
                last_page=min(ocr_pages_limit, 10)
            )

            if use_vision and vision_model:
                st.info(f"🖼️ Używam {vision_model} do analizy obrazów...")
                for idx, img in enumerate(images[:5], 1):
                    st.caption(f"Przetwarzam stronę {idx}/{len(images[:5])}")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buf.getvalue()).decode()

                    response = query_ollama_vision(VISION_TRANSCRIBE_PROMPT, img_b64, vision_model)
                    texts.append(f"\n--- Strona {idx} (Vision) ---\n{response}")
            else:
                st.info("📝 OCR Tesseract...")
                for idx, img in enumerate(images[:ocr_pages_limit], 1):
                    st.caption(f"OCR strona {idx}/{len(images)}")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    ocr_text = ocr_image_bytes(buf.getvalue())
                    texts.append(f"\n--- Strona {idx} (OCR) ---\n{ocr_text}")

        meta = {"type": "pdf", "pages": len(texts)}
        return "\n".join(texts), len(texts), meta
    except Exception as e:
        logger.error(f"PDF extract error: {e}")
        return f"[BŁĄD PDF: {e}]", 0, {"type": "pdf", "error": str(e)}

def extract_pptx(file, use_vision: bool, vision_model: str):
    """PPTX: tekst + obrazy (opcjonalnie Vision)."""
    try:
        file.seek(0)
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
                    if getattr(shape, "shape_type", None) == 13:  # PICTURE
                        try:
                            img_stream = shape.image.blob
                            img_b64 = base64.b64encode(img_stream).decode()
                            response = query_ollama_vision(VISION_DESCRIBE_PROMPT, img_b64, vision_model)
                            parts.append(f"[Obraz] {response}")
                        except Exception:
                            pass

            slides_text.append("\n".join(parts))

        return "\n\n".join(slides_text), len(prs.slides), {"type": "pptx", "slides": len(prs.slides)}
    except Exception as e:
        logger.error(f"PPTX error: {e}")
        return f"[BŁĄD PPTX: {e}]", 0, {"type": "pptx", "error": str(e)}

def extract_docx(file):
    """DOCX: tekst + tabele."""
    try:
        file.seek(0)
        doc = Document(file)
        paras = [p.text for p in doc.paragraphs if p.text]

        for tbl in doc.tables:
            for row in tbl.rows:
                paras.append(" | ".join(cell.text for cell in row.cells))

        return "\n".join(paras), len(paras), {"type": "docx"}
    except Exception as e:
        logger.error(f"DOCX error: {e}")
        return f"[BŁĄD DOCX: {e}]", 0, {"type": "docx", "error": str(e)}

def extract_image(file, use_vision: bool, vision_model: str, image_mode: str):
    """Obraz: OCR / Vision (przepisz) / Vision (opisz) / OCR+opis."""
    try:
        file.seek(0)
        img_bytes = file.read()
        results = []
        meta = {"type": "image", "mode": image_mode}

        if image_mode in ("ocr", "ocr_plus_vision_desc"):
            ocr_text = ocr_image_bytes(img_bytes)
            results.append(f"=== OCR ===\n{ocr_text}")

        if image_mode in ("vision_transcribe", "vision_describe", "ocr_plus_vision_desc"):
            if use_vision and vision_model:
                img_b64 = base64.b64encode(img_bytes).decode()
                prompt = VISION_TRANSCRIBE_PROMPT if image_mode == "vision_transcribe" else VISION_DESCRIBE_PROMPT
                vis = query_ollama_vision(prompt, img_b64, vision_model)
                tag = "Vision (transkrypcja)" if image_mode == "vision_transcribe" else "Vision (opis)"
                results.append(f"=== {tag} ===\n{vis}")
                meta["vision_model"] = vision_model
            else:
                results.append("[Vision niedostępne]")

        txt = "\n\n".join(results).strip()
        return txt, 1, meta
    except Exception as e:
        logger.error(f"Image error: {e}")
        return f"[BŁĄD IMG: {e}]", 0, {"type": "image", "error": str(e)}

def process_file(file, use_vision: bool, vision_model: str, ocr_limit: int, image_mode: str):
    """Router do odpowiedniego ekstraktora."""
    name = file.name.lower()

    if name.endswith('.pdf'):
        return extract_pdf(file, use_vision, vision_model, ocr_limit)
    elif name.endswith(('.pptx', '.ppt')):
        return extract_pptx(file, use_vision, vision_model)
    elif name.endswith('.docx'):
        return extract_docx(file)
    elif name.endswith(('.jpg', '.jpeg', '.png')):
        return extract_image(file, use_vision, vision_model, image_mode)
    elif name.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        # Sprawdź czy Pyannote dostępny
        pyannote_url = os.getenv("PYANNOTE_URL", "http://pyannote:8000")
        try:
            r = requests.get(f"{pyannote_url}/health", timeout=2)
            if r.ok and r.json().get("model_loaded"):
                return extract_audio_with_speakers(file)
        except Exception:
            pass
        # Fallback: tylko Whisper
        return extract_audio_whisper(file)
    elif name.endswith('.txt'):
        file.seek(0)
        content = file.read().decode('utf-8', errors='ignore')
        return content, 1, {"type": "txt"}
    else:
        return "[Nieobsługiwany format]", 0, {"type": "unknown"}

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

# --- PROMPTY DO PODSUMOWAŃ AUDIO ---
MAP_PROMPT_TEMPLATE = """
Jesteś asystentem ds. spotkań (PL). Otrzymasz fragment transkrypcji rozmowy z klientem (możliwe znaczniki SPEAKER_1, SPEAKER_2 i znaczniki czasu).
Zrób skrót tego fragmentu i wylistuj najważniejsze informacje.

WYMAGANY JSON:
{{
  "summary": "1-2 akapity skrótu (PL)",
  "key_points": ["punkt 1", "punkt 2", "..."],
  "decisions": ["decyzja 1", "decyzja 2"],
  "to_be_decided": ["kwestia do ustalenia 1", "kwestia 2"],
  "action_items": [{"owner":"", "task":"", "due":"", "notes":""}],
  "risks": [{"risk":"", "impact":"niski/średni/wysoki", "mitigation":""}],
  "open_questions": ["pytanie 1", "pytanie 2"]
}}

ZASADY:
- Nie wymyślaj informacji. Jeśli czegoś brak, zostaw puste pola lub wpisz [].
- Zostaw język polski.
- "to_be_decided" zawiera kwestie wymagające decyzji lub doprecyzowania.
- Jeśli są mówcy (SPEAKER_x) – staraj się zmapować właścicieli zadań (owner).
- Nie dodawaj komentarzy poza JSON.
Fragment:
{fragment}
"""

REDUCE_PROMPT_TEMPLATE = """
Jesteś asystentem ds. spotkań (PL). Otrzymasz listę częściowych podsumowań w JSON (z pól: summary, key_points, decisions, to_be_decided, action_items, risks, open_questions).
Scal je i zwróć jeden końcowy JSON w tym samym formacie. Usuń duplikaty, uczyść i pogrupuj logicznie.

WYMAGANY JSON:
{{
  "summary": "skondensowany skrót całości",
  "key_points": [...],
  "decisions": [...],
  "to_be_decided": [...],
  "action_items": [...],
  "risks": [...],
  "open_questions": [...]
}}

Wejście (lista JSON fragmentów):
{partials}

Nie dodawaj komentarzy poza JSON.
"""

def chunk_text(text: str, max_chars: int = 6000, overlap: int = 500) -> List[str]:
    """Dzielenie długiego tekstu na fragmenty do map-reduce (proste, po znakach)."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def try_parse_json(s: str) -> Dict[str, Any]:
    """Próba parsowania JSON z opcją oczyszczenia code fence."""
    if not s:
        return {}
    clean = s.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean)
    except Exception:
        return {}

def merge_summary_dicts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sklejanie listy słowników w formacie summary JSON (fallback)."""
    out = {
        "summary": "",
        "key_points": [],
        "decisions": [],
        "to_be_decided": [],
        "action_items": [],
        "risks": [],
        "open_questions": []
    }
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("summary"):
            out["summary"] += (("\n" if out["summary"] else "") + it["summary"])
        for k in ["key_points", "decisions", "to_be_decided", "open_questions"]:
            out[k].extend(it.get(k, []))
        for ai in it.get("action_items", []):
            if isinstance(ai, dict):
                out["action_items"].append(ai)
        for r in it.get("risks", []):
            if isinstance(r, dict):
                out["risks"].append(r)
    # deduplikacja prosta
    for k in ["key_points", "decisions", "to_be_decided", "open_questions"]:
        out[k] = list(dict.fromkeys(out[k]))
    return out

def build_meeting_summary_markdown(data: Dict[str, Any]) -> str:
    """Ładne formatowanie Markdown z danych JSON podsumowania."""
    if not data:
        return "_Brak danych do podsumowania_"
    md = []
    md.append("# Podsumowanie rozmowy")
    if data.get("summary"):
        md.append(data["summary"])

    # Kluczowe punkty
    md.append("\n## Kluczowe punkty")
    key_points = data.get("key_points", [])
    if not key_points:
        md.append("- brak")
    else:
        for x in key_points:
            md.append(f"- {x}")

    # Decyzje vs Do ustalenia
    md.append("\n## Decyzje vs Do ustalenia")
    decisions = data.get("decisions", [])
    tbd = data.get("to_be_decided", [])
    md.append("### Decyzje")
    if decisions:
        for d in decisions:
            md.append(f"- {d}")
    else:
        md.append("- brak")
    md.append("### Do ustalenia")
    if tbd:
        for q in tbd:
            md.append(f"- {q}")
    else:
        md.append("- brak")

    # Zadania
    md.append("\n## Zadania (Action Items)")
    action_items = data.get("action_items", [])
    if not action_items:
        md.append("- brak")
    else:
        for ai in action_items:
            owner = ai.get("owner", "") or "N/A"
            task = ai.get("task", "") or "N/A"
            due = ai.get("due", "") or "-"
            notes = ai.get("notes", "") or ""
            md.append(f"- [ ] {task} (owner: {owner}, termin: {due}) {('- ' + notes) if notes else ''}")

    # Ryzyka
    md.append("\n## Ryzyka")
    risks = data.get("risks", [])
    if not risks:
        md.append("- brak")
    else:
        for r in risks:
            risk = r.get("risk", "")
            impact = r.get("impact", "")
            mit = r.get("mitigation", "")
            md.append(f"- {risk} (wpływ: {impact}) → mitygacja: {mit}")

    # Pytania do klienta
    md.append("\n## Pytania do klienta (otwarte kwestie)")
    open_q = data.get("open_questions", [])
    if not open_q:
        md.append("- brak")
    else:
        for q in open_q:
            md.append(f"- {q}")

    return "\n".join(md)

def summarize_meeting_transcript(transcript: str, model: str = "llama3:latest", max_chars: int = 6000, diarized: bool = False) -> Dict[str, Any]:
    """
    Map-Reduce: dzieli transkrypcję na fragmenty, robi częściowe JSON-y, a następnie scali w jeden JSON.
    diarized: jeśli True, model zostanie poinformowany, że są SPEAKER_x (w prompt'cie już zaznaczone jako możliwe).
    """
    if not transcript or len(transcript.strip()) < 20:
        return {}

    # MAP
    parts = chunk_text(transcript, max_chars=max_chars, overlap=500)
    partials: List[Dict[str, Any]] = []
    for p in parts:
        prompt = MAP_PROMPT_TEMPLATE.format(fragment=p)
        resp = query_ollama_text(prompt, model=model, json_mode=True, timeout=180)
        data = try_parse_json(resp)
        if not data:
            # fallback: spróbuj bez format json
            resp2 = query_ollama_text(prompt, model=model, json_mode=False, timeout=180)
            data = try_parse_json(resp2)
        if data:
            partials.append(data)

    if not partials:
        # jeśli nie udało się uzyskać żadnego JSON - zwróć prymityw
        return {"summary": transcript[:1200] + ("..." if len(transcript) > 1200 else "")}

    # REDUCE
    partials_str = json.dumps(partials, ensure_ascii=False, indent=2)
    reduce_prompt = REDUCE_PROMPT_TEMPLATE.format(partials=partials_str)
    reduce_resp = query_ollama_text(reduce_prompt, model=model, json_mode=True, timeout=240)
    final_data = try_parse_json(reduce_resp)
    if not final_data:
        # fallback: proste merge lokalnie
        final_data = merge_summary_dicts(partials)

    return final_data

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

    st.subheader("Obrazy (IMG)")
    if use_vision and selected_vision:
        image_mode_label = st.selectbox(
            "Tryb dla obrazów",
            options=list(IMAGE_MODE_MAP.keys()),
            index=3
        )
    else:
        image_mode_label = st.selectbox(
            "Tryb dla obrazów",
            options=["OCR"],
            index=0,
            disabled=True
        )
    image_mode = IMAGE_MODE_MAP.get(image_mode_label, "ocr")

    st.subheader("Zapis lokalny")
    enable_local_save = st.checkbox("Zapisz wyniki lokalnie", value=False)
    base_output_dir = st.text_input("Katalog wyjściowy", value="outputs")
    per_file_save = st.checkbox("Zapisz też każdy plik osobno", value=True)

    st.subheader("AnythingLLM")
    has_anythingllm = bool(ANYTHINGLLM_URL and ANYTHINGLLM_API_KEY)
    st.caption(f"Status: {'✅ Skonfigurowane' if has_anythingllm else '❌ Brak config'}")

    st.subheader("🧠 Podsumowanie audio (AI)")
    summarize_audio_enabled = st.checkbox("Włącz podsumowanie rozmów audio", value=True)
    summarize_model_candidates = [
        m for m in list_ollama_models()
        if not any(m.startswith(p) for p in ("llava", "bakllava", "moondream", "llava-phi", "nomic-embed"))
    ]
    summarize_model = st.selectbox("Model do podsumowania", options=summarize_model_candidates or ["llama3:latest"])
    chunk_chars = st.slider("Rozmiar chunku (znaki)", min_value=2000, max_value=8000, value=6000, step=500)

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

        run_dir = None
        if enable_local_save:
            run_dir = create_run_dir(base_output_dir)
            st.info(f"💾 Wyniki będą zapisane w: {run_dir}")

        progress = st.progress(0)
        audio_items = []  # [(name, text, meta)]

        for idx, file in enumerate(uploaded_files):
            progress.progress((idx + 1) / len(uploaded_files), text=f"Przetwarzam: {file.name}")

            st.subheader(f"📄 {file.name}")

            try:
                extracted_text, pages, meta = process_file(file, use_vision, selected_vision, ocr_pages_limit, image_mode)

                all_texts.append(f"\n{'='*80}\n")
                all_texts.append(f"PLIK: {file.name}\n")
                all_texts.append(f"Typ: {getattr(file, 'type', 'unknown')}, Rozmiar: {getattr(file, 'size', 0)/1024:.1f} KB\n")
                all_texts.append(f"{'='*80}\n")
                all_texts.append(extracted_text)
                all_texts.append(f"\n[Stron/sekcji: {pages}]\n")

                stats['processed'] += 1
                stats['pages'] += pages

                with st.expander(f"Preview: {file.name}"):
                    st.text(extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""))

                if enable_local_save and per_file_save and run_dir:
                    fname_base = os.path.splitext(safe_filename(file.name))[0]
                    out_txt = os.path.join(run_dir, f"{fname_base}.txt")
                    save_text(out_txt, extracted_text)
                    st.caption(f"💾 Zapisano: {out_txt}")

                    if isinstance(meta, dict) and meta.get("type") == "audio":
                        segments = meta.get("segments", [])
                        if segments:
                            srt_path = os.path.join(run_dir, f"{fname_base}.srt")
                            save_text(srt_path, segments_to_srt(segments))
                            st.caption(f"💾 Zapisano SRT: {srt_path}")

                # Zbieraj audio do podsumowania
                if isinstance(meta, dict) and meta.get("type") == "audio":
                    audio_items.append((file.name, extracted_text, meta))

            except Exception as e:
                st.error(f"❌ Błąd: {e}")
                logger.exception(f"Error processing {file.name}")
                stats['errors'] += 1

        progress.empty()

        st.success(f"✅ Przetworzono: {stats['processed']}/{len(uploaded_files)}")
        st.metric("Strony/sekcje", stats['pages'])

        combined_text = "\n".join(all_texts)

        if enable_local_save and run_dir:
            combined_path = os.path.join(run_dir, f"combined_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
            save_text(combined_path, combined_text)
            st.success(f"📦 Połączony wynik: {combined_path}")

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

        # === PODSUMOWANIE AUDIO ===
        if summarize_audio_enabled and audio_items:
            st.subheader("🧠 Podsumowania rozmów audio")
            for (aname, atext, ameta) in audio_items:
                st.markdown(f"### 🎧 {aname}")
                diarized = bool(ameta.get("has_speakers"))
                with st.spinner(f"Tworzę podsumowanie dla {aname}..."):
                    summary_json = summarize_meeting_transcript(
                        transcript=atext,
                        model=summarize_model if summarize_model_candidates else "llama3:latest",
                        max_chars=chunk_chars,
                        diarized=diarized
                    )
                    summary_md = build_meeting_summary_markdown(summary_json)
                    st.markdown(summary_md)

                    # Zapis podsumowania (MD + JSON)
                    if enable_local_save and run_dir:
                        fname_base = os.path.splitext(safe_filename(aname))[0]
                        out_summary_md = os.path.join(run_dir, f"{fname_base}.summary.md")
                        out_summary_json = os.path.join(run_dir, f"{fname_base}.summary.json")
                        save_text(out_summary_md, summary_md)
                        save_text(out_summary_json, json.dumps(summary_json, ensure_ascii=False, indent=2))
                        st.caption(f"💾 Zapisano podsumowanie MD: {out_summary_md}")
                        st.caption(f"💾 Zapisano podsumowanie JSON: {out_summary_json}")

                    # Pobierz jako pliki
                    st.download_button(
                        "⬇️ Pobierz podsumowanie (MD)",
                        summary_md.encode("utf-8"),
                        file_name=f"{os.path.splitext(safe_filename(aname))[0]}_summary.md",
                        mime="text/markdown",
                        key=f"dl_md_{aname}"
                    )
                    st.download_button(
                        "⬇️ Pobierz podsumowanie (JSON)",
                        json.dumps(summary_json, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"{os.path.splitext(safe_filename(aname))[0]}_summary.json",
                        mime="application/json",
                        key=f"dl_json_{aname}"
                    )
