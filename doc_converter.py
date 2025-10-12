import streamlit as st
import io
import base64
import re
import requests
import logging
import os
import json
import socket
import ipaddress
import shutil
import subprocess
import platform
import importlib.util
from urllib.parse import urlparse
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

from typing import List, Dict, Any

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc-converter")

st.set_page_config(page_title="📄 Document Converter", layout="wide", page_icon="📄")

# === CONFIG (domyślnie localhost; offline mode ON) ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
ANYTHINGLLM_URL = os.getenv("ANYTHINGLLM_URL", "")  # domyślnie puste, wyłączone
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
WHISPER_URL = os.getenv("WHISPER_URL", "http://127.0.0.1:9000")
PYANNOTE_URL = os.getenv("PYANNOTE_URL", "http://127.0.0.1:8000")
OFFLINE_MODE = os.getenv("STRICT_OFFLINE", "1").lower() in ("1", "true", "yes")

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

# === OFFLINE GUARD ===
def is_private_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            sockaddr = info[4]
            ip = sockaddr[0] if isinstance(sockaddr, tuple) else sockaddr
            try:
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                    return True
            except ValueError:
                continue
    except Exception:
        return False
    return False

def assert_private_url(url: str):
    if not OFFLINE_MODE:
        return
    try:
        p = urlparse(url)
        host = p.hostname or ""
        if not host or is_private_host(host):
            return
    except Exception:
        pass
    raise RuntimeError(f"Zablokowano żądanie poza sieć lokalną: {url}")

def http_get(url, **kwargs):
    assert_private_url(url)
    return requests.get(url, **kwargs)

def http_post(url, **kwargs):
    assert_private_url(url)
    return requests.post(url, **kwargs)

# === HELPERY ===
def safe_filename(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r'[^A-Za-z0-9.-]+', '', base)
    return base or "plik"

def create_run_dir(base_dir: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

def format_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h = s // 3600
    s = s % 3600
    m = s // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def segments_to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.get("start", 0.0))
        end = format_timestamp(seg.get("end", 0.0))
        text_seg = (seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text_seg}\n")
    return "\n".join(lines)

# --- TIMEOUTY I HELPERY ---
def get_file_size(file) -> int:
    try:
        sz = getattr(file, "size", None)
        if sz is not None:
            return sz
        try:
            return len(file.getvalue())
        except Exception:
            pass
        pos = file.tell()
        file.seek(0, os.SEEK_END)
        sz = file.tell()
        file.seek(pos)
        return sz
    except Exception:
        return 0

def calculate_timeout(file_size_bytes: int, base: int = 240, per_mb: int = 25) -> int:
    size_mb = max(1.0, file_size_bytes / 1024 / 1024)
    return int(max(base, size_mb * per_mb))

# === DIAGNOSTYKA ===
def has_module(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

def module_version(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return "?"

def cmd_version(cmd: str, args: list = ["--version"]) -> str:
    try:
        res = subprocess.run([cmd] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3)
        out = (res.stdout or "").strip().splitlines()
        return out[0] if out else "?"
    except Exception:
        return "not found"

def list_tesseract_langs() -> List[str]:
    try:
        res = subprocess.run(["tesseract", "--list-langs"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3)
        out = res.stdout or ""
        langs = [l.strip() for l in out.splitlines() if l.strip() and not l.lower().startswith("list of available languages")]
        return langs
    except Exception:
        return []

def probe_service(name: str, base_url: str, paths: List[str]) -> Dict[str, Any]:
    info = {"name": name, "url": base_url, "ok": False, "detail": ""}
    try:
        for p in paths:
            url = base_url.rstrip("/") + p
            try:
                r = http_get(url, timeout=3)
                if r.ok:
                    info["ok"] = True
                    info["detail"] = f"OK {p} ({r.status_code})"
                    return info
                else:
                    info["detail"] = f"{p} -> HTTP {r.status_code}"
            except Exception as e:
                info["detail"] = f"{p} -> {e}"
        return info
    except Exception as e:
        info["detail"] = str(e)
        return info

def run_diagnostics() -> Dict[str, Any]:
    py_mods = ["pdfplumber", "pdf2image", "pytesseract", "pptx", "docx", "cv2", "PIL", "numpy", "requests", "pydub"]
    modules = {m: {"present": has_module(m), "version": module_version(m) if has_module(m) else ""} for m in py_mods}

    exes = {
        "tesseract": shutil.which("tesseract") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "pdftoppm": shutil.which("pdftoppm") is not None,
        "pdftocairo": shutil.which("pdftocairo") is not None,
        "nvidia-smi": shutil.which("nvidia-smi") is not None
    }
    versions = {
        "tesseract": cmd_version("tesseract"),
        "ffmpeg": cmd_version("ffmpeg"),
        "pdftoppm": cmd_version("pdftoppm", ["-v"]),
        "pdftocairo": cmd_version("pdftocairo", ["-v"]),
        "nvidia-smi": cmd_version("nvidia-smi")
    }

    langs = list_tesseract_langs()
    has_pol = any(l.lower() in ("pol", "polish") for l in langs)
    has_eng = any(l.lower() in ("eng", "english") for l in langs)

    services = {
        "ollama": probe_service("ollama", OLLAMA_URL, ["/api/tags", "/"]),
        "whisper": probe_service("whisper", WHISPER_URL, ["/health", "/", "/status"]),
        "pyannote": probe_service("pyannote", PYANNOTE_URL, ["/health", "/status", "/ping"])
    }

    missing_sys = []
    if not exes["tesseract"]: missing_sys.append("tesseract")
    if not exes["ffmpeg"]: missing_sys.append("ffmpeg")
    if not exes["pdftoppm"]: missing_sys.append("poppler-tools (pdftoppm)")
    missing_langs = []
    if not has_pol: missing_langs.append("tesseract-langpack-pol (pol)")
    if not has_eng: missing_langs.append("tesseract-langpack-eng (eng)")

    missing_py = [m for m, v in modules.items() if not v["present"]]

    rec_zypper = []
    if "tesseract" in missing_sys: rec_zypper.append("tesseract")
    if "ffmpeg" in missing_sys: rec_zypper.append("ffmpeg")
    if "poppler-tools (pdftoppm)" in missing_sys: rec_zypper.append("poppler-tools")
    rec_zypper += [pkg for pkg in ["tesseract-langpack-pol", "tesseract-langpack-eng"] if pkg in [x.split()[0] for x in missing_langs]]

    rec_pip = []
    if missing_py:
        rec_pip = missing_py

    gpu = {}
    if exes["nvidia-smi"]:
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], stdout=subprocess.PIPE, text=True, timeout=3)
            gpu["info"] = res.stdout.strip()
        except Exception as e:
            gpu["info"] = f"nvidia-smi error: {e}"
    else:
        gpu["info"] = "GPU nie wykryto (nvidia-smi brak)"

    return {
        "system": {
            "os": platform.platform(),
            "executables": exes,
            "versions": versions,
            "tesseract_langs": langs,
            "gpu": gpu
        },
        "python_modules": modules,
        "services": services,
        "missing": {
            "system_packages": missing_sys,
            "tesseract_languages": missing_langs,
            "python_modules": missing_py
        },
        "recommend_install": {
            "zypper": rec_zypper,
            "pip": rec_pip
        }
    }

# === OLLAMA/LLM/WIZJA ===
def list_ollama_models():
    try:
        r = http_get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.ok:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        logger.error(f"Ollama connection error: {e}")
    return []

def list_vision_models():
    all_models = list_ollama_models()
    prefixes = ("llava", "bakllava", "moondream", "llava-phi", "qwen2-vl")
    return [m for m in all_models if any(m.startswith(p) for p in prefixes)]

def query_ollama_vision(prompt: str, image_b64: str, model: str):
    try:
        payload = {"model": model, "prompt": prompt, "images": [image_b64], "stream": False}
        r = http_post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        logger.error(f"Vision model error: {e}")
        return f"[BŁĄD VISION: {e}]"

def query_ollama_text(prompt: str, model: str = "llama3:latest", json_mode: bool = False, timeout: int = 120) -> str:
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        r = http_post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama text error: {e}")
        return f"[BŁĄD OLLAMA: {e}]"

# === OCR ===
def ocr_image_bytes(img_bytes: bytes, lang: str = 'pol+eng') -> str:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        np_img = np.array(img)
        _, thr = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_thr = Image.fromarray(thr)
        return pytesseract.image_to_string(img_thr, lang=lang) or ""
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""

# === AUDIO: Whisper / Pyannote ===
def extract_audio_whisper(file):
    try:
        size_bytes = get_file_size(file)
        timeout_read = calculate_timeout(size_bytes, base=240, per_mb=25)
        file.seek(0)
        raw = file.read()
        mime = getattr(file, "type", None) or "application/octet-stream"
        fname = file.name

        try:
            if size_bytes > 25 * 1024 * 1024:
                from pydub import AudioSegment
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(fname)[1], delete=False) as tmp_in:
                    tmp_in.write(raw)
                    tmp_in_path = tmp_in.name
                audio = AudioSegment.from_file(tmp_in_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                buf = io.BytesIO()
                audio.export(buf, format="wav", parameters=["-acodec", "pcm_s16le"])
                raw = buf.getvalue()
                mime = "audio/wav"
                fname = os.path.splitext(fname)[0] + "_16k.wav"
                try:
                    os.remove(tmp_in_path)
                except Exception:
                    pass
        except Exception:
            pass

        files = {"audio_file": (fname, raw, mime)}
        r = http_post(
            f"{WHISPER_URL}/asr?task=transcribe&language=pl&word_timestamps=false&output=json",
            files=files,
            timeout=(30, timeout_read)
        )
        r.raise_for_status()

        try:
            result = r.json()
        except json.JSONDecodeError as je:
            logger.error(f"Whisper JSON decode error: {je}, response: {r.text[:500]}")
            return "[BŁĄD: Whisper zwrócił nieprawidłowy format]", 0, {"type": "audio", "error": "invalid_json"}

        text_res = result.get("text", "") or ""
        segments = result.get("segments", [])
        meta = {
            "type": "audio",
            "segments_count": len(segments),
            "duration": result.get("duration"),
            "language": result.get("language"),
            "segments": segments
        }

        if segments:
            lines = ["=== TRANSKRYPCJA Z TIMESTAMPAMI ===", ""]
            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                txt = (seg.get("text") or "").strip()
                lines.append(f"[{start:.1f}s - {end:.1f}s] {txt}")
            text_res = "\n".join(lines)

        return text_res, 1, meta

    except requests.exceptions.Timeout:
        logger.error(f"Whisper timeout after {timeout_read}s")
        return "[BŁĄD: Timeout - plik zbyt długi]", 0, {"type": "audio", "error": "timeout"}
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return f"[BŁĄD AUDIO: {e}]", 0, {"type": "audio", "error": str(e)}

def check_pyannote_health(url: str):
    url = (url or "").rstrip("/")
    for path in ("/health", "/status", "/ping"):
        try:
            r = http_get(url + path, timeout=3)
            if r.ok:
                try:
                    js = r.json()
                except Exception:
                    js = {"raw": r.text}
                if "model_loaded" in js:
                    return bool(js.get("model_loaded")), js
                return True, js
        except Exception:
            continue
    return False, {}

def normalize_diarization(resp: dict) -> list:
    if not resp:
        return []
    raw = resp.get("segments") or resp.get("turns") or []
    out = []
    for seg in raw:
        start = seg.get("start") or seg.get("start_time") or seg.get("begin") or 0.0
        end = seg.get("end") or seg.get("end_time") or seg.get("stop") or 0.0
        speaker = seg.get("speaker") or seg.get("label") or "SPEAKER_?"
        out.append({"start": float(start), "end": float(end), "speaker": speaker})
    return out

def pick_speaker_for_interval(diar_segments: list, start: float, end: float) -> str:
    best_spk, best_overlap = "SPEAKER_?", 0.0
    for s in diar_segments:
        ov = max(0.0, min(end, s["end"]) - max(start, s["start"]))
        if ov > best_overlap:
            best_overlap = ov
            best_spk = s["speaker"]
    return best_spk

def diarize_audio(file) -> dict:
    pyannote_url = PYANNOTE_URL.rstrip("/")
    try:
        size_bytes = get_file_size(file)
        timeout_read = calculate_timeout(size_bytes, base=300, per_mb=30)
        file.seek(0)
        files = {'file': (file.name, file.read(), getattr(file, "type", None) or "application/octet-stream")}
        r = http_post(f"{pyannote_url}/diarize", files=files, timeout=(30, timeout_read))
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        logger.error("Pyannote read timeout")
        return {"error": "timeout"}
    except Exception as e:
        logger.error(f"Pyannote error: {e}")
        return {"error": str(e)}

def extract_audio_with_speakers(file):
    try:
        text_only, _, meta = extract_audio_whisper(file)
        segments = meta.get("segments", [])
        if not segments:
            return text_only, 1, meta

        ok, _ = check_pyannote_health(PYANNOTE_URL)
        if not ok:
            st.warning("Nie udało się połączyć z Pyannote — zwracam samą transkrypcję.")
            return text_only, 1, meta

        st.info("🎤 Identyfikuję głosy...")
        file.seek(0)
        diarization = diarize_audio(file)
        diar_segments = normalize_diarization(diarization)

        if not diar_segments:
            st.warning("Pyannote nie zwrócił poprawnych segmentów — zwracam samą transkrypcję.")
            return text_only, 1, meta

        output_lines = ["=== TRANSKRYPCJA Z IDENTYFIKACJĄ GŁOSÓW ===", ""]
        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            txt = (seg.get("text") or "").strip()
            speaker = pick_speaker_for_interval(diar_segments, start, end)
            output_lines.append(f"[{start:.1f}s - {end:.1f}s] {speaker}: {txt}")

        result = "\n".join(output_lines)
        meta['has_speakers'] = True
        return result, 1, meta

    except Exception as e:
        logger.error(f"Audio with speakers error: {e}")
        return f"[BŁĄD: {e}]", 0, {"type": "audio", "error": str(e)}

# === EKSTRAKTORY ===
def extract_pdf(file, use_vision: bool, vision_model: str, ocr_pages_limit: int = 20):
    texts = []
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= ocr_pages_limit:
                    texts.append(f"\n[... limit {ocr_pages_limit} stron ...]")
                    break
                t = page.extract_text() or ""
                texts.append(t)

        full_text = "\n".join(texts)

        if len(full_text.strip()) < MIN_TEXT_FOR_OCR_SKIP:
            file.seek(0)
            raw = file.read()
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                total_pages = len(pdf.pages)

            images = convert_from_bytes(
                raw,
                fmt="jpeg",
                dpi=150,
                first_page=1,
                last_page=min(ocr_pages_limit, total_pages)
            )

            if use_vision and vision_model:
                st.info(f"🖼️ Używam {vision_model} do analizy obrazów...")
                for idx, img in enumerate(images[:ocr_pages_limit], 1):
                    st.caption(f"Przetwarzam stronę {idx}/{min(ocr_pages_limit, len(images))}")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                    response = query_ollama_vision(VISION_TRANSCRIBE_PROMPT, img_b64, vision_model)
                    texts.append(f"\n--- Strona {idx} (Vision) ---\n{response}")
            else:
                st.info("📝 OCR Tesseract...")
                for idx, img in enumerate(images[:ocr_pages_limit], 1):
                    st.caption(f"OCR strona {idx}/{min(ocr_pages_limit, len(images))}")
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
        ok, _ = check_pyannote_health(PYANNOTE_URL)
        if ok:
            return extract_audio_with_speakers(file)
        return extract_audio_whisper(file)
    elif name.endswith('.txt'):
        file.seek(0)
        content = file.read().decode('utf-8', errors='ignore')
        return content, 1, {"type": "txt"}
    else:
        return "[Nieobsługiwany format]", 0, {"type": "unknown"}

def send_to_anythingllm(text: str, filename: str):
    if OFFLINE_MODE:
        return False, "Tryb offline — wysyłka zablokowana"
    if not ANYTHINGLLM_URL or not ANYTHINGLLM_API_KEY:
        return False, "Brak konfiguracji AnythingLLM"
    try:
        headers = {"Authorization": f"Bearer {ANYTHINGLLM_API_KEY}"}
        payload = {"name": filename, "content": text, "type": "text/plain"}
        r = http_post(f"{ANYTHINGLLM_URL}/api/v1/document-upload", headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return True, "✅ Wysłano do AnythingLLM"
    except Exception as e:
        return False, f"Błąd AnythingLLM: {e}"

# --- PROMPTY DO PODSUMOWAŃ AUDIO ---
MAP_PROMPT_TEMPLATE = """
Jesteś asystentem ds. spotkań (PL). Otrzymasz fragment transkrypcji rozmowy z klientem (możliwe znaczniki SPEAKER_1, SPEAKER_2 i znaczniki czasu).
Zrób skrót tego fragmentu i wylistuj najważniejsze informacje.

WYMAGANY JSON:
{
  "summary": "1-2 akapity skrótu (PL)",
  "key_points": ["punkt 1", "punkt 2", "..."],
  "decisions": ["decyzja 1", "decyzja 2"],
  "to_be_decided": ["kwestia do ustalenia 1", "kwestia 2"],
  "action_items": [{ "owner":"", "task":"", "due":"", "notes":"" }],
  "risks": [{ "risk":"", "impact":"niski/średni/wysoki", "mitigation":"" }],
  "open_questions": ["pytanie 1", "pytanie 2"]
}

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
{
  "summary": "skondensowany skrót całości",
  "key_points": [...],
  "decisions": [...],
  "to_be_decided": [...],
  "action_items": [...],
  "risks": [...],
  "open_questions": [...]
}

Wejście (lista JSON fragmentów):
{partials}

Nie dodawaj komentarzy poza JSON.
"""

def chunk_text(text: str, max_chars: int = 6000, overlap: int = 500) -> List[str]:
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
    for k in ["key_points", "decisions", "to_be_decided", "open_questions"]:
        out[k] = list(dict.fromkeys(out[k]))
    return out

def build_meeting_summary_markdown(data: Dict[str, Any]) -> str:
    if not data:
        return "_Brak danych do podsumowania_"
    md = []
    md.append("# Podsumowanie rozmowy")
    if data.get("summary"):
        md.append(data["summary"])
    md.append("\n## Kluczowe punkty")
    key_points = data.get("key_points", [])
    if not key_points:
        md.append("- brak")
    else:
        for x in key_points:
            md.append(f"- {x}")
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
    md.append("\n## Pytania do klienta (otwarte kwestie)")
    open_q = data.get("open_questions", [])
    if not open_q:
        md.append("- brak")
    else:
        for q in open_q:
            md.append(f"- {q}")
    return "\n".join(md)

def summarize_meeting_transcript(transcript: str, model: str = "llama3:latest", max_chars: int = 6000, diarized: bool = False) -> Dict[str, Any]:
    if not transcript or len(transcript.strip()) < 20:
        return {}
    parts = chunk_text(transcript, max_chars=max_chars, overlap=500)
    partials: List[Dict[str, Any]] = []
    for p in parts:
        prompt = MAP_PROMPT_TEMPLATE.format(fragment=p)
        resp = query_ollama_text(prompt, model=model, json_mode=True, timeout=180)
        data = try_parse_json(resp)
        if not data:
            resp2 = query_ollama_text(prompt, model=model, json_mode=False, timeout=180)
            data = try_parse_json(resp2)
        if data:
            partials.append(data)
    if not partials:
        return {"summary": transcript[:1200] + ("..." if len(transcript) > 1200 else "")}
    partials_str = json.dumps(partials, ensure_ascii=False, indent=2)
    reduce_prompt = REDUCE_PROMPT_TEMPLATE.format(partials=partials_str)
    reduce_resp = query_ollama_text(reduce_prompt, model=model, json_mode=True, timeout=240)
    final_data = try_parse_json(reduce_resp)
    if not final_data:
        final_data = merge_summary_dicts(partials)
    return final_data

# === SESSION STATE INIT ===
def init_state():
    ss = st.session_state
    ss.setdefault("results", [])           # lista: {name, text, meta, pages}
    ss.setdefault("combined_text", "")
    ss.setdefault("audio_items", [])       # lista: (name, text, meta)
    ss.setdefault("audio_summaries", [])   # lista: {name, md, json}
    ss.setdefault("run_dir", None)         # stały katalog dla tego przebiegu
    ss.setdefault("stats", {'processed': 0, 'errors': 0, 'pages': 0})
    ss.setdefault("converted", False)      # czy mamy gotowe wyniki na ekranie
    ss.setdefault("files_sig", None)       # sygnatura zestawu plików (nazwa+rozmiar)
    ss.setdefault("diag", None)            # wyniki diagnostyki

def files_signature(files) -> int:
    try:
        items = [(f.name, getattr(f, 'size', None) or len(f.getvalue())) for f in files]
        return hash(tuple(items))
    except Exception:
        return 0

init_state()

# === UI ===
st.title("📄 Document Converter Pro")
st.caption("Konwersja PDF/DOCX/PPTX/IMG/AUDIO → TXT z OCR, Vision lub Whisper (offline)")

with st.sidebar:
    st.header("⚙️ Ustawienia")

    # Tryb offline
    offline_toggle = st.checkbox("Tryb offline (blokuj internet)", value=OFFLINE_MODE, help="Zezwalaj tylko na połączenia lokalne/prywatne")
    OFFLINE_MODE = offline_toggle

    def _status_url(name, url):
        try:
            host = urlparse(url).hostname or ""
            st.caption(f"{name}: {url} → {'✅ lokalny/prywatny' if is_private_host(host) else '❌ zewnętrzny'}")
        except Exception:
            st.caption(f"{name}: {url} → ⚠️ nie można zweryfikować")
    _status_url("Ollama", OLLAMA_URL)
    _status_url("Whisper", WHISPER_URL)
    _status_url("Pyannote", PYANNOTE_URL)

    vision_models = list_vision_models()
    use_vision = st.checkbox("Użyj modelu wizyjnego (Ollama Vision)", value=True if vision_models else False)

    if vision_models:
        selected_vision = st.selectbox("Model wizyjny", vision_models, index=0)
    else:
        selected_vision = None
        st.warning("⚠️ Brak modeli Vision w Ollama (np. llava:13b / qwen2-vl:7b)")

    st.subheader("OCR")
    ocr_pages_limit = st.slider("Limit stron OCR", 5, 50, 20)

    st.subheader("Obrazy (IMG)")
    if use_vision and selected_vision:
        image_mode_label = st.selectbox("Tryb dla obrazów", options=list(IMAGE_MODE_MAP.keys()), index=3)
    else:
        image_mode_label = st.selectbox("Tryb dla obrazów", options=["OCR"], index=0, disabled=True)
    image_mode = IMAGE_MODE_MAP.get(image_mode_label, "ocr")

    st.subheader("Zapis lokalny")
    enable_local_save = st.checkbox("Zapisz wyniki lokalnie (folder)", value=False)
    base_output_dir = st.text_input("Katalog wyjściowy", value="outputs")

    st.subheader("🧠 Podsumowanie audio (AI)")
    summarize_audio_enabled = st.checkbox("Włącz podsumowanie rozmów audio", value=True)
    summarize_model_candidates = [
        m for m in list_ollama_models()
        if not any(m.startswith(p) for p in ("llava", "bakllava", "moondream", "llava-phi", "nomic-embed", "qwen2-vl"))
    ]
    summarize_model = st.selectbox("Model do podsumowania", options=summarize_model_candidates or ["llama3:latest"])
    chunk_chars = st.slider("Rozmiar chunku (znaki)", min_value=2000, max_value=8000, value=6000, step=500)

    st.subheader("🔧 Diagnostyka środowiska")
    if st.button("Skanuj środowisko"):
        st.session_state["diag"] = run_diagnostics()
    if st.session_state.get("diag"):
        diag = st.session_state["diag"]
        st.caption("Wyniki diagnostyki (skrót):")
        miss = diag.get("missing", {})
        st.write(f"- Brakujące systemowe: {', '.join(miss.get('system_packages', []) ) or '—'}")
        st.write(f"- Brakujące języki Tesseract: {', '.join(miss.get('tesseract_languages', []) ) or '—'}")
        st.write(f"- Brakujące moduły Python: {', '.join(miss.get('python_modules', []) ) or '—'}")
        rec = diag.get("recommend_install", {})
        if rec.get("zypper") or rec.get("pip"):
            st.markdown("Zalecane instalacje:")
            if rec.get("zypper"):
                st.code("sudo zypper in -y " + " ".join(sorted(set(rec["zypper"])) ), language="bash")
            if rec.get("pip"):
                st.code("pip install " + " ".join(sorted(set(rec["pip"])) ), language="bash")
        with st.expander("Pełne szczegóły diagnostyki"):
            st.json(diag, expanded=False)

uploaded_files = st.file_uploader(
    "Wgraj dokumenty",
    type=['pdf', 'docx', 'pptx', 'ppt', 'jpg', 'jpeg', 'png', 'txt', 'mp3', 'wav', 'm4a', 'ogg', 'flac'],
    accept_multiple_files=True
)

# KONWERSJA → zapis do session_state (bez resetu przy zapisie)
if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} plików")
    if st.button("🚀 Konwertuj wszystkie", type="primary", key="btn_convert_all"):
        # Reset stanu dla nowego przebiegu
        st.session_state["results"] = []
        st.session_state["combined_text"] = ""
        st.session_state["audio_items"] = []
        st.session_state["audio_summaries"] = []
        st.session_state["stats"] = {'processed': 0, 'errors': 0, 'pages': 0}
        st.session_state["converted"] = False
        st.session_state["files_sig"] = files_signature(uploaded_files)
        st.session_state["run_dir"] = create_run_dir(base_output_dir) if enable_local_save else None
        if enable_local_save:
            st.info(f"💾 Wyniki będą zapisane w: {st.session_state['run_dir']}")

        progress = st.progress(0)
        all_texts = []

        for idx, file in enumerate(uploaded_files):
            try:
                progress.progress((idx + 1) / len(uploaded_files), text=f"Przetwarzam: {file.name}")
            except TypeError:
                progress.progress((idx + 1) / len(uploaded_files))
            st.subheader(f"📄 {file.name}")

            try:
                extracted_text, pages, meta = process_file(file, use_vision, selected_vision, ocr_pages_limit, image_mode)

                st.session_state["results"].append({
                    "name": file.name,
                    "text": extracted_text,
                    "meta": meta,
                    "pages": pages
                })

                all_texts.append(f"\n{'='*80}\n")
                all_texts.append(f"PLIK: {file.name}\n")
                all_texts.append(f"Typ: {getattr(file, 'type', 'unknown')}, Rozmiar: {getattr(file, 'size', 0)/1024:.1f} KB\n")
                all_texts.append(f"{'='*80}\n")
                all_texts.append(extracted_text)
                all_texts.append(f"\n[Stron/sekcji: {pages}]\n")

                st.session_state["stats"]["processed"] += 1
                st.session_state["stats"]["pages"] += pages

                with st.expander(f"Preview: {file.name}"):
                    st.text(extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""))

                # Audio do podsumowania
                if isinstance(meta, dict) and meta.get("type") == "audio":
                    st.session_state["audio_items"].append((file.name, extracted_text, meta))

            except Exception as e:
                st.error(f"❌ Błąd: {e}")
                logger.exception(f"Error processing {file.name}")
                st.session_state["stats"]["errors"] += 1

        progress.empty()
        st.session_state["combined_text"] = "\n".join(all_texts)
        st.session_state["converted"] = True

# === SEKCJA WYNIKÓW (bez resetu po zapisie) ===
if st.session_state.get("converted"):
    st.success(f"✅ Przetworzono: {st.session_state['stats']['processed']} plików")
    st.metric("Strony/sekcje", st.session_state["stats"]["pages"])

    # Pobierz/wyświetl łączny TXT
    st.download_button(
        "⬇️ Pobierz TXT",
        st.session_state["combined_text"].encode('utf-8'),
        file_name=f"converted_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        key="dl_combined_txt"
    )

    # Przyciski zapisu bez resetu
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💾 Zapisz połączony TXT na dysk", key="btn_save_combined_txt"):
            out_dir = st.session_state["run_dir"] or create_run_dir("outputs")
            st.session_state["run_dir"] = out_dir
            combined_path = os.path.join(out_dir, f"converted_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
            save_text(combined_path, st.session_state["combined_text"])
            st.success(f"Zapisano: {combined_path}")

    with c2:
        if st.button("💾 Zapisz wszystkie SRT (audio)", key="btn_save_all_srt"):
            out_dir = st.session_state["run_dir"] or create_run_dir("outputs")
            st.session_state["run_dir"] = out_dir
            saved = 0
            for it in st.session_state["results"]:
                meta = it.get("meta") or {}
                if meta.get("type") == "audio" and meta.get("segments"):
                    fname_base = os.path.splitext(safe_filename(it["name"]))[0]
                    srt_path = os.path.join(out_dir, f"{fname_base}.srt")
                    save_text(srt_path, segments_to_srt(meta["segments"]))
                    saved += 1
            st.success(f"Zapisano SRT dla {saved} plików audio w: {out_dir}")

    with c3:
        if st.button("🧠 Generuj podsumowania audio (MD+JSON)", key="btn_make_summaries"):
            st.session_state["audio_summaries"] = []
            if st.session_state["audio_items"]:
                for (aname, atext, ameta) in st.session_state["audio_items"]:
                    diarized = bool(ameta.get("has_speakers"))
                    with st.spinner(f"Tworzę podsumowanie dla {aname}..."):
                        summary_json = summarize_meeting_transcript(
                            transcript=atext,
                            model=summarize_model if summarize_model_candidates else "llama3:latest",
                            max_chars=chunk_chars,
                            diarized=diarized
                        )
                        summary_md = build_meeting_summary_markdown(summary_json)
                        st.session_state["audio_summaries"].append({"name": aname, "md": summary_md, "json": summary_json})
                st.success("Gotowe podsumowania — poniżej do pobrania/zapisania.")

    # Sekcja podsumowań (jeśli są)
    if st.session_state["audio_summaries"]:
        st.subheader("🧠 Podsumowania rozmów audio")
        for s in st.session_state["audio_summaries"]:
            aname = s["name"]
            summary_md = s["md"]
            summary_json = s["json"]

            st.markdown(f"### 🎧 {aname}")
            st.markdown(summary_md)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.download_button(
                    "⬇️ MD",
                    summary_md.encode("utf-8"),
                    file_name=f"{os.path.splitext(safe_filename(aname))[0]}_summary.md",
                    mime="text/markdown",
                    key=f"dl_md_{aname}"
                )
            with col_b:
                st.download_button(
                    "⬇️ JSON",
                    json.dumps(summary_json, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{os.path.splitext(safe_filename(aname))[0]}_summary.json",
                    mime="application/json",
                    key=f"dl_json_{aname}"
                )
            with col_c:
                if st.button("💾 Zapisz (MD+JSON) na dysk", key=f"btn_save_sum_{aname}"):
                    out_dir = st.session_state["run_dir"] or create_run_dir("outputs")
                    st.session_state["run_dir"] = out_dir
                    base = os.path.splitext(safe_filename(aname))[0]
                    save_text(os.path.join(out_dir, f"{base}.summary.md"), summary_md)
                    save_text(os.path.join(out_dir, f"{base}.summary.json"), json.dumps(summary_json, ensure_ascii=False, indent=2))
                    st.success(f"Zapisano do: {out_dir}")

    # Reset sesji (nie rusza plików na dysku)
    if st.button("♻️ Reset sesji (wyczyść wyniki)", type="secondary", key="btn_reset_session"):
        for k in ["results", "combined_text", "audio_items", "audio_summaries", "run_dir"]:
            st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else None
        st.session_state["stats"] = {'processed': 0, 'errors': 0, 'pages': 0}
        st.session_state["converted"] = False
        st.info("Wyczyszczono wyniki z pamięci sesji.")
