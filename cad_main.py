# main.py
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
from datetime import datetime
import base64
import re
import logging
from contextlib import contextmanager
import time
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
import io
from io import BytesIO
import plotly.express as px
from PyPDF2 import PdfReader
from rapidfuzz import fuzz, process
from openpyxl import load_workbook

=== KONFIGURACJA i LOGGING ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cad-estimator")

st.set_page_config(page_title="CAD Estimator Pro", layout="wide", page_icon="🚀")

=== ENV ===
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')
DB_HOST = os.getenv('DB_HOST', 'cad-postgres')
DB_NAME = os.getenv('DB_NAME', 'cad_estimator')
DB_USER = os.getenv('DB_USER', 'cad_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'cad_password_2024')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
EMBED_DIM = int(os.getenv('EMBED_DIM', '768'))

=== DZIAŁY ===
DEPARTMENTS = {
'131': 'Automotive',
'132': 'Industrial Machinery',
'133': 'Transportation',
'134': 'Heavy Equipment',
'135': 'Special Purpose Machinery'
}
DEPARTMENT_CONTEXT = {
'131': """Branża: AUTOMOTIVE (Faurecia, VW, Merit, Sitech, Joyson)
Specyfika: Komponenty samochodowe, wysokie wymagania jakościowe, spawanie precyzyjne, duże serie produkcyjne, normy automotive (IATF 16949).""",
'132': """Branża: INDUSTRIAL MACHINERY (PMP, ITM, Amazon)
Specyfika: Maszyny przemysłowe, automatyka, systemy pakowania, linie produkcyjne, robotyka przemysłowa, PLC.""",
'133': """Branża: TRANSPORTATION (Volvo, Scania)
Specyfika: Pojazdy ciężarowe, autobusy, systemy transportowe, wytrzymałość strukturalna, normy transportowe.""",
'134': """Branża: HEAVY EQUIPMENT (Volvo CE, Mine Master)
Specyfika: Maszyny budowlane, koparki, ładowarki, ekstremalne obciążenia, odporność na warunki terenowe.""",
'135': """Branża: SPECIAL PURPOSE MACHINERY (Bosch, Chassis Brakes, BWI, Besta)
Specyfika: Maszyny specjalne, niestandardowe rozwiązania, prototypy, unikalne wymagania klienta."""
}

=== SŁOWNIK NORMALIZACJI KOMPONENTÓW (PL/DE/EN -> EN) ===
COMPONENT_ALIASES = {
# Wsporniki
'wspornik': 'bracket', 'halterung': 'bracket', 'halter': 'bracket', 'träger': 'bracket',
'support': 'bracket', 'konsole': 'bracket',

# Ramy
'rama': 'frame', 'rahmen': 'frame', 'gestell': 'frame', 'chassis': 'frame',

# Przenośniki
'przenośnik': 'conveyor', 'förderband': 'conveyor', 'förderer': 'conveyor', 'transport': 'conveyor',

# Płyty
'płyta': 'plate', 'platte': 'plate', 'sheet': 'plate', 'panel': 'plate',

# Pokrywy
'pokrywa': 'cover', 'deckel': 'cover', 'abdeckung': 'cover',

# Obudowy
'obudowa': 'housing', 'gehäuse': 'housing', 'casing': 'housing',

# Napędy / siłowniki
'napęd': 'drive', 'antrieb': 'drive', 'actuator': 'drive',
'siłownik': 'cylinder', 'cylinder': 'cylinder', 'zylinder': 'cylinder',

# Prowadnice
'prowadnica': 'guide', 'führung': 'guide', 'rail': 'guide',

# Osłony
'osłona': 'shield', 'schutz': 'shield', 'guard': 'shield',

# Podstawy
'podstawa': 'base', 'basis': 'base', 'fundament': 'base', 'sockel': 'base',

# Wały
'wał': 'shaft', 'welle': 'shaft', 'axle': 'shaft',

# Łożyska
'łożysko': 'bearing', 'lager': 'bearing',

# Śruby / bolty
'śruba': 'screw', 'schraube': 'screw', 'bolt': 'bolt',
}

=== PROMPTY ===
MASTER_PROMPT = """Jesteś senior konstruktorem CAD z 20-letnim doświadczeniem w:

Projektowaniu ram spawalniczych i konstrukcji stalowych
Automatyce przemysłowej (PLC, robotyka, pozycjonery)
Systemach wizyjnych i kontroli jakości
Narzędziach CAD: CATIA V5, SolidWorks, AutoCAD
Odpowiadaj ZAWSZE w języku polskim.

METODYKA SZACOWANIA:

ANALIZA WYMAGAŃ (10-15% czasu)
KONCEPCJA I MODELOWANIE (40-50% czasu)
OBLICZENIA I WERYFIKACJA (20-30% czasu)
DOKUMENTACJA (15-20% czasu)
RYZYKA - każde MUSI mieć: "risk", "impact", "mitigation"
CZYNNIKI KOMPLIKUJĄCE (dodaj czas):

Spawanie precyzyjne: +20%
Części ruchome/kinematyka: +30%
Automatyzacja/PLC: +25%
Specjalne normy: +15%
Niestandardowe materiały: +10%
Duże wymiary (>10m): +25%
WYMAGANY FORMAT ODPOWIEDZI - ZWRÓĆ TYLKO CZYSTY JSON:
{
"components": [
{"name": "Nazwa", "layout_h": 12.5, "detail_h": 42.0, "doc_h": 28.0}
],
"sums": {"layout": 12.5, "detail": 42.0, "doc": 28.0, "total": 82.5},
"assumptions": ["Założenie 1"],
"risks": [
{"risk": "Opis ryzyka", "impact": "wysoki/średni/niski", "mitigation": "Jak zminimalizować"}
],
"adjustments": [
{
"parent": "Nazwa komponentu z głównej listy",
"adds": [
{"name": "nazwa sub-komponentu", "qty": 2, "layout_add": 0.5, "detail_add": 3.0, "doc_add": 1.0, "reason": "dlaczego"}
]
}
]
}

WAŻNE: Zwróć WYŁĄCZNIE JSON bez tekstu.
"""
def build_brief_prompt(description, components_excel, pdf_text, department):
    comps = components_excel or []
    parts_only = [c for c in comps if not c.get('is_summary')]
    total_layout = sum(c.get('hours_3d_layout', 0) for c in parts_only)
    total_detail = sum(c.get('hours_3d_detail', 0) for c in parts_only)
    total_2d = sum(c.get('hours_2d', 0) for c in parts_only)

    lines = []
    lines.append("Jesteś doświadczonym inżynierem/konstruktorem. Opracuj krótki opis zadania dla zespołu projektowego.")
    lines.append("Odpowiadaj ZAWSZE po polsku. Zwróć WYŁĄCZNIE JSON.")
    if department and department in DEPARTMENT_CONTEXT:
        lines.append("\n[Kontext działu]\n" + DEPARTMENT_CONTEXT[department])

    lines.append("\n[Opis użytkownika]\n" + (description or "(brak)"))

    lines.append("\n[Podsumowanie z Excela]")
    lines.append(f"- 3D Layout: {total_layout:.1f}h | 3D Detail: {total_detail:.1f}h | 2D: {total_2d:.1f}h | TOTAL: {total_layout+total_detail+total_2d:.1f}h")
    if parts_only:
        lines.append("\n[Komponenty (przykłady)]")
        for c in parts_only[:15]:
            row = f"- {c.get('name','?')}: L {c.get('hours_3d_layout',0):.1f}h, D {c.get('hours_3d_detail',0):.1f}h, 2D {c.get('hours_2d',0):.1f}h"
            if c.get('comment'):
                row += f" | Komentarz: {c['comment'][:120]}"
            subs = c.get('subcomponents') or []
            if subs:
                subs_str = ", ".join([f"{s.get('quantity',1)}x {s.get('name')}" for s in subs])[:180]
                row += f" | Zawiera: {subs_str}"
            lines.append(row)

    if pdf_text:
        lines.append("\n[Wyciąg z PDF (skrót)]")
        lines.append(pdf_text[:3000])

    lines.append("""
[Twoje zadanie]
Zbuduj brief zadania i checklistę:
- brief_md: 5-10 zdań podsumowania (markdown), jasno co robimy i co jest out-of-scope,
- scope: punktowo zakres (co robimy),
- assumptions: kluczowe założenia,
- missing_info: lista brakujących informacji/pytań do klienta (priorytetyzuj),
- risks: ryzyka (obowiązkowo: risk, impact: niski/średni/wysoki, mitigation),
- checklist: kontrolna lista do przejścia przed startem,
- open_questions: pytania do klienta/PM.

[Format JSON – zwróć tylko to]
{
  "brief_md": "markdown",
  "scope": ["..."],
  "assumptions": ["..."],
  "missing_info": ["..."],
  "risks": [{"risk":"...", "impact":"niski/średni/wysoki", "mitigation":"..."}],
  "checklist": ["..."],
  "open_questions": ["..."]
}
""")
    return "\n".join(lines)


def parse_brief_response(text: str) -> dict:
    if not text:
        return {"brief_md": "", "scope": [], "assumptions": [], "missing_info": [], "risks": [], "checklist": [], "open_questions": []}

    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]

    try:
        data = json.loads(clean)
        # Normalizacja pól
        data.setdefault("brief_md", "")
        data.setdefault("scope", [])
        data.setdefault("assumptions", [])
        data.setdefault("missing_info", [])
        risks = []
        for r in data.get("risks", []):
            if isinstance(r, dict):
                risks.append({
                    "risk": r.get("risk",""),
                    "impact": r.get("impact","nieznany"),
                    "mitigation": r.get("mitigation","")
                })
            elif isinstance(r, str):
                risks.append({"risk": r, "impact":"nieznany", "mitigation":""})
        data["risks"] = risks
        data.setdefault("checklist", [])
        data.setdefault("open_questions", [])
        return data
    except Exception:
        # fallback – potraktuj całość jako markdown brief
        return {
            "brief_md": text,
            "scope": [], "assumptions": [], "missing_info": [], "risks": [], "checklist": [], "open_questions": []
        }
def build_analysis_prompt(description, components_excel, learned_patterns, pdf_text, department):
    sections = []
    sections.append(MASTER_PROMPT)
    if department and department in DEPARTMENT_CONTEXT:
        sections.append(f"\n{DEPARTMENT_CONTEXT[department]}\n")
    sections.append(f"\nOPIS KLIENTA:\n{description or '(brak)'}\n")

    if components_excel:
        limited = components_excel[:30]
        total_layout = sum(c.get('hours_3d_layout', 0) for c in components_excel if not c.get('is_summary'))
        total_detail = sum(c.get('hours_3d_detail', 0) for c in components_excel if not c.get('is_summary'))
        total_2d = sum(c.get('hours_2d', 0) for c in components_excel if not c.get('is_summary'))

        sections.append(f"\nKOMPONENTY Z EXCEL ({len(limited)} z {len(components_excel)} pozycji):")
        sections.append(f"- 3D Layout: {total_layout:.1f}h")
        sections.append(f"- 3D Detail: {total_detail:.1f}h")
        sections.append(f"- 2D Documentation: {total_2d:.1f}h")
        sections.append("\nPrzykładowe komponenty:")
        for comp in limited[:15]:
            if not comp.get('is_summary'):
                sections.append(f"- {comp['name']}: Layout {comp.get('hours_3d_layout',0):.1f}h + Detail {comp.get('hours_3d_detail',0):.1f}h + 2D {comp.get('hours_2d',0):.1f}h")
                if comp.get('comment'):
                    sections.append(f"  Uwagi: {comp['comment'][:100]}")

        # SUBKOMPONENTY Z KOMENTARZY
        sections.append("\nSUBKOMPONENTY Z KOMENTARZY (nazwa + ilość):")
        for comp in limited[:15]:
            subs = comp.get('subcomponents', [])
            if subs:
                subs_txt = ", ".join([f"{s.get('quantity',1)}x {s.get('name')}" for s in subs])
                sections.append(f"- {comp['name']}: {subs_txt}")

    if pdf_text:
        sections.append("\nTREŚĆ Z DOKUMENTÓW PDF (skrót):\n")
        sections.append(pdf_text[:5000])

    if learned_patterns:
        sections.append(f"\n🧠 TWOJE DOŚWIADCZENIE W DZIALE {department}:")
        for p in learned_patterns[:10]:
            mark = "✅" if p['occurrences'] > 5 else "⚠️"
            layout_h = p.get('avg_hours_3d_layout', 0)
            detail_h = p.get('avg_hours_3d_detail', 0)
            doc_h = p.get('avg_hours_2d', 0)
            total_h = p.get('avg_hours_total', layout_h + detail_h + doc_h)
            sections.append(f"{mark} '{p['name']}': Total {total_h:.1f}h (Layout {layout_h:.1f}h, Detail {detail_h:.1f}h, 2D {doc_h:.1f}h) - {p['occurrences']}x")

    sections.append("\nWAŻNE: Zwróć WYŁĄCZNIE JSON bez tekstu.")
    return "\n".join(sections)

=== REQUESTS z retry ===
_session = None
def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        s.mount('http://', HTTPAdapter(max_retries=retries))
        s.mount('https://', HTTPAdapter(max_retries=retries))
        _session = s
    return _session

=== WEKTORY ===
def to_pgvector(vec):
    if not vec:
        return None
    return '[' + ','.join(f'{float(x):.6f}' for x in vec) + ']'

@st.cache_data(ttl=86400, show_spinner=False)
def get_embedding_ollama(text: str, model: str = EMBED_MODEL) -> list:
    try:
        s = get_session()
        r = s.post(f"{OLLAMA_URL}/api/embeddings", json={"model": model, "prompt": text}, timeout=30)
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Embeddings error: {e}")
        return []

def ensure_project_embedding(cur, project_id: int, description: str):
    if not description or len(description.strip()) < 10:
        return
    emb = get_embedding_ollama(description)
    if emb and len(emb) == EMBED_DIM:
        try:
            cur.execute("UPDATE projects SET description_embedding = %s::vector WHERE id=%s",
                        (to_pgvector(emb), project_id))
        except Exception as e:
            logger.warning(f"Embedding failed for project {project_id}: {e}")

def ensure_pattern_embedding(cur, pattern_key: str, dept: str, text_for_embed: str):
    if not text_for_embed or len(text_for_embed.strip()) < 3:
        return
    emb = get_embedding_ollama(text_for_embed)
    if emb and len(emb) == EMBED_DIM:
        try:
            cur.execute("""
            UPDATE component_patterns
            SET name_embedding = %s::vector
            WHERE pattern_key=%s AND department=%s
            """, (to_pgvector(emb), pattern_key, dept))
        except Exception as e:
            logger.warning(f"Embedding failed for pattern {pattern_key}: {e}")

=== AI API ===
@lru_cache(maxsize=1)
def list_local_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.ok:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except:
        pass
    return []

def model_available(prefix: str) -> bool:
    return any(m.startswith(prefix) for m in list_local_models())

@st.cache_data(ttl=3600, show_spinner=False)
def query_ollama_cached(_payload_str: str) -> str:
    payload = json.loads(_payload_str)
    try:
        s = get_session()
        r = s.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=90)
        r.raise_for_status()
        return r.json().get('response', 'Brak odpowiedzi.')
    except Exception as e:
        logger.error(f"Błąd AI: {e}")
        return f"Błąd Ollama: {e}"

def query_ollama(prompt: str, model: str = "llama3:latest", images_b64=None, format_json=False) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if images_b64:
        payload["images"] = images_b64
    if format_json:
        payload["format"] = "json"
    return query_ollama_cached(json.dumps(payload))

def encode_image_b64(file, max_px=1280, quality=85):
    try:
        im = Image.open(file).convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Błąd kompresji: {e}")
        return base64.b64encode(file.getvalue()).decode("utf-8")

=== NORMALIZACJA NAZW ===
def canonicalize_name(name: str) -> str:
    """Normalizuje nazwę komponentu do porównań i uczenia (z aliasami PL/DE/EN)."""
    if not name:
        return ""
    n = name.lower()
    # Usuń wymiary i liczby
    n = re.sub(r'\b\d+[.,]?\d*\s*(mm|cm|m|kg|t|ton|szt|pcs|inch|")\b', ' ', n)
    n = re.sub(r'\b\d+[.,]?\d*\b', ' ', n)
    # Tokenizacja i mapowanie aliasów
    tokens = re.split(r'[\s-_.,;/]+', n)
    norm_tokens = []
    stoplist = {'i', 'a', 'the', 'and', 'or', 'der', 'die', 'das', 'und', 'ein', 'eine', 'of', 'for'}
    for tok in tokens:
        if not tok or tok in stoplist:
            continue
        mapped = COMPONENT_ALIASES.get(tok)
        if not mapped:
            for alias, canonical in COMPONENT_ALIASES.items():
                if alias in tok:
                    mapped = canonical
                    break
        norm_tokens.append(mapped or tok)
    seen, out = set(), []
    for t in norm_tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ' '.join(out).strip()

=== PARSERY ===
def parse_subcomponents_from_comment(comment):
    """
    Ulepszony parser: obsługuje mieszane wpisy z i bez ilości oraz usuwa myślnik po liczbie (np. '2x - docisk').
    """
    if not comment or not isinstance(comment, str):
        return []

    def clean_name(s: str) -> str:
        s = re.sub(r'^\s*[-–—]\s*', '', s.strip())
        return s

    subcomponents = []
    qty_re = re.compile(r'(\d+)\s*(?:x|szt\.?|sztuk|pcs)?\s*[-–—]?\s*([^,;\n]+?)(?=[,;\n]|$)', re.IGNORECASE)

    consumed_spans = []
    for m in qty_re.finditer(comment):
        try:
            qty = int(m.group(1))
            name = clean_name(m.group(2))
            if len(name) >= 3 and not re.match(r'^\d+\s*(mm|cm|m|kg|ton|h)$', name, re.IGNORECASE):
                subcomponents.append({'quantity': qty, 'name': name})
                consumed_spans.append(m.span())
        except Exception:
            continue

    if consumed_spans:
        remainder_parts = []
        last = 0
        for a, b in consumed_spans:
            remainder_parts.append(comment[last:a])
            last = b
        remainder_parts.append(comment[last:])
        remainder = ';'.join(remainder_parts)
    else:
        remainder = comment

    for part in re.split(r'[;,]', remainder):
        name = clean_name(part)
        if not name or len(name) < 3:
            continue
        if re.match(r'^\d+(\.\d+)?\s*(mm|cm|m|kg|ton|h)?$', name, re.IGNORECASE):
            continue
        if qty_re.search(name):
            continue
        subcomponents.append({'quantity': 1, 'name': name})

    logger.info(f"Parsed {len(subcomponents)} subcomponents from: {comment[:80]}...")
    return subcomponents

def parse_ai_response(text: str, components_from_excel=None):
    """Parsing z priorytetem JSON, fallback do regex i Excel. Wyciąga też 'adjustments'."""
    warnings = []
    parsed_components = []
    total_layout = total_detail = total_2d = 0.0
    data = {}

    if not text:
        warnings.append("Brak odpowiedzi od AI")
        return {
            "total_hours": 0.0, "total_layout": 0.0, "total_detail": 0.0, "total_2d": 0.0,
            "components": components_from_excel or [], "raw_text": "", "warnings": warnings,
            "analysis": {}, "missing_info": [], "phases": {},
            "risks_detailed": [], "recommendations": [], "ai_adjustments": []
        }

    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]

    try:
        data = json.loads(clean)

        # Normalizacja risks
        risks = []
        for r in data.get("risks", []):
            if isinstance(r, str):
                risks.append({"risk": r, "impact": "nieznany", "mitigation": "Do określenia"})
            else:
                risks.append({
                    "risk": r.get("risk", "Nieznane ryzyko"),
                    "impact": r.get("impact", "nieznany"),
                    "mitigation": r.get("mitigation", "Brak")
                })
        data["risks"] = risks

        # Components
        for c in data.get("components", []):
            item = {
                "name": c.get("name", "bez nazwy"),
                "hours_3d_layout": float(c.get("layout_h", 0) or 0),
                "hours_3d_detail": float(c.get("detail_h", 0) or 0),
                "hours_2d": float(c.get("doc_h", 0) or 0),
            }
            item["hours"] = item["hours_3d_layout"] + item["hours_3d_detail"] + item["hours_2d"]
            parsed_components.append(item)

        sums = data.get("sums", {})
        total_layout = float(sums.get("layout", 0) or sum(x["hours_3d_layout"] for x in parsed_components))
        total_detail = float(sums.get("detail", 0) or sum(x["hours_3d_detail"] for x in parsed_components))
        total_2d = float(sums.get("doc", 0) or sum(x["hours_2d"] for x in parsed_components))

        # Adjustments (AI)
        ai_adj = []
        for ad in data.get("adjustments", []):
            parent = ad.get("parent")
            adds = ad.get("adds", [])
            norm_adds = []
            for a in adds:
                norm_adds.append({
                    "name": a.get("name", "sub"),
                    "qty": int(a.get("qty", 1) or 1),
                    "layout_add": float(a.get("layout_add", 0) or 0),
                    "detail_add": float(a.get("detail_add", 0) or 0),
                    "doc_add": float(a.get("doc_add", 0) or 0),
                    "reason": a.get("reason", "")
                })
            ai_adj.append({"parent": parent, "adds": norm_adds})
        data["ai_adjustments"] = ai_adj

    except json.JSONDecodeError:
        warnings.append("Fallback do regex")
        pattern = r"-\s*([^\n]+?)\s+Layout:\s*(\d+[.,]?\d*)\s*h?,?\s*Detail:\s*(\d+[.,]?\d*)\s*h?,?\s*2D:\s*(\d+[.,]?\d*)\s*h?"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                parsed_components.append({
                    "name": m.group(1).strip(),
                    "hours_3d_layout": float(m.group(2).replace(',', '.')),
                    "hours_3d_detail": float(m.group(3).replace(',', '.')),
                    "hours_2d": float(m.group(4).replace(',', '.')),
                    "hours": sum(float(m.group(i).replace(',', '.')) for i in [2,3,4])
                })
            except:
                pass
        data["ai_adjustments"] = []

    # fallback do Excela
    excel_parts = [c for c in (components_from_excel or []) if not c.get('is_summary', False)]
    if not parsed_components and excel_parts:
        warnings.append("Użyto danych z Excel - AI nie zwróciło komponentów")
        parsed_components = excel_parts
    elif parsed_components and excel_parts and len(parsed_components) < len(excel_parts) * 0.5:
        warnings.append(f"AI zwróciło tylko {len(parsed_components)} z {len(excel_parts)} komponentów - użyto danych z Excel")
        parsed_components = excel_parts

    if total_layout == 0 and parsed_components:
        total_layout = sum(c.get('hours_3d_layout', 0) for c in parsed_components)
    if total_detail == 0 and parsed_components:
        total_detail = sum(c.get('hours_3d_detail', 0) for c in parsed_components)
    if total_2d == 0 and parsed_components:
        total_2d = sum(c.get('hours_2d', 0) for c in parsed_components)

    return {
        "total_hours": max(0.0, total_layout + total_detail + total_2d),
        "total_layout": total_layout,
        "total_detail": total_detail,
        "total_2d": total_2d,
        "components": parsed_components,
        "raw_text": text,
        "warnings": warnings,
        "analysis": data.get("analysis", {}),
        "missing_info": data.get("missing_info", []),
        "phases": data.get("phases", {}),
        "risks_detailed": data.get("risks", []),
        "recommendations": data.get("recommendations", []),
        "ai_adjustments": data.get("ai_adjustments", [])
    }

def parse_cad_project_structured(file_stream):
    """Parser Excel z hierarchią, komentarzami i godzinami."""
    result = {'components': [], 'multipliers': {}, 'totals': {}, 'statistics': {}}
    df = pd.read_excel(file_stream, header=None)

    # Kolumny
    COL_POS, COL_DESC, COL_COMMENT = 0, 1, 2
    COL_STD_PARTS, COL_SPEC_PARTS = 3, 4
    COL_HOURS_LAYOUT, COL_HOURS_DETAIL, COL_HOURS_DOC = 7, 9, 11

    # Multipliers
    try:
        result['multipliers']['layout'] = float(df.iloc[9, COL_HOURS_LAYOUT]) if pd.notna(df.iloc[9, COL_HOURS_LAYOUT]) else 1.0
        result['multipliers']['detail'] = float(df.iloc[9, COL_HOURS_DETAIL]) if pd.notna(df.iloc[9, COL_HOURS_DETAIL]) else 1.0
        result['multipliers']['documentation'] = float(df.iloc[9, COL_HOURS_DOC]) if pd.notna(df.iloc[9, COL_HOURS_DOC]) else 1.0
    except:
        result['multipliers'] = {'layout': 1.0, 'detail': 1.0, 'documentation': 1.0}

    data_start_row = 11
    for row_idx in range(data_start_row, df.shape[0]):
        try:
            pos = str(df.iloc[row_idx, COL_POS]).strip() if pd.notna(df.iloc[row_idx, COL_POS]) else ""
            name = str(df.iloc[row_idx, COL_DESC]).strip() if pd.notna(df.iloc[row_idx, COL_DESC]) else ""
            if not pos or pos in ['nan', 'None', '']: continue
            if not name or name in ['nan', 'None']: name = f"[Pozycja {pos}]"
            comment = str(df.iloc[row_idx, COL_COMMENT]).strip() if pd.notna(df.iloc[row_idx, COL_COMMENT]) else ""

            hours_layout = float(df.iloc[row_idx, COL_HOURS_LAYOUT]) if pd.notna(df.iloc[row_idx, COL_HOURS_LAYOUT]) else 0.0
            hours_detail = float(df.iloc[row_idx, COL_HOURS_DETAIL]) if pd.notna(df.iloc[row_idx, COL_HOURS_DETAIL]) else 0.0
            hours_doc = float(df.iloc[row_idx, COL_HOURS_DOC]) if pd.notna(df.iloc[row_idx, COL_HOURS_DOC]) else 0.0

            is_summary = bool(re.match(r'^\d+,0$', pos) or pos.isdigit())

            subcomponents = parse_subcomponents_from_comment(comment)

            component = {
                'id': pos, 'name': name, 'comment': comment,
                'type': 'assembly' if is_summary else 'part',
                'level': pos.count(','),
                'parts': {'standard': int(float(df.iloc[row_idx, COL_STD_PARTS])) if pd.notna(df.iloc[row_idx, COL_STD_PARTS]) else 0,
                          'special': int(float(df.iloc[row_idx, COL_SPEC_PARTS])) if pd.notna(df.iloc[row_idx, COL_SPEC_PARTS]) else 0},
                'hours_3d_layout': hours_layout, 'hours_3d_detail': hours_detail, 'hours_2d': hours_doc,
                'hours': hours_layout + hours_detail + hours_doc,
                'is_summary': is_summary,
                'subcomponents': subcomponents
            }
            result['components'].append(component)
        except Exception as e:
            logger.warning(f"Błąd wiersz {row_idx + 1}: {e}")
            continue

    parts_only = [
        c for c in result['components']
        if not c.get('is_summary', False) and c.get('hours', 0) > 0 and c.get('name') not in ['[part]', '[assembly]', '', ' ']
    ]
    result['totals']['layout'] = sum(c['hours_3d_layout'] for c in parts_only)
    result['totals']['detail'] = sum(c['hours_3d_detail'] for c in parts_only)
    result['totals']['documentation'] = sum(c['hours_2d'] for c in parts_only)
    result['totals']['total'] = sum(c['hours'] for c in parts_only)
    result['statistics']['parts_count'] = len(parts_only)
    result['statistics']['assemblies_count'] = sum(1 for c in result['components'] if c.get('is_summary', False))
    return result

def parse_cad_project_structured_with_xlsx_comments(file_like):
    """
    Parser Excel z odczytem:
    - wartości (pandas),
    - komentarzy/note komórek (openpyxl),
    - łączeniem komentarzy z całego wiersza.
    Zwraca strukturę identyczną jak parse_cad_project_structured.
    """
    # Wczytaj bajty i utwórz dwa niezależne strumienie: dla pandas i openpyxl
    if hasattr(file_like, "read"):
        content = file_like.read()
    elif isinstance(file_like, bytes):
        content = file_like
    else:
        # zakładamy BytesIO
        file_like.seek(0)
        content = file_like.read()

    bio_pd = BytesIO(content)
    bio_xl = BytesIO(content)

    # 1) Dane tabelaryczne
    df = pd.read_excel(bio_pd, header=None)

    # 2) Komentarze komórek (openpyxl)
    comments_map = {}
    try:
        wb = load_workbook(bio_xl, data_only=True)
        ws = wb.active
        for r in ws.iter_rows():
            for cell in r:
                if cell.comment and cell.comment.text:
                    comments_map[(cell.row - 1, cell.column - 1)] = cell.comment.text.strip()
    except Exception as e:
        logger.info(f"Brak lub nie udało się odczytać komentarzy z pliku xlsx: {e}")

    result = {'components': [], 'multipliers': {}, 'totals': {}, 'statistics': {}}

    # Kolumny (spójne z istniejącym parserem)
    COL_POS, COL_DESC, COL_COMMENT = 0, 1, 2
    COL_STD_PARTS, COL_SPEC_PARTS = 3, 4
    COL_HOURS_LAYOUT, COL_HOURS_DETAIL, COL_HOURS_DOC = 7, 9, 11

    # Multipliers
    try:
        result['multipliers']['layout'] = float(df.iloc[9, COL_HOURS_LAYOUT]) if pd.notna(df.iloc[9, COL_HOURS_LAYOUT]) else 1.0
        result['multipliers']['detail'] = float(df.iloc[9, COL_HOURS_DETAIL]) if pd.notna(df.iloc[9, COL_HOURS_DETAIL]) else 1.0
        result['multipliers']['documentation'] = float(df.iloc[9, COL_HOURS_DOC]) if pd.notna(df.iloc[9, COL_HOURS_DOC]) else 1.0
    except Exception:
        result['multipliers'] = {'layout': 1.0, 'detail': 1.0, 'documentation': 1.0}

    data_start_row = 11
    for row_idx in range(data_start_row, df.shape[0]):
        try:
            pos = str(df.iloc[row_idx, COL_POS]).strip() if pd.notna(df.iloc[row_idx, COL_POS]) else ""
            name = str(df.iloc[row_idx, COL_DESC]).strip() if pd.notna(df.iloc[row_idx, COL_DESC]) else ""
            if not pos or pos in ['nan', 'None', '']:
                continue
            if not name or name in ['nan', 'None']:
                name = f"[Pozycja {pos}]"

            # Tekst z kolumny "Komentarz"
            txt_comment_col = str(df.iloc[row_idx, COL_COMMENT]).strip() if pd.notna(df.iloc[row_idx, COL_COMMENT]) else ""

            # Zbierz komentarze z całego wiersza (wszystkie kolumny)
            row_comments = []
            # Używamy maksymalnej liczby kolumn z arkusza openpyxl jeśli dostępny
            try:
                max_cols = max(c for (_, c) in comments_map.keys() if _ == row_idx) + 1 if comments_map else df.shape[1]
            except ValueError:
                max_cols = df.shape[1]

            for col in range(max_cols):
                txt = comments_map.get((row_idx, col))
                if txt:
                    row_comments.append(txt)

            joined_cell_comments = "; ".join([t for t in row_comments if t])
            # Finalny komentarz = kolumna + wszystkie komentarze/note z wiersza
            comment = "; ".join([t for t in [txt_comment_col, joined_cell_comments] if t])

            hours_layout = float(df.iloc[row_idx, COL_HOURS_LAYOUT]) if pd.notna(df.iloc[row_idx, COL_HOURS_LAYOUT]) else 0.0
            hours_detail = float(df.iloc[row_idx, COL_HOURS_DETAIL]) if pd.notna(df.iloc[row_idx, COL_HOURS_DETAIL]) else 0.0
            hours_doc = float(df.iloc[row_idx, COL_HOURS_DOC]) if pd.notna(df.iloc[row_idx, COL_HOURS_DOC]) else 0.0

            # Złożenia sumaryczne: "1,0" lub "1"
            is_summary = bool(re.match(r'^\d+,0$', pos) or pos.isdigit())

            # Sub‑komponenty z komentarza (Twoja funkcja)
            subcomponents = parse_subcomponents_from_comment(comment)

            component = {
                'id': pos, 'name': name, 'comment': comment,
                'type': 'assembly' if is_summary else 'part',
                'level': pos.count(','),
                'parts': {
                    'standard': int(float(df.iloc[row_idx, COL_STD_PARTS])) if pd.notna(df.iloc[row_idx, COL_STD_PARTS]) else 0,
                    'special': int(float(df.iloc[row_idx, COL_SPEC_PARTS])) if pd.notna(df.iloc[row_idx, COL_SPEC_PARTS]) else 0
                },
                'hours_3d_layout': hours_layout,
                'hours_3d_detail': hours_detail,
                'hours_2d': hours_doc,
                'hours': hours_layout + hours_detail + hours_doc,
                'is_summary': is_summary,
                'subcomponents': subcomponents
            }
            result['components'].append(component)
        except Exception as e:
            logger.warning(f"Błąd wiersz {row_idx + 1}: {e}")
            continue

    # Podsumowania jak w oryginale
    parts_only = [
        c for c in result['components']
        if not c.get('is_summary', False) and c.get('hours', 0) > 0 and c.get('name') not in ['[part]', '[assembly]', '', ' ']
    ]
    result['totals']['layout'] = sum(c['hours_3d_layout'] for c in parts_only)
    result['totals']['detail'] = sum(c['hours_3d_detail'] for c in parts_only)
    result['totals']['documentation'] = sum(c['hours_2d'] for c in parts_only)
    result['totals']['total'] = sum(c['hours'] for c in parts_only)
    result['statistics']['parts_count'] = len(parts_only)
    result['statistics']['assemblies_count'] = sum(1 for c in result['components'] if c.get('is_summary', False))
    return result

def process_excel(file):
    try:
        # Wczytaj bytes raz i użyj dla obu parserów
        content = file.read()
        bio1 = BytesIO(content)
        bio2 = BytesIO(content)

        try:
            result = parse_cad_project_structured_with_xlsx_comments(bio1)
            used_parser = "xlsx+comments"
        except Exception as e:
            logger.info(f"Parser z komentarzami nieudany, fallback: {e}")
            result = parse_cad_project_structured(bio2)
            used_parser = "basic"

        if result and result.get('components'):
            parts_only = [c for c in result['components'] if not c.get('is_summary', False)]
            st.success(f"✅ {len(parts_only)} komponentów: Layout {result['totals']['layout']:.1f}h + Detail {result['totals']['detail']:.1f}h + 2D {result['totals']['documentation']:.1f}h (parser: {used_parser})")
            if result.get('multipliers'):
                st.info(f"Współczynniki: Layout={result['multipliers']['layout']}, Detail={result['multipliers']['detail']}, Doc={result['multipliers']['documentation']}")
                st.session_state["excel_multipliers"] = result['multipliers']
            return result['components']
        else:
            st.warning("Brak komponentów w pliku")
            return []
    except Exception as e:
        st.error(f"Błąd parsowania: {e}")
        logger.exception("Błąd parsowania Excel")
        return []

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        max_pages = 200
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                text += f"\n[... {len(reader.pages)} stron, przetworzono {max_pages} ...]"
                break
            text += (page.extract_text() or "") + "\n"
        logger.info(f"PDF: {len(text)} znaków")
        return text
    except Exception as e:
        logger.error(f"Błąd PDF: {e}")
        return f"[Błąd PDF: {e}]"

def categorize_component(name: str) -> str:
    categories = {
        "analiza": ["przegląd", "analiza", "normy"],
        "modelowanie": ["modelowanie", "3d", "konstrukcja"],
        "obliczenia": ["obliczenia", "mes", "fem"],
        "rysunki": ["rysunek", "dokumentacja", "bom"],
        "spawanie": ["spawanie", "weld"],
        "automatyka": ["plc", "robot"]
    }
    n = name.lower()
    for cat, keys in categories.items():
        if any(k in n for k in keys):
            return cat
    return "inne"

def show_project_timeline(components):
    if not components:
        st.info("Brak komponentów do wyświetlenia")
        return
    parts = [c for c in components if not c.get('is_summary', False) and c.get('hours', 0) > 0]
    if not parts:
        st.info("Brak komponentów z godzinami")
        return
    timeline_data = []
    cumulative = 0
    for comp in parts:
        hours = comp.get('hours', 0)
        timeline_data.append({
            'Task': comp['name'][:30] + "..." if len(comp['name']) > 30 else comp['name'],
            'Start': cumulative,
            'Finish': cumulative + hours,
            'Hours': hours
        })
        cumulative += hours
    df = pd.DataFrame(timeline_data)
    fig = px.bar(df, x='Hours', y='Task', orientation='h', title="Harmonogram realizacji (sekwencyjnie)", labels={'Hours': 'Godziny', 'Task': 'Komponent'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def export_quotation_to_excel(project_data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        components = [c for c in project_data.get('components', []) if not c.get('is_summary', False)]
        df_components = pd.DataFrame(components)
        if not df_components.empty:
            df_components.to_excel(writer, sheet_name='Wycena', index=False)
        else:
            pd.DataFrame([{"info": "Brak"}]).to_excel(writer, sheet_name='Wycena', index=False)
        summary = pd.DataFrame({
            'Parametr': ['Nazwa', 'Klient', 'Dział', 'Suma', 'Data'],
            'Wartość': [
                project_data.get('name', ''), project_data.get('client', ''),
                project_data.get('department', ''), f"{project_data.get('total_hours', 0):.1f}",
                datetime.now().strftime('%Y-%m-%d')
            ]
        })
        summary.to_excel(writer, sheet_name='Podsumowanie', index=False)
    output.seek(0)
    return output.getvalue()

def save_project_version(conn, project_id, version, components, estimated_hours, layout_h, detail_h, doc_h, change_desc, changed_by):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO project_versions (
            project_id, version, components, estimated_hours,
            estimated_hours_3d_layout, estimated_hours_3d_detail, estimated_hours_2d,
            change_description, changed_by
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        """, (project_id, version, json.dumps(components, ensure_ascii=False),
              estimated_hours, layout_h, detail_h, doc_h, change_desc, changed_by))
        version_id = cur.fetchone()[0]
        conn.commit()
        return version_id

def get_project_versions(conn, project_id):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
        SELECT id, version, estimated_hours, estimated_hours_3d_layout,
               estimated_hours_3d_detail, estimated_hours_2d,
               change_description, changed_by, is_approved, created_at
        FROM project_versions WHERE project_id = %s ORDER BY created_at DESC
        """, (project_id,))
        return cur.fetchall()

def find_similar_projects(conn, description, department, limit=3):
    if not description:
        return []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
        SELECT id, name, client, estimated_hours, actual_hours, department
        FROM projects
        WHERE department = %s
        AND to_tsvector('simple', coalesce(name,'') || ' ' || coalesce(client,'') || ' ' || coalesce(description,'')) @@ websearch_to_tsquery('simple', %s)
        ORDER BY created_at DESC LIMIT %s
        """, (department, description, limit))
        return cur.fetchall()

def find_similar_projects_semantic(conn, description, department, limit=5):
    if not description:
        return []
    emb = get_embedding_ollama(description)
    if not emb:
        return []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
        SELECT id, name, client, department, estimated_hours, actual_hours,
               1 - (description_embedding <-> %s::vector) AS similarity
        FROM projects
        WHERE department = %s AND description_embedding IS NOT NULL
        ORDER BY description_embedding <-> %s::vector
        LIMIT %s
        """, (to_pgvector(emb), department, to_pgvector(emb), limit))
        return cur.fetchall()

def find_similar_components(conn, name, department, limit=5):
    key = canonicalize_name(name)
    emb = get_embedding_ollama(key)
    if not emb:
        return []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
        SELECT name, avg_hours_total, avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d,
               confidence, occurrences, 1 - (name_embedding <-> %s::vector) AS similarity
        FROM component_patterns
        WHERE department=%s AND name_embedding IS NOT NULL
        ORDER BY name_embedding <-> %s::vector
        LIMIT %s
        """, (to_pgvector(emb), department, to_pgvector(emb), limit))
        return cur.fetchall()

=== DB ===
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=5432)
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"Błąd połączenia: {e}")
        st.error("Błąd połączenia z bazą.")
        st.stop()
    except Exception as e:
        logger.error(f"Błąd: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

@st.cache_resource
def init_db():
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            # Extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Tabele bazowe (jak w Twoim kodzie)
            cur.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                client VARCHAR(255),
                department VARCHAR(10),
                cad_system VARCHAR(50),
                components JSONB,
                estimated_hours_3d_layout FLOAT DEFAULT 0,
                estimated_hours_3d_detail FLOAT DEFAULT 0,
                estimated_hours_2d FLOAT DEFAULT 0,
                estimated_hours FLOAT,
                actual_hours FLOAT,
                complexity_score INTEGER,
                accuracy FLOAT,
                description TEXT,
                ai_analysis TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                description_embedding vector(%s)
            )
            ''', (EMBED_DIM,))

            cur.execute('''
            CREATE TABLE IF NOT EXISTS component_patterns (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                department VARCHAR(10),
                avg_hours_3d_layout FLOAT DEFAULT 0,
                avg_hours_3d_detail FLOAT DEFAULT 0,
                avg_hours_2d FLOAT DEFAULT 0,
                avg_hours_total FLOAT DEFAULT 0,
                proportion_layout FLOAT DEFAULT 0.33,
                proportion_detail FLOAT DEFAULT 0.33,
                proportion_doc FLOAT DEFAULT 0.33,
                std_dev_hours FLOAT DEFAULT 0,
                occurrences INTEGER DEFAULT 0,
                min_hours FLOAT,
                max_hours FLOAT,
                typical_complexity FLOAT DEFAULT 1.0,
                cad_systems JSONB,
                last_updated TIMESTAMP DEFAULT NOW(),
                pattern_key TEXT,
                name_embedding vector(%s),
                m2_layout DOUBLE PRECISION DEFAULT 0,
                m2_detail DOUBLE PRECISION DEFAULT 0,
                m2_doc DOUBLE PRECISION DEFAULT 0,
                m2_total DOUBLE PRECISION DEFAULT 0,
                confidence DOUBLE PRECISION DEFAULT 0,
                source TEXT,
                last_actual_sample_at TIMESTAMP,
                UNIQUE(name, department)
            )
            ''', (EMBED_DIM,))

            cur.execute('''
            CREATE TABLE IF NOT EXISTS project_versions (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                version VARCHAR(20) NOT NULL,
                components JSONB,
                estimated_hours FLOAT,
                estimated_hours_3d_layout FLOAT DEFAULT 0,
                estimated_hours_3d_detail FLOAT DEFAULT 0,
                estimated_hours_2d FLOAT DEFAULT 0,
                change_description TEXT,
                changed_by VARCHAR(100),
                is_approved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
            ''')

            cur.execute('''
            CREATE TABLE IF NOT EXISTS category_baselines (
                id SERIAL PRIMARY KEY,
                department VARCHAR(10),
                category TEXT,
                mean_layout DOUBLE PRECISION DEFAULT 0,
                mean_detail DOUBLE PRECISION DEFAULT 0,
                mean_doc DOUBLE PRECISION DEFAULT 0,
                m2_layout DOUBLE PRECISION DEFAULT 0,
                m2_detail DOUBLE PRECISION DEFAULT 0,
                m2_doc DOUBLE PRECISION DEFAULT 0,
                occurrences INTEGER DEFAULT 0,
                confidence DOUBLE PRECISION DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW(),
                UNIQUE(department, category)
            )
            ''')

            # NOWA TABELA: relacje parent -> sub (bundles)
            cur.execute('''
            CREATE TABLE IF NOT EXISTS component_bundles (
                id SERIAL PRIMARY KEY,
                department VARCHAR(10),
                parent_key TEXT,
                parent_name TEXT,
                sub_key TEXT,
                sub_name TEXT,
                occurrences INTEGER DEFAULT 0,
                total_qty DOUBLE PRECISION DEFAULT 0,
                confidence DOUBLE PRECISION DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW(),
                UNIQUE (department, parent_key, sub_key)
            )
            ''')

            # Indeksy istniejące
            cur.execute('CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_projects_department ON projects(department)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_patterns_department ON component_patterns(department)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_component_patterns_name ON component_patterns(name, department)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_versions_project ON project_versions(project_id, created_at DESC)')
            cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_desc_embed_hnsw ON projects USING hnsw (description_embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_component_patterns_name_embed_hnsw ON component_patterns USING hnsw (name_embedding vector_l2_ops);")

            # NOWE INDEXY dla bundles
            cur.execute('CREATE INDEX IF NOT EXISTS idx_bundles_dept_parent ON component_bundles(department, parent_key)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_bundles_dept_parent_occ ON component_bundles(department, parent_key, occurrences DESC)')

            # NOWE KOLUMNY w projects (bezpiecznie, idempotentnie)
            cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS is_historical BOOLEAN DEFAULT FALSE;")
            cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS estimation_mode VARCHAR(30) DEFAULT 'ai';")
            cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS totals_source VARCHAR(30) DEFAULT 'ai';")
            cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS locked_totals BOOLEAN DEFAULT FALSE;")

            # Indeksy dla nowych kolumn
            cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_hist ON projects(is_historical);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_estimation_mode ON projects(estimation_mode);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_totals_source ON projects(totals_source);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_locked_totals ON projects(locked_totals);")

            conn.commit()
            logger.info("Baza zainicjalizowana + migracje (bundles, flagi projektów).")
            return True
    except Exception as e:
        logger.error(f"Błąd inicjalizacji: {e}")
        st.error(f"Błąd inicjalizacji: {e}")
        return False

=== STATYSTYKA (WELFORD) ===
def _welford_step(mean, m2, n, x):
    """Algorytm Welforda - aktualizacja średniej i wariancji + odrzucanie outlierów po min prób."""
    if n and n >= 5:
        std = (m2 / max(n - 1, 1)) ** 0.5
        if mean and abs(x - mean) > 2.5 * std:
            return mean, m2, n # outlier - odrzuć
    n_new = (n or 0) + 1
    delta = x - (mean or 0)
    mean_new = (mean or 0) + delta / n_new
    delta2 = x - mean_new
    m2_new = (m2 or 0) + delta * delta2
    return mean_new, m2_new, n_new

def best_pattern_key(cur, dept: str, key: str, threshold: int = 88) -> str:
    cur.execute("SELECT pattern_key FROM component_patterns WHERE pattern_key=%s AND department=%s", (key, dept))
    if cur.fetchone():
        return key
    cur.execute("SELECT DISTINCT pattern_key FROM component_patterns WHERE department=%s AND pattern_key IS NOT NULL", (dept,))
    keys = [r[0] for r in cur.fetchall()]
    if not keys:
        return key
    match, score, _ = process.extractOne(key, keys, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else key

def update_pattern_smart(cur, name, dept, layout_h, detail_h, doc_h, source='actual'):
    """Welford + outlier + confidence + fuzzy + embedding (n++ tylko raz)."""
    key = best_pattern_key(cur, dept, canonicalize_name(name))
    total = float(layout_h) + float(detail_h) + float(doc_h)

    cur.execute("""
        SELECT avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d, avg_hours_total,
               m2_layout, m2_detail, m2_doc, m2_total, occurrences
        FROM component_patterns
        WHERE pattern_key=%s AND department=%s
    """, (key, dept))
    row = cur.fetchone()

    if row:
        ml, md, mc, mt, m2l, m2d, m2c, m2t, n0 = row
        ml, m2l, _ = _welford_step(ml, m2l, n0, float(layout_h))
        md, m2d, _ = _welford_step(md, m2d, n0, float(detail_h))
        mc, m2c, _ = _welford_step(mc, m2c, n0, float(doc_h))
        mt, m2t, _ = _welford_step(mt, m2t, n0, float(total))

        n1 = (n0 or 0) + 1
        std_total = (m2t / max(n1 - 1, 1)) ** 0.5 if n1 > 1 else 0.0
        confidence = min(1.0, n1 / 10.0) * (1.0 / (1.0 + (std_total / (mt or 1e-6))))

        cur.execute("""
            UPDATE component_patterns
            SET avg_hours_3d_layout=%s, avg_hours_3d_detail=%s, avg_hours_2d=%s, avg_hours_total=%s,
                m2_layout=%s, m2_detail=%s, m2_doc=%s, m2_total=%s,
                occurrences=%s, confidence=%s, source=%s,
                last_updated=NOW(),
                last_actual_sample_at=CASE WHEN %s='actual' THEN NOW() ELSE last_actual_sample_at END,
                pattern_key=%s
            WHERE pattern_key=%s AND department=%s
        """, (ml, md, mc, mt, m2l, m2d, m2c, m2t, n1, confidence, source, source, key, key, dept))
    else:
        cur.execute("""
            INSERT INTO component_patterns (
                name, pattern_key, department,
                avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d, avg_hours_total,
                m2_layout, m2_detail, m2_doc, m2_total,
                occurrences, confidence, source, last_actual_sample_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 0, 0, 0, 0, 1, 0.1, %s,
                      CASE WHEN %s='actual' THEN NOW() ELSE NULL END)
        """, (name, key, dept, layout_h, detail_h, doc_h, total, source, source))

    ensure_pattern_embedding(cur, key, dept, name)
    return True

def update_category_baseline(cur, dept, category, layout_h, detail_h, doc_h):
    """Aktualizuje baseline dla kategorii (n++ 1x)."""
    cur.execute("""
    SELECT mean_layout, mean_detail, mean_doc, m2_layout, m2_detail, m2_doc, occurrences
    FROM category_baselines WHERE department=%s AND category=%s
    """, (dept, category))
    row = cur.fetchone()

    if row:
        ml, md, mc, m2l, m2d, m2c, n0 = row
        ml, m2l, _ = _welford_step(ml, m2l, n0, float(layout_h))
        md, m2d, _ = _welford_step(md, m2d, n0, float(detail_h))
        mc, m2c, _ = _welford_step(mc, m2c, n0, float(doc_h))
        n1 = (n0 or 0) + 1
        conf = min(1.0, n1 / 10.0)

        cur.execute("""
            UPDATE category_baselines
            SET mean_layout=%s, mean_detail=%s, mean_doc=%s,
                m2_layout=%s, m2_detail=%s, m2_doc=%s,
                occurrences=%s, confidence=%s, last_updated=NOW()
            WHERE department=%s AND category=%s
        """, (ml, md, mc, m2l, m2d, m2c, n1, conf, dept, category))
    else:
        cur.execute("""
            INSERT INTO category_baselines (department, category, mean_layout, mean_detail, mean_doc, occurrences, confidence)
            VALUES (%s, %s, %s, %s, %s, 1, 0.1)
        """, (dept, category, layout_h, detail_h, doc_h))

def update_bundle(cur, dept: str, parent_name: str, sub_name: str, qty: int):
    """
    Aktualizuje tabelę component_bundles: relacja parent -> sub,
    liczy częstość współwystępowania i sumę ilości (qty).
    """
    try:
        pkey = canonicalize_name(parent_name or "")
        skey = canonicalize_name(sub_name or "")
        if not pkey or not skey:
            return
        q = max(1, int(qty or 1))
        cur.execute("""
            INSERT INTO component_bundles (department, parent_key, parent_name, sub_key, sub_name, occurrences, total_qty, confidence)
            VALUES (%s, %s, %s, %s, %s, 1, %s, 0.1)
            ON CONFLICT (department, parent_key, sub_key)
            DO UPDATE SET
                occurrences = component_bundles.occurrences + 1,
                total_qty = component_bundles.total_qty + EXCLUDED.total_qty,
                confidence = LEAST(1.0, (component_bundles.occurrences + 1) / 10.0),
                last_updated = NOW()
        """, (dept, pkey, parent_name, skey, sub_name, float(q)))
    except Exception as e:
        logger.warning(f"update_bundle err: {e}")

def learn_from_historical_components(cur, dept: str, components: list, distribute: str = 'qty'):
    """
    Uczy:
    - component_patterns: komponenty główne + sub-komponenty (proporcjonalnie do ilości lub po równo),
    - component_bundles: relacje parent->sub z typową ilością.

    distribute: 'qty' (waga = qty / sum_qty) lub 'equal' (po równo).
    """
    for comp in components or []:
        try:
            name = comp.get('name', '')
            if not name:
                continue

            is_summary = bool(comp.get('is_summary'))
            subs = comp.get('subcomponents', []) or []

            # ZAWSZE: aktualizuj bundles (wiedza co z czym)
            for sub in subs:
                update_bundle(cur, dept, name, sub.get('name', ''), sub.get('quantity', 1))

            # Komponent sumaryczny nie wnosi godzin - nie ucz wzorca godzin
            if is_summary:
                continue

            layout = float(comp.get('hours_3d_layout', 0) or 0)
            detail = float(comp.get('hours_3d_detail', 0) or 0)
            doc = float(comp.get('hours_2d', 0) or 0)
            total = layout + detail + doc

            # 1) ucz wzorzec komponentu (jeśli są godziny)
            if total > 0:
                update_pattern_smart(cur, name, dept, layout, detail, doc, source='historical_excel')

            # 2) rozdziel godziny na sub-komponenty
            if subs and total > 0:
                if distribute == 'qty':
                    total_qty = sum(int(s.get('quantity', 1) or 1) for s in subs) or len(subs)
                    for sub in subs:
                        q = max(1, int(sub.get('quantity', 1) or 1))
                        w = (q / total_qty) if total_qty else (1.0 / len(subs))
                        sl, sd, sdoc = layout * w, detail * w, doc * w
                        update_pattern_smart(cur, sub.get('name', ''), dept, sl, sd, sdoc, source='historical_excel_sub')
                else:
                    # equal
                    n = len(subs)
                    if n > 0:
                        w = 1.0 / n
                        for sub in subs:
                            sl, sd, sdoc = layout * w, detail * w, doc * w
                            update_pattern_smart(cur, sub.get('name', ''), dept, sl, sd, sdoc, source='historical_excel_sub')

        except Exception as e:
            logger.warning(f"learn_from_historical_components err for '{comp.get('name','?')}': {e}")

=== HEURYSTYKI I PROPOZYCJE Z KOMENTARZY ===
HEURISTIC_LIBRARY = [
    # keywords, per-piece hours L/D/2D
    (['docisk', 'clamp'], 0.5, 1.5, 0.5),
    (['śruba trapezowa', 'trapez'], 0.2, 0.8, 0.3),
    (['konsola', 'bracket'], 0.3, 1.0, 0.4),
    (['płyta', 'plate'], 0.2, 0.7, 0.4),
]

def heuristic_estimate_for_name(name: str):
    n = name.lower()
    for keys, l, d, doc in HEURISTIC_LIBRARY:
        if any(k in n for k in keys):
            return l, d, doc, f"Heurystyka: {', '.join(keys)}"
    return 0.0, 0.0, 0.0, ""

def propose_adjustments_for_components(conn, components, department, conservativeness=1.0, sim_threshold=0.6):
    """
    Dla każdego komponentu z sub-komponentami proponuje dodatki godzin:
    - najpierw wzorce (embedding),
    - jeśli brak, heurystyki,
    - zwraca listę {"parent": name, "adds": [{name, qty, layout_add, detail_add, doc_add, reason, source, confidence}]}
    """
    proposals = []
    for comp in components:
        subs = comp.get('subcomponents', [])
        if not subs:
            continue

        # Agregacja
        agg = {}
        for s in subs:
            qty = int(s.get('quantity', 1) or 1)
            nm = s.get('name', '').strip()
            if not nm:
                continue
            key = canonicalize_name(nm)
            if key not in agg:
                agg[key] = {'display_name': nm, 'qty': 0}
            agg[key]['qty'] += qty

        adds = []
        for key_name, info in agg.items():
            display_name = info['display_name']
            qty = info['qty']

            # 1) wzorzec
            similar = find_similar_components(conn, display_name, department, limit=1)
            used = False
            if similar:
                s0 = similar[0]
                sim = float(s0.get('similarity') or 0.0)
                if sim >= sim_threshold:
                    l = float(s0.get('avg_hours_3d_layout') or 0.0)
                    d = float(s0.get('avg_hours_3d_detail') or 0.0)
                    doc = float(s0.get('avg_hours_2d') or 0.0)
                    tot = float(s0.get('avg_hours_total') or (l + d + doc))
                    if tot > 0 and (l + d + doc) == 0:
                        l = tot * 0.3
                        d = tot * 0.5
                        doc = tot * 0.2
                    adds.append({
                        "name": display_name, "qty": qty,
                        "layout_add": l * qty * conservativeness,
                        "detail_add": d * qty * conservativeness,
                        "doc_add": doc * qty * conservativeness,
                        "reason": f"Wzorzec: {s0.get('name')} (sim={sim*100:.0f}%)",
                        "source": "pattern",
                        "confidence": float(s0.get('confidence') or 0.5)
                    })
                    used = True

            # 2) heurystyka
            if not used:
                l, d, doc, why = heuristic_estimate_for_name(display_name)
                if l + d + doc > 0:
                    adds.append({
                        "name": display_name, "qty": qty,
                        "layout_add": l * qty * conservativeness,
                        "detail_add": d * qty * conservativeness,
                        "doc_add": doc * qty * conservativeness,
                        "reason": why or "Heurystyka ogólna",
                        "source": "heuristic",
                        "confidence": 0.4
                    })
        if adds:
            proposals.append({"parent": comp.get('name', 'bez nazwy'), "adds": adds})
    return proposals

def propose_bundles_for_component(conn, parent_name: str, department: str,
                                  conservativeness: float = 1.0,
                                  top_k: int = 5, min_occ: int = 2) -> list:
    """
    Zwraca listę dodatków bazujących na historii (component_bundles):
    - wybiera typowe sub-komponenty dla parent_name,
    - próbuje dociągnąć średnie godziny z component_patterns (embedding),
    - fallback do heurystyk,
    - skaluje konserwatywnością.

    Zwrót: list[{"name","qty","layout_add","detail_add","doc_add","reason","source","confidence"}]
    """
    pkey = canonicalize_name(parent_name or "")
    if not pkey:
        return []

    rows = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT sub_name, occurrences, total_qty, confidence
                FROM component_bundles
                WHERE department=%s AND parent_key=%s AND occurrences >= %s
                ORDER BY occurrences DESC
                LIMIT %s
            """, (department, pkey, min_occ, top_k))
            rows = cur.fetchall() or []
    except Exception as e:
        logger.warning(f"propose_bundles_for_component query err: {e}")
        return []

    proposals = []
    for r in rows:
        occ = max(1, int(r.get('occurrences') or 1))
        total_qty = float(r.get('total_qty') or 0.0)
        typical_qty = int(round(total_qty / occ)) if total_qty > 0 else 1
        typical_qty = max(1, typical_qty)

        # dociągnięcie wzorca godzin
        try:
            similar = find_similar_components(conn, r['sub_name'], department, limit=1)
        except Exception:
            similar = []

        if similar:
            s0 = similar[0]
            l = float(s0.get('avg_hours_3d_layout') or 0.0)
            d = float(s0.get('avg_hours_3d_detail') or 0.0)
            dc = float(s0.get('avg_hours_2d') or 0.0)
            tot = float(s0.get('avg_hours_total') or (l + d + dc))
            if tot > 0 and (l + d + dc) == 0:
                # rozbicie domyślne, jeśli tylko total dostępny
                l, d, dc = tot * 0.3, tot * 0.5, tot * 0.2

            proposals.append({
                "name": r['sub_name'],
                "qty": typical_qty,
                "layout_add": l * typical_qty * conservativeness,
                "detail_add": d * typical_qty * conservativeness,
                "doc_add": dc * typical_qty * conservativeness,
                "reason": f"Historia: często występuje z {parent_name} (occ={occ})",
                "source": "bundle",
                "confidence": float(r.get('confidence') or 0.5)
            })
        else:
            # fallback do heurystyki
            l, d, dc, why = heuristic_estimate_for_name(r['sub_name'])
            if l + d + dc > 0:
                proposals.append({
                    "name": r['sub_name'],
                    "qty": typical_qty,
                    "layout_add": l * typical_qty * conservativeness,
                    "detail_add": d * typical_qty * conservativeness,
                    "doc_add": dc * typical_qty * conservativeness,
                    "reason": why or f"Historia (bundle) bez wzorca",
                    "source": "bundle_heuristic",
                    "confidence": 0.35
                })

    return proposals

=== DEMO / PRÓBNE DANE ===
def generate_sample_excel() -> bytes:
    """
    Generuje przykładowy Excel pasujący do parsera:
    - Multipliers w wierszu 10 (index 9): kolumny H, J, L (7,9,11)
    - Dane od wiersza 12 (index 11)
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # utwórz pusty arkusz
        pd.DataFrame().to_excel(writer, sheet_name='Dane', index=False)
        ws = writer.sheets['Dane']

        # Multipliers (wiersz 10 zero-based -> index 9)
        ws.write(9, 7, 1.0)   # Layout
        ws.write(9, 9, 1.0)   # Detail
        ws.write(9, 11, 1.0)  # Doc

        # Nagłówki (opcjonalnie)
        headers = ["Pozycja", "Opis", "Komentarz", "Części std", "Części spec", "", "", "Layout [h]", "", "Detail [h]", "", "Doc [h]"]
        for col, h in enumerate(headers):
            ws.write(10, col, h)  # wiersz 11 (index 10)

        # Dane od wiersza 12 (index 11)
        row = 11
        # Złożenie główne (sumaryczne)
        ws.write(row, 0, "1,0"); ws.write(row, 1, "Stacja dociskania omega (złożenie)"); row += 1

        # Komponent 1: z komentarzem z sub-komponentami
        ws.write(row, 0, "1,1")
        ws.write(row, 1, "Dociski omega boczna; blachy")
        ws.write(row, 2, "2x - docisk śrubowy odrzucany; śruba trapezowa; konsola docisku")
        ws.write(row, 7, 2.0)  # Layout
        ws.write(row, 9, 6.0)  # Detail
        ws.write(row, 11, 3.0) # Doc
        row += 1

        # Komponent 2
        ws.write(row, 0, "1,2")
        ws.write(row, 1, "Konsola główna")
        ws.write(row, 2, "płyta montażowa; 4x wspornik; osłona boczna")
        ws.write(row, 7, 1.0); ws.write(row, 9, 4.0); ws.write(row, 11, 2.0)
        row += 1

        # Komponent 3
        ws.write(row, 0, "1,3")
        ws.write(row, 1, "Płyta bazowa z otworami")
        ws.write(row, 2, "8x otwór M12; fazowanie")
        ws.write(row, 7, 0.5); ws.write(row, 9, 2.5); ws.write(row, 11, 1.0)
        row += 1

    output.seek(0)
    return output.getvalue()

def generate_sample_pdf() -> bytes:
    """
    Generuje prosty PDF (wymaga reportlab). Jeśli brak, zwraca None.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        textobject = c.beginText(40, 800)
        lines = [
            "Specyfikacja: Stacja dociskania omega (boczna)",
            "- Wymagane dociski śrubowe z mechanizmem odrzucania",
            "- Konsola docisku i płyta bazowa",
            "- Śruby trapezowe w mechanizmie odrzutu",
            "Normy: ISO 12100, EN 1090",
            "Uwagi: kinematyka docisku, docisk boczny, kontrola luzu"
        ]
        for l in lines:
            textobject.textLine(l)
        c.drawText(textobject)
        c.showPage()
        c.save()
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Nie można wygenerować PDF (brak reportlab?): {e}")
        return None

def generate_sample_image() -> bytes:
    """
    Generuje prosty obraz PNG (schemat blokowy) dla testu multimodalnego.
    """
    w, h = 800, 400
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    # Ramka
    draw.rectangle([20, 20, w-20, h-20], outline=(50, 50, 50), width=3)
    # Elementy
    draw.rectangle([60, 150, 220, 250], outline="navy", width=3) # baza
    draw.text((70, 260), "Płyta bazowa", fill="navy")
    draw.rectangle([300, 120, 520, 180], outline="darkgreen", width=3) # docisk
    draw.text((310, 185), "Docisk śrubowy", fill="darkgreen")
    draw.line([520, 150, 700, 150], fill="black", width=3) # odrzut
    draw.text((600, 160), "Odrzut", fill="black")
    # Tekst tytułu
    draw.text((30, 30), "Stacja dociskania omega (schemat poglądowy)", fill=(0, 0, 0))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

def fill_demo_fields():
    """
    Wypełnia formularz przykładowymi danymi w sesji.
    """
    st.session_state['project_name'] = "Stacja dociskania omega - DEMO"
    st.session_state['client'] = "Klient Demo sp. z o.o."
    st.session_state['description'] = (
        "Stacja dociskania detalu typu omega z dociskami bocznymi. "
        "Wymagania: mechanizm odrzutu docisku śrubowego, konsola docisku, płyta bazowa. "
        "Normy: ISO 12100, EN 1090. Złożoność średnia, kinematyka docisków."
    )
    st.success("Wypełniono formularz przykładowymi danymi.")

=== UI: Dashboard, Nowy projekt, Historia ===
def render_dashboard_page():
    st.header("📊 Dashboard")
    with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*) as count FROM projects")
        project_count = cur.fetchone()['count']
        cur.execute("SELECT AVG(accuracy) as avg FROM projects WHERE accuracy IS NOT NULL")
        avg_accuracy = (cur.fetchone() or {}).get('avg') or 0
        cur.execute("""
        SELECT department, COUNT(*) as count
        FROM projects WHERE department IS NOT NULL
        GROUP BY department ORDER BY department
        """)
        dept_stats = cur.fetchall()

    col1, col2 = st.columns(2)
    col1.metric("Projekty", project_count)
    col2.metric("Średnia dokładność", f"{avg_accuracy*100:.1f}%")

    if dept_stats:
        st.subheader("Projekty wg działów")
        df_dept = pd.DataFrame(dept_stats)
        df_dept['department_name'] = df_dept['department'].map(DEPARTMENTS)
        st.bar_chart(df_dept.set_index('department_name')['count'])

    st.header("🔍 Wyszukaj projekty")
    search_dept = st.selectbox("Dział", options=[''] + list(DEPARTMENTS.keys()),
                               format_func=lambda x: 'Wszystkie' if x == '' else f"{x} - {DEPARTMENTS[x]}")
    search_query = st.text_input("Słowa kluczowe")
    if search_query:
        with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            if search_dept:
                cur.execute("""
                    SELECT id, name, client, department, estimated_hours, description
                    FROM projects WHERE department = %s
                    AND to_tsvector('simple', coalesce(name,'') || ' ' || coalesce(client,'') || ' ' || coalesce(description,'')) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY created_at DESC LIMIT 10
                """, (search_dept, search_query))
            else:
                cur.execute("""
                    SELECT id, name, client, department, estimated_hours, description
                    FROM projects
                    WHERE to_tsvector('simple', coalesce(name,'') || ' ' || coalesce(client,'') || ' ' || coalesce(description,'')) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY created_at DESC LIMIT 10
                """, (search_query,))
            results = cur.fetchall()
        if results:
            st.write(f"Znaleziono {len(results)} projektów:")
            df_results = pd.DataFrame(results)
            df_results['department_name'] = df_results['department'].map(DEPARTMENTS)
            st.dataframe(df_results, use_container_width=True)

            selected_project = st.selectbox(
                "Historia wersji",
                options=results,
                format_func=lambda p: f"{p['name']} ({p['department']})"
            )
            if selected_project:
                with get_db_connection() as conn:
                    versions = get_project_versions(conn, selected_project['id'])
                if versions:
                    st.subheader(f"📜 Historia: {selected_project['name']}")
                    for v in versions:
                        with st.expander(f"{v['version']} - {v['created_at'].strftime('%Y-%m-%d %H:%M')} {'✅' if v['is_approved'] else ''}", expanded=(v == versions[0])):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Layout", f"{v['estimated_hours_3d_layout']:.1f}h")
                            col2.metric("Detail", f"{v['estimated_hours_3d_detail']:.1f}h")
                            col3.metric("2D", f"{v['estimated_hours_2d']:.1f}h")
                            st.metric("TOTAL", f"{v['estimated_hours']:.1f}h")
                            if v['change_description']:
                                st.text_area("Opis", v['change_description'], height=100, disabled=True, key=f"d_{v['id']}")
                            st.caption(f"Autor: {v['changed_by']}")
        else:
            st.info("Nie znaleziono")

def render_new_project_page(selected_model):
    st.header("🆕 Nowy Projekt")

    department = st.selectbox(
        "Wybierz dział*",
        options=list(DEPARTMENTS.keys()),
        format_func=lambda x: f"{x} - {DEPARTMENTS[x]}",
        key="department"
    )

    st.info(f"📋 {DEPARTMENT_CONTEXT[department]}")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Nazwa projektu*", key="project_name")
        st.text_input("Klient", key="client")
        st.text_area("Opis", height=200, key="description")
    with col2:
        excel_file = st.file_uploader("Excel", type=['xlsx', 'xls'])
        image_files = st.file_uploader("Zdjęcia/Rysunki", type=['jpg', 'png'], accept_multiple_files=True)
        pdf_files = st.file_uploader("PDF", type=['pdf'], accept_multiple_files=True)

    # Parametry sugestii
    st.subheader("⚙️ Uwzględnianie komentarzy")
    use_comments = st.checkbox("Uwzględnij sub-komponenty z komentarzy w estymacji", value=True)
    conserv = st.slider("Konserwatywność proponowanych dodatków", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
    enable_bundles = st.checkbox("Włącz podpowiedzi z historii (bundles)", value=True,
                                 help="Podpowiada typowe sub‑komponenty dla podobnych pozycji na bazie importów historycznych")
   
    # 🔹 AI Brief: opis zadania i checklista
    st.subheader("📝 AI: Opis zadania i checklista")
    if st.button("📝 Generuj opis zadania (AI)", use_container_width=True):
        try:
            # Zbuduj wejście (bez wywoływania pełnej analizy)
            components_for_brief = []
            if excel_file is not None:
                # Parsuj bez side-effectów UI
                try:
                    components_for_brief = parse_cad_project_structured_with_xlsx_comments(BytesIO(excel_file.getvalue()))['components']
                except Exception:
                    components_for_brief = []
    
            pdf_text_for_brief = ""
            if pdf_files:
                pdf_text_for_brief = "\n".join([extract_text_from_pdf(pf) for pf in pdf_files])
    
            prompt_brief = build_brief_prompt(
                st.session_state.get("description", ""),
                components_for_brief,
                pdf_text_for_brief,
                department
            )
    
            ai_model_brief = selected_model or "llama3:latest"
            resp = query_ollama(prompt_brief, model=ai_model_brief, format_json=True)
            brief = parse_brief_response(resp)
            st.session_state["ai_brief"] = brief
            st.success("Opis wygenerowany ✅")
        except Exception as e:
            logger.exception("Brief generation failed")
            st.error(f"Nie udało się wygenerować opisu: {e}")
    
    # Wyświetl brief (jeśli jest w sesji)
    if "ai_brief" in st.session_state:
        b = st.session_state["ai_brief"]
        if b.get("brief_md"):
            st.markdown(b["brief_md"])
        cols = st.columns(2)
        with cols[0]:
            if b.get("missing_info"):
                st.markdown("**Brakujące informacje / pytania:**")
                for it in b["missing_info"]:
                    st.write(f"• {it}")
            if b.get("assumptions"):
                st.markdown("**Założenia:**")
                for it in b["assumptions"]:
                    st.write(f"• {it}")
            if b.get("scope"):
                st.markdown("**Zakres:**")
                for it in b["scope"]:
                    st.write(f"• {it}")
        with cols[1]:
            if b.get("risks"):
                st.markdown("**Ryzyka:**")
                for r in b["risks"]:
                    st.write(f"• {r.get('risk','')} (impact: {r.get('impact','')})")
                    if r.get("mitigation"):
                        st.caption(f"Mitigation: {r['mitigation']}")
            if b.get("checklist"):
                st.markdown("**Checklist:**")
                for it in b["checklist"]:
                    st.write(f"☑︎ {it}")
            if b.get("open_questions"):
                st.markdown("**Otwarte pytania:**")
                for it in b["open_questions"]:
                    st.write(f"• {it}")
    
        # Pobranie do .md
        md_export = "# Opis zadania (AI)\n\n" + b.get("brief_md","") + "\n\n"
        if b.get("missing_info"):
            md_export += "## Brakujące informacje\n" + "\n".join([f"- {x}" for x in b["missing_info"]]) + "\n\n"
        if b.get("assumptions"):
            md_export += "## Założenia\n" + "\n".join([f"- {x}" for x in b["assumptions"]]) + "\n\n"
        if b.get("scope"):
            md_export += "## Zakres\n" + "\n".join([f"- {x}" for x in b["scope"]]) + "\n\n"
        if b.get("risks"):
            md_export += "## Ryzyka\n" + "\n".join([f"- {r['risk']} (impact: {r['impact']}) — {r.get('mitigation','')}" for r in b["risks"]]) + "\n\n"
        if b.get("checklist"):
            md_export += "## Checklist\n" + "\n".join([f"- {x}" for x in b["checklist"]]) + "\n\n"
        if b.get("open_questions"):
            md_export += "## Otwarte pytania\n" + "\n".join([f"- {x}" for x in b["open_questions"]]) + "\n\n"

    st.download_button("⬇️ Pobierz opis (.md)", md_export.encode("utf-8"),
                       file_name=f"opis_zadania_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                       mime="text/markdown")


    
    if st.button("🤖 Analizuj z AI", use_container_width=True):
        if not st.session_state.get("description") and not excel_file and not image_files and not pdf_files:
            st.warning("Podaj opis lub wgraj pliki")
        else:
            progress_bar = st.progress(0, text="Startuję...")
            try:
                components_from_excel = []
                if excel_file:
                    progress_bar.progress(15, text="Wczytuję Excel...")
                    components_from_excel = process_excel(excel_file)

                images_b64 = []
                if image_files:
                    progress_bar.progress(25, text="Analizuję obrazy...")
                    for img in image_files:
                        images_b64.append(encode_image_b64(img))

                pdf_text = ""
                if pdf_files:
                    progress_bar.progress(30, text="PDF...")
                    pdf_text = "\n".join([extract_text_from_pdf(pf) for pf in pdf_files])

                # Wzorce
                with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT name, avg_hours_total, avg_hours_3d_layout,
                               avg_hours_3d_detail, avg_hours_2d,
                               proportion_layout, proportion_detail, proportion_doc, occurrences
                        FROM component_patterns
                        WHERE department = %s AND occurrences > 2
                        ORDER BY occurrences DESC LIMIT 20
                    """, (department,))
                    learned_patterns = cur.fetchall()

                st.write(f"🧠 {len(learned_patterns)} wzorców z działu {department}")

                prompt = build_analysis_prompt(st.session_state.get("description", ""), components_from_excel, learned_patterns, pdf_text, department)

                if images_b64 and model_available("llava"):
                    ai_model = "llava:13b" if model_available("llava:13b") else "llava:latest"
                else:
                    ai_model = selected_model or "llama3:latest"

                progress_bar.progress(60, text=f"AI ({ai_model})...")
                ai_text = query_ollama(prompt, model=ai_model, images_b64=images_b64, format_json=True)

                progress_bar.progress(80, text="Parsuję...")
                parsed = parse_ai_response(ai_text, components_from_excel=components_from_excel)

                # Kategoryzacja
                if parsed.get('components'):
                    for c in parsed['components']:
                        if not c.get('is_summary', False):
                            c['category'] = categorize_component(c.get('name', ''))

                st.session_state["ai_analysis"] = parsed
                st.session_state["base_components"] = parsed.get('components', [])
                st.session_state["ai_adjustments"] = parsed.get('ai_adjustments', [])

                # Propozycje wzorce/heurystyki (zależnie od checkboxa i suwaka)
                if use_comments:
                    with get_db_connection() as conn:
                        proposals = propose_adjustments_for_components(conn, st.session_state["base_components"], department, conserv)
                    st.session_state["rule_adjustments"] = proposals
                else:
                    st.session_state["rule_adjustments"] = []

                progress_bar.progress(100, text="Gotowe ✅")
                time.sleep(0.6)
                progress_bar.empty()

            except Exception as e:
                logger.exception("Analiza failed")
                st.error(f"Błąd: {e}")

    if "ai_analysis" in st.session_state:
        analysis = st.session_state["ai_analysis"]
        base_components = st.session_state.get("base_components", [])
        ai_adjustments = st.session_state.get("ai_adjustments", [])
        rule_adjustments = st.session_state.get("rule_adjustments", [])

        # Bundles z historii dla komponentów bez sub‑komponentów
        bundle_adjustments = []
        if enable_bundles and base_components:
            with get_db_connection() as conn:
                for comp in base_components:
                    if comp.get('is_summary'):
                        continue
                    # Preferencyjnie sugeruj dla tych BEZ komentarzy/subkomponentów
                    if comp.get('subcomponents'):
                        continue
                    adds = propose_bundles_for_component(conn, comp.get('name',''), department, conservativeness=conserv)
                    if adds:
                        bundle_adjustments.append({"parent": comp.get('name',''), "adds": adds})

        # Połącz do jednego źródła “regułowego”
        combined_rule_adjustments = (rule_adjustments or []) + (bundle_adjustments or [])

        st.subheader("Wynik analizy")
        for w in analysis.get("warnings", []):
            st.warning(w)

        layout_h = analysis.get("total_layout", 0.0)
        detail_h = analysis.get("total_detail", 0.0)
        doc_h = analysis.get("total_2d", 0.0)
        estimated_hours = max(0.0, layout_h + detail_h + doc_h)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Layout", f"{layout_h:.1f}h")
        col2.metric("Detail", f"{detail_h:.1f}h")
        col3.metric("2D", f"{doc_h:.1f}h")
        hourly_rate = st.sidebar.number_input("Stawka PLN/h", min_value=1, max_value=1000, value=150, step=10)
        col4.metric("TOTAL", f"{estimated_hours:.1f}h", delta=f"{(estimated_hours * hourly_rate):.0f} PLN")

        # Proponowane dodatki (AI)
        st.subheader("💡 Proponowane dodatki (AI z komentarzy)")
        ai_selected = []
        if ai_adjustments:
            for i, adj in enumerate(ai_adjustments):
                parent = adj.get("parent", "komponent")
                with st.expander(f"AI: {parent}"):
                    for j, add in enumerate(adj.get("adds", [])):
                        key = f"ai_adj_{i}_{j}"
                        default = True
                        checked = st.checkbox(
                            f"{add['qty']}x {add['name']} → +L {add['layout_add']:.1f}h, +D {add['detail_add']:.1f}h, +2D {add['doc_add']:.1f}h",
                            value=default, key=key
                        )
                        st.caption(f"Powód: {add.get('reason','')}")
                        if checked:
                            ai_selected.append({"parent": parent, "add": add})
        else:
            st.caption("Brak propozycji AI lub model nie zwrócił 'adjustments'.")

        # Proponowane dodatki (Wzorce/Heurystyki)
        st.subheader("🧠 Proponowane dodatki (wzorce/heurystyki + historia bundles)")
        rule_selected = []
        if combined_rule_adjustments:
            for i, adj in enumerate(combined_rule_adjustments):
                parent = adj.get("parent", "komponent")
                with st.expander(f"Wzorce/Heurystyki/Historia: {parent}"):
                    for j, add in enumerate(adj.get("adds", [])):
                        key = f"rule_adj_{i}_{j}"
                        # domyślnie zaznacz: wzorzec z patterns (wyższa pewność); bundle/heurystyczne odznaczone
                        default = True if add.get("source") == "pattern" else False
                        checked = st.checkbox(
                            f"{add['qty']}x {add['name']} → +L {add['layout_add']:.1f}h, +D {add['detail_add']:.1f}h, +2D {add['doc_add']:.1f}h  ({add.get('source','')}, conf={add.get('confidence',0):.2f})",
                            value=default, key=key
                        )
                        st.caption(f"Powód: {add.get('reason','')}")
                        if checked:
                            rule_selected.append({"parent": parent, "add": add})
        else:
            st.caption("Brak propozycji z komentarzy/historii lub funkcja wyłączona.")

        # Zbuduj komponenty z zaakceptowanych dodatków (AI + reguły)
        adjustment_components = []
        for src, group in [("AI", ai_selected), ("RULE", rule_selected)]:
            for item in group:
                parent = item["parent"]
                add = item["add"]
                comp = {
                    "name": f"ADJ: {add['name']} (x{add['qty']})",
                    "hours_3d_layout": float(add["layout_add"]),
                    "hours_3d_detail": float(add["detail_add"]),
                    "hours_2d": float(add["doc_add"]),
                    "hours": float(add["layout_add"] + add["detail_add"] + add["doc_add"]),
                    "is_adjustment": True,
                    "parent": parent,
                    "source": src
                }
                adjustment_components.append(comp)

        st.subheader("🔧 Edytuj wycenę (komponenty bazowe)")
        final_components_base = []
        if base_components:
            parts_only = [
                c for c in base_components
                if not c.get('is_summary', False) and c.get('name') not in ['[part]', '[assembly]', '', ' ']
            ]
            st.caption(f"ℹ️ Pokazano {len(parts_only)} komponentów")
            for i, comp in enumerate(parts_only):
                display_name = comp['name'][:50] + "..." if len(comp['name']) > 50 else comp['name']
                with st.expander(f"{display_name} - {comp.get('hours', 0):.1f}h"):
                    st.markdown(f"**Pełna nazwa:** {comp['name']}")
                    c1, c2, c3 = st.columns(3)
                    new_layout = c1.number_input("Layout", value=float(comp.get('hours_3d_layout', 0)), key=f"l_{i}")
                    new_detail = c2.number_input("Detail", value=float(comp.get('hours_3d_detail', 0)), key=f"d_{i}")
                    new_doc = c3.number_input("2D", value=float(comp.get('hours_2d', 0)), key=f"doc_{i}")
                    comp2 = dict(comp)
                    comp2['hours_3d_layout'] = new_layout
                    comp2['hours_3d_detail'] = new_detail
                    comp2['hours_2d'] = new_doc
                    comp2['hours'] = new_layout + new_detail + new_doc
                    final_components_base.append(comp2)
                    if comp.get('subcomponents'):
                        st.markdown("**Zawiera:**")
                        for sub in comp['subcomponents']:
                            qty = sub.get('quantity', 1)
                            st.text(f"  • {qty}x {sub['name']}" if qty > 1 else f"  • {sub['name']}")
                    if comp.get('comment'):
                        st.caption(f"💬 {comp['comment']}")

        # Połączenie bazowych i dodatków
        combined_components = final_components_base + adjustment_components

        # Współczynniki z Excela
        if "excel_multipliers" in st.session_state and combined_components:
            st.subheader("📊 Współczynniki z Excela")
            mult = st.session_state["excel_multipliers"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Layout", f"x{mult['layout']:.2f}")
            c2.metric("Detail", f"x{mult['detail']:.2f}")
            c3.metric("Doc", f"x{mult['documentation']:.2f}")
            apply_mult = st.checkbox("Zastosuj współczynniki do wszystkich pozycji (w tym dodatków)", value=False, key="apply_mult")
            if apply_mult:
                combined_scaled = []
                for c in combined_components:
                    c2 = dict(c)
                    c2['hours_3d_layout'] = c.get('hours_3d_layout', 0) * mult['layout']
                    c2['hours_3d_detail'] = c.get('hours_3d_detail', 0) * mult['detail']
                    c2['hours_2d'] = c.get('hours_2d', 0) * mult['documentation']
                    c2['hours'] = c2['hours_3d_layout'] + c2['hours_3d_detail'] + c2['hours_2d']
                    combined_scaled.append(c2)
                combined_components = combined_scaled

        # Podsumowanie końcowe
        sum_layout = sum(c.get('hours_3d_layout', 0) for c in combined_components if not c.get('is_summary', False))
        sum_detail = sum(c.get('hours_3d_detail', 0) for c in combined_components if not c.get('is_summary', False))
        sum_doc = sum(c.get('hours_2d', 0) for c in combined_components if not c.get('is_summary', False))
        sum_total = sum_layout + sum_detail + sum_doc

        st.metric("🔢 Suma (po dodatkach i multipliers)", f"{sum_total:.1f}h")

        st.subheader("🗂️ Harmonogram")
        show_project_timeline(combined_components)

        # Podobne projekty
        with get_db_connection() as conn:
            similar = find_similar_projects(conn, st.session_state.get("description"), department)
        st.subheader(f"📊 Podobne projekty ({department})")
        if similar:
            for proj in similar:
                cc1, cc2, cc3 = st.columns([3,1,1])
                cc1.write(f"**{proj['name']}** ({proj['client'] or '-'})")
                cc2.metric("Szacowano", f"{(proj['estimated_hours'] or 0):.1f}h")
                if proj['actual_hours']:
                    cc3.metric("Rzeczywiście", f"{proj['actual_hours']:.1f}h")
        else:
            st.info("Brak podobnych")

        with get_db_connection() as conn:
            similar_sem = find_similar_projects_semantic(conn, st.session_state.get("description"), department)
        st.subheader(f"🧭 Semantycznie podobne projekty (pgvector)")
        if similar_sem:
            for sp in similar_sem:
                sim_pct = sp['similarity'] * 100
                st.write(f"- **{sp['name']}** (sim={sim_pct:.0f}%) — est: {(sp['estimated_hours'] or 0):.1f}h" +
                         (f", act: {sp['actual_hours']:.1f}h" if sp['actual_hours'] else ""))
        else:
            st.caption("Brak embeddingów — dodaj projekty i uruchom przeliczanie")

        st.subheader("📤 Eksport")
        if st.button("📥 Export do Excel"):
            excel_data = export_quotation_to_excel({
                'name': st.session_state.get("project_name"),
                'client': st.session_state.get("client"),
                'department': department,
                'components': combined_components,
                'total_hours': sum_total
            })
            st.download_button("⬇️ Pobierz", excel_data,
                               file_name=f"wycena_{st.session_state.get('project_name','p')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.subheader("💾 Zapisz projekt")
        c1, c2 = st.columns([3,1])
        with c1:
            change_desc = st.text_input("Opis zmian", placeholder="np. 'Pierwsza wycena'")
        with c2:
            is_approved = st.checkbox("Zatwierdzone", value=False)

        if st.button("💾 Zapisz", type="primary", use_container_width=True):
            errors = []
            name = st.session_state.get("project_name")
            if not name or not name.strip():
                errors.append("Nazwa nie może być pusta")
            if sum_total < 0:
                errors.append("Godziny < 0")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    components_to_save = [c for c in combined_components if not c.get('is_summary', False)]
                    with get_db_connection() as conn, conn.cursor() as cur:
                        cur.execute("SELECT id FROM projects WHERE name = %s AND department = %s",
                                    (name, department))
                        existing = cur.fetchone()
                        if existing:
                            project_id = existing[0]
                            cur.execute("SELECT COUNT(*) FROM project_versions WHERE project_id = %s", (project_id,))
                            version_num = f"v1.{cur.fetchone()[0] + 1}"
                            cur.execute("""
                                UPDATE projects SET components = %s,
                                estimated_hours_3d_layout = %s, estimated_hours_3d_detail = %s,
                                estimated_hours_2d = %s, estimated_hours = %s,
                                ai_analysis = %s, updated_at = NOW()
                                WHERE id = %s
                            """, (json.dumps(components_to_save, ensure_ascii=False),
                                  float(sum_layout), float(sum_detail),
                                  float(sum_doc), float(sum_total),
                                  analysis["raw_text"], project_id))

                            save_project_version(conn, project_id, version_num, components_to_save,
                                                sum_total, sum_layout, sum_detail, sum_doc,
                                                change_desc or "", "System")

                            if is_approved:
                                cur.execute("UPDATE project_versions SET is_approved = TRUE WHERE project_id = %s AND version = %s",
                                          (project_id, version_num))
                            conn.commit()
                            st.success(f"✅ Zaktualizowano! {version_num}")
                        else:
                            cur.execute("""
                                INSERT INTO projects (name, client, department, description, components,
                                estimated_hours_3d_layout, estimated_hours_3d_detail, estimated_hours_2d,
                                estimated_hours, ai_analysis)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                            """, (
                                (st.session_state.get("project_name") or "").strip(),
                                (st.session_state.get("client") or "").strip(),
                                department, (st.session_state.get("description") or "").strip(),
                                json.dumps(components_to_save, ensure_ascii=False),
                                float(sum_layout), float(sum_detail),
                                float(sum_doc), float(sum_total),
                                analysis["raw_text"]
                            ))
                            project_id = cur.fetchone()[0]
                            ensure_project_embedding(cur, project_id, st.session_state.get("description", ""))

                            save_project_version(conn, project_id, "v1.0", components_to_save,
                                                sum_total, sum_layout, sum_detail, sum_doc,
                                                change_desc or "Pierwsza wycena", "System")

                            if is_approved:
                                cur.execute("UPDATE project_versions SET is_approved = TRUE WHERE project_id = %s AND version = 'v1.0'",
                                          (project_id,))
                            conn.commit()
                            st.success(f"✅ Zapisano! ID: {project_id}")

                        logger.info(f"Zapisano: {project_id} - {st.session_state.get('project_name')}")
                        st.balloons()
                        time.sleep(1.2)
                        st.rerun()
                except Exception as e:
                    st.error(f"Błąd: {e}")
                    logger.exception("Zapis failed")

def batch_import_excels(files, department: str, learn_from_import: bool = False, distribute: str = 'qty'):
    """
    Batch import historycznych plików Excel:
    - parsuje wartości + komentarze/note (openpyxl),
    - zapisuje projekt jako Excel-only (is_historical/locked),
    - opcjonalnie uczy wzorce (komponenty + sub-komponenty) oraz bundles.

    distribute: 'qty' lub 'equal' (dotyczy podziału godzin na sub-komponenty jeśli learn_from_import=True).
    Zwraca listę wyników {file, status, project_id?, hours?, error?}.
    """
    results = []
    with get_db_connection() as conn, conn.cursor() as cur:
        for f in files:
            try:
                # Nazwa projektu z nazwy pliku
                fname = getattr(f, "name", "import.xlsx")
                proj_name = os.path.splitext(os.path.basename(fname))[0]

                # Parsuj (wartości + komentarze komórek)
                parsed = parse_cad_project_structured_with_xlsx_comments(f)
                comps_full = parsed.get('components', []) or []
                totals = parsed.get('totals', {}) or {}

                est_l = float(totals.get('layout', 0) or 0)
                est_d = float(totals.get('detail', 0) or 0)
                est_doc = float(totals.get('documentation', 0) or 0)
                est_total = float(totals.get('total', est_l + est_d + est_doc) or 0)

                # Zapisz projekt jako historyczny, zablokowany dla AI
                cur.execute("""
                    INSERT INTO projects (
                        name, client, department, description, components,
                        estimated_hours_3d_layout, estimated_hours_3d_detail, estimated_hours_2d,
                        estimated_hours, ai_analysis,
                        is_historical, estimation_mode, totals_source, locked_totals
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                             TRUE, 'excel_only', 'excel', TRUE)
                    RETURNING id
                """, (
                    proj_name, None, department, None,
                    json.dumps(comps_full, ensure_ascii=False),
                    est_l, est_d, est_doc, est_total,
                    '[HISTORICAL_IMPORT]'
                ))
                pid = cur.fetchone()[0]

                # Uczenie z importu (opcjonalnie)
                if learn_from_import and comps_full:
                    learn_from_historical_components(cur, department, comps_full, distribute=distribute)

                conn.commit()
                results.append({"file": fname, "status": "success", "project_id": pid, "hours": est_total})
            except Exception as e:
                conn.rollback()
                logger.exception("Batch import error")
                results.append({"file": getattr(f, 'name', 'unknown'), "status": "error", "error": str(e)})
    return results

def render_history_page():
    st.header("📚 Historia i Uczenie")
    tab1, tab2, tab3 = st.tabs(["✏️ Feedback", "🧠 Wzorce", "📦 Batch Import"])

    with tab1:
        st.subheader("Dodaj feedback")
        feedback_dept = st.selectbox("Dział", options=[''] + list(DEPARTMENTS.keys()),
                                     format_func=lambda x: 'Wszystkie' if x == '' else f"{x} - {DEPARTMENTS[x]}")

        with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            if feedback_dept:
                cur.execute("""
                    SELECT id, name, department, estimated_hours
                    FROM projects WHERE actual_hours IS NULL AND department = %s
                    ORDER BY created_at DESC
                """, (feedback_dept,))
            else:
                cur.execute("SELECT id, name, department, estimated_hours FROM projects WHERE actual_hours IS NULL ORDER BY created_at DESC")
            pending = cur.fetchall()

        if pending:
            proj = st.selectbox("Projekt", options=pending,
                               format_func=lambda p: f"[{p['department']}] {p['name']} (ID: {p['id']}) | est: {p['estimated_hours']:.1f}h")
            actual_hours = st.number_input("Rzeczywiste godziny", min_value=0.0, step=0.5, value=float(proj['estimated_hours']))

            if st.button("💾 Zapisz feedback", type="primary"):
                if actual_hours <= 0:
                    st.error("Godziny > 0")
                else:
                    with get_db_connection() as conn, conn.cursor() as cur:
                        estimated = float(proj['estimated_hours'])
                        accuracy = 1 - abs(estimated - actual_hours) / estimated if estimated > 0 else 0

                        cur.execute("UPDATE projects SET actual_hours = %s, accuracy = %s WHERE id = %s",
                                  (actual_hours, accuracy, proj['id']))

                        cur.execute("SELECT components FROM projects WHERE id = %s", (proj['id'],))
                        components_data = (cur.fetchone() or [None])[0] or []

                        if components_data:
                            ratio = actual_hours / estimated if estimated > 0 else 1.0
                            for comp in components_data:
                                if comp.get('is_summary'):
                                    continue

                                layout_est = float(comp.get('hours_3d_layout', 0))
                                detail_est = float(comp.get('hours_3d_detail', 0))
                                doc_est = float(comp.get('hours_2d', 0))
                                total_est = float(comp.get('hours', 0))

                                if total_est > 0:
                                    update_pattern_smart(
                                        cur, comp.get('name', 'nieznany'), proj['department'],
                                        layout_est * ratio, detail_est * ratio,
                                        doc_est * ratio, source='actual'
                                    )

                                    subcomponents = comp.get('subcomponents', [])
                                    if subcomponents:
                                        total_qty = sum(s.get('quantity', 1) for s in subcomponents)
                                        for sub in subcomponents:
                                            qty = sub.get('quantity', 1)
                                            weight = qty / total_qty if total_qty > 0 else 1.0 / len(subcomponents)
                                            sub_layout = layout_est * ratio * weight
                                            sub_detail = detail_est * ratio * weight
                                            sub_doc = doc_est * ratio * weight
                                            update_pattern_smart(cur, sub['name'], proj['department'], sub_layout, sub_detail, sub_doc, source='subcomponent')
                                            logger.info(f"  └─ Sub: {qty}x {sub['name']}")

                        # Aktualizuj baseline kategorii
                        agg_cat = {}
                        for comp in components_data:
                            if comp.get('is_summary'):
                                continue
                            layout_act = float(comp.get('hours_3d_layout', 0)) * ratio
                            detail_act = float(comp.get('hours_3d_detail', 0)) * ratio
                            doc_act = float(comp.get('hours_2d', 0)) * ratio

                            cat = comp.get('category') or categorize_component(comp.get('name',''))
                            agg_cat.setdefault(cat, [0.0,0.0,0.0])
                            agg_cat[cat][0] += layout_act
                            agg_cat[cat][1] += detail_act
                            agg_cat[cat][2] += doc_act

                        for cat, (l,d,dc) in agg_cat.items():
                            update_category_baseline(cur, proj['department'], cat, l, d, dc)

                        conn.commit()
                    st.success("Dziękuję! System zaktualizowany.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("🎉 Wszystkie projekty mają feedback!")

    with tab2:
        st.subheader("Wzorce komponentów")
        pattern_dept = st.selectbox("Filtruj", options=[''] + list(DEPARTMENTS.keys()),
                                   format_func=lambda x: 'Wszystkie' if x == '' else f"{x} - {DEPARTMENTS[x]}")

        with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            if pattern_dept:
                cur.execute("""
                    SELECT name, department, avg_hours_total, avg_hours_3d_layout,
                           avg_hours_3d_detail, avg_hours_2d, proportion_layout,
                           proportion_detail, proportion_doc, occurrences
                    FROM component_patterns
                    WHERE department = %s AND occurrences > 0 ORDER BY occurrences DESC
                """, (pattern_dept,))
            else:
                cur.execute("""
                    SELECT name, department, avg_hours_total, avg_hours_3d_layout,
                           avg_hours_3d_detail, avg_hours_2d, proportion_layout,
                           proportion_detail, proportion_doc, occurrences
                    FROM component_patterns WHERE occurrences > 0
                    ORDER BY department, occurrences DESC
                """)
            patterns = cur.fetchall()

        if patterns:
            df = pd.DataFrame(patterns)
            df['department_name'] = df['department'].map(DEPARTMENTS)
            df['proportion_layout'] = (df['proportion_layout'] * 100).round(1).astype(str) + '%'
            df['proportion_detail'] = (df['proportion_detail'] * 100).round(1).astype(str) + '%'
            df['proportion_doc'] = (df['proportion_doc'] * 100).round(1).astype(str) + '%'
            st.dataframe(df, use_container_width=True)
            st.info(f"{len(patterns)} wzorców")
        else:
            st.info("Brak wzorców")

        with st.expander("🧰 Admin: przelicz embeddingi"):
            if st.button("🔄 Przelicz embeddingi dla istniejących danych"):
                with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT id, description FROM projects WHERE description IS NOT NULL")
                    projects_to_embed = cur.fetchall()

                    cur.execute("SELECT pattern_key, department, name FROM component_patterns WHERE pattern_key IS NOT NULL")
                    patterns_to_embed = cur.fetchall()

                    total_items = len(projects_to_embed) + len(patterns_to_embed)
                    if total_items == 0:
                        st.warning("Brak danych do przeliczenia")
                    else:
                        progress = st.progress(0, text="Przeliczam embeddingi...")
                        for idx, p in enumerate(projects_to_embed):
                            ensure_project_embedding(cur, p['id'], p['description'])
                            progress.progress(int((idx + 1) / total_items * 100), text=f"Projekty: {idx+1}/{len(projects_to_embed)}")
                        for idx, r in enumerate(patterns_to_embed):
                            ensure_pattern_embedding(cur, r['pattern_key'], r['department'], r['name'])
                            progress.progress(int((len(projects_to_embed) + idx + 1) / total_items * 100), text=f"Wzorce: {idx+1}/{len(patterns_to_embed)}")
                        conn.commit()
                        progress.empty()
                        st.success(f"✅ Przeliczono {len(projects_to_embed)} projektów + {len(patterns_to_embed)} wzorców")

    with tab3:
        st.subheader("📦 Batch Import")
        st.info("Import wielu plików Excel naraz")

        batch_dept = st.selectbox("Dział dla importu", options=list(DEPARTMENTS.keys()),
                                 format_func=lambda x: f"{x} - {DEPARTMENTS[x]}", key="batch_dept")

        excel_files = st.file_uploader("Excel (wiele)", type=['xlsx', 'xls'], accept_multiple_files=True, key="batch")
        if excel_files:
            st.write(f"📁 {len(excel_files)} plików")
            for f in excel_files[:10]:
                st.write(f"• {f.name}")
            if len(excel_files) > 10:
                st.write(f"... +{len(excel_files) - 10}")

            # NOWE OPCJE: uczenie z importu + metoda podziału godzin
            learn_from_import = st.checkbox("Ucz wzorce z importu (komponenty + sub‑komponenty z komentarzy)", value=True)
            distribute_method = st.radio(
                "Rozdział godzin na sub‑komponenty",
                options=['qty', 'equal'],
                format_func=lambda v: "Proporcjonalnie do ilości (qty)" if v == 'qty' else "Po równo",
                horizontal=True
            )

            if st.button("🚀 Importuj", type="primary", use_container_width=True):
                st.info(f"Import {len(excel_files)} do {batch_dept}...")
                results = batch_import_excels(excel_files, batch_dept,
                                              learn_from_import=learn_from_import,
                                              distribute=distribute_method)
                success = sum(1 for r in results if r['status'] == 'success')
                errors = sum(1 for r in results if r['status'] == 'error')
                c1, c2 = st.columns(2)
                c1.metric("✅ Sukces", success)
                c2.metric("❌ Błędy", errors)
                st.subheader("Szczegóły")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                if success > 0:
                    st.success(f"🎉 {success} projektów!")
                if errors > 0:
                    st.warning(f"⚠️ {errors} błędów")

def main():
    st.title("🚀 CAD Estimator Pro")

    if not init_db():
        st.stop()

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Nawigacja", ["Dashboard", "Nowy projekt", "Historia i Uczenie"])

    st.sidebar.subheader("Ustawienia AI")
    available_models = [m for m in list_local_models() if "embed" not in m]
    selected_model = st.sidebar.selectbox(
        "Wybierz model AI",
        options=available_models or ["llama3:latest"],
        index=(available_models.index("mistral:7b-instruct") if "mistral:7b-instruct" in available_models else 0) if available_models else 0
    )

    st.sidebar.subheader("Status Systemu")
    st.sidebar.write(f"Ollama AI: {'✅ Połączony' if any(list_local_models()) else '❌ Brak połączenia'}")

    with st.sidebar.expander("Dostępne modele"):
        models = list_local_models()
        if models:
            st.write("\n".join(f"- `{m}`" for m in models))
        else:
            st.write("Brak modeli")

    # DEMO / PRÓBNE DANE
    with st.sidebar.expander("🧪 Demo / Próbne dane", expanded=False):
        if st.button("Wypełnij formularz przykładowymi danymi"):
            fill_demo_fields()
        demo_excel = generate_sample_excel()
        st.download_button("📥 Pobierz przykładowy Excel", demo_excel, file_name="demo_estymacja.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        demo_pdf = generate_sample_pdf()
        if demo_pdf:
            st.download_button("📥 Pobierz przykładowy PDF", demo_pdf, file_name="demo_spec.pdf", mime="application/pdf")
        else:
            st.info("Aby generować PDF, zainstaluj: pip install reportlab")
        demo_img = generate_sample_image()
        st.download_button("📥 Pobierz przykładowy obraz (PNG)", demo_img, file_name="demo_schemat.png", mime="image/png")

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Nowy projekt":
        render_new_project_page(selected_model)
    elif page == "Historia i Uczenie":
        render_history_page()

if __name__ == "__main__":
    main()
