import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json
import os
from datetime import datetime
import base64
import re
import logging
from contextlib import contextmanager
import time
from functools import lru_cache
from PIL import Image
import io

from io import BytesIO
import plotly.express as px
from PyPDF2 import PdfReader
from rapidfuzz import fuzz, process
import unicodedata
#import numpy as np

# === KONFIGURACJA i LOGGING ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cad-estimator")

st.set_page_config(page_title="CAD Estimator Pro", layout="wide", page_icon="🚀")
OLLAMA_URL = os.getenv('OLLAMA_URL', 'https://ollama.polmicai.pl')
DB_HOST = os.getenv('DB_HOST', 'cad-postgres')
DB_NAME = os.getenv('DB_NAME', 'cad_estimator')
DB_USER = os.getenv('DB_USER', 'cad_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'cad_password_2024')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')


# === DZIAŁY ===
DEPARTMENTS = {
    '131': 'Automotive',
    '132': 'Industrial Machinery',
    '133': 'Transportation',
    '134': 'Heavy Equipment',
    '135': 'Special Purpose Machinery'
}
DEPARTMENT_META = {
'131': {
'name': 'Automotive',
'norms': ['IATF 16949', 'ISO 12100', 'ISO 13849-1', 'ISO 2768-mK', 'ISO 5817 B'],
'phase_shares': {'layout': 0.15, 'detail': 0.55, 'doc': 0.30},
'checklist': [
'Tolerancje ISO 2768 na rysunkach',
'Spoiny wg ISO 5817 klasa B',
'Dokumentacja PPAP dostępna'
]
},
'132': {
'name': 'Industrial Machinery',
'norms': ['ISO 12100', 'EN 60204-1', 'ISO 13849-1', 'EN ISO 14120'],
'phase_shares': {'layout': 0.20, 'detail': 0.50, 'doc': 0.30},
'checklist': ['Ocena ryzyka ISO 12100', 'Osłony wg EN ISO 14120']
},
'133': {'name': 'Transportation', 'norms': ['EN 1090','ISO 3834-2'], 'phase_shares': {'layout': 0.18, 'detail': 0.52, 'doc': 0.30}},
'134': {'name': 'Heavy Equipment', 'norms': ['EN 1090','ISO 3834-2'], 'phase_shares': {'layout': 0.20, 'detail': 0.50, 'doc': 0.30}},
'135': {'name': 'Special Purpose Machinery', 'norms': ['ISO 12100','EN 60204-1'], 'phase_shares': {'layout': 0.22, 'detail': 0.48, 'doc': 0.30}},
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

# === SŁOWNIK NORMALIZACJI KOMPONENTÓW ===
COMPONENT_ALIASES = {
    # Wsporniki
    'wspornik': 'bracket',
    'halterung': 'bracket',
    'halter': 'bracket',
    'träger': 'bracket',
    'support': 'bracket',
    'konsole': 'bracket',
    
    # Ramy
    'rama': 'frame',
    'rahmen': 'frame',
    'gestell': 'frame',
    'chassis': 'frame',
    
    # Przenośniki
    'przenośnik': 'conveyor',
    'förderband': 'conveyor',
    'förderer': 'conveyor',
    'transport': 'conveyor',
    
    # Płyty
    'płyta': 'plate',
    'platte': 'plate',
    'sheet': 'plate',
    'panel': 'plate',
    
    # Pokrywy
    'pokrywa': 'cover',
    'deckel': 'cover',
    'abdeckung': 'cover',
    
    # Obudowy
    'obudowa': 'housing',
    'gehäuse': 'housing',
    'casing': 'housing',
    
    # Napędy
    'napęd': 'drive',
    'antrieb': 'drive',
    'actuator': 'drive',
    
    # Cylindry
    'siłownik': 'cylinder',
    'cylinder': 'cylinder',
    'zylinder': 'cylinder',
    
    # Prowadnice
    'prowadnica': 'guide',
    'führung': 'guide',
    'rail': 'guide',
    
    # Osłony
    'osłona': 'shield',
    'schutz': 'shield',
    'guard': 'shield',
    
    # Podstawy
    'podstawa': 'base',
    'basis': 'base',
    'fundament': 'base',
    'sockel': 'base',
    
    # Wały
    'wał': 'shaft',
    'welle': 'shaft',
    'axle': 'shaft',
    
    # Łożyska
    'łożysko': 'bearing',
    'lager': 'bearing',
    
    # Śruby
    'śruba': 'screw',
    'schraube': 'screw',
    'bolt': 'bolt',
}

def extract_component_features(name):
    """Wyciąga cechy charakterystyczne z nazwy."""
    features = []
    
    # 1. Wymiary (liczby + jednostki)
    dimensions = re.findall(r'\d+(?:\.\d+)?\s*(?:mm|cm|m|inch|")', name, re.IGNORECASE)
    features.extend(dimensions)
    
    # 2. Materiał
    materials = ['stal', 'steel', 'stahl', 'aluminium', 'alu', 'plastic', 'tworzywo']
    for mat in materials:
        if mat in name.lower():
            features.append(mat)
            break
    
    # 3. Kształt/typ
    shapes = ['l-shape', 'l-kształt', 'u-shape', 't-shape', 'flat', 'płaski']
    for shape in shapes:
        if shape in name.lower():
            features.append(shape)
    
    # 4. Strona (left/right/center)
    sides = ['left', 'right', 'center', 'lewy', 'prawy', 'środkowy', 'links', 'rechts']
    for side in sides:
        if side in name.lower():
            features.append(side)
    
    # 5. Złożoność (spawany, prosty, etc)
    complexity = ['spawany', 'welded', 'geschweißt', 'simple', 'prosty', 'complex', 'złożony']
    for comp in complexity:
        if comp in name.lower():
            features.append(comp)
    
    return features

def normalize_component_name(name):
    """Normalizuje nazwę ZACHOWUJĄC kluczowe cechy."""
    if not name or not isinstance(name, str):
        return name
    
    # 1. Cleanup
    name = ' '.join(name.split())
    name = name.strip('.,;:-_')
    
    # 2. Wyciągnij cechy PRZED tłumaczeniem
    features = extract_component_features(name)
    
    # 3. Tłumacz bazową nazwę
    name_lower = name.lower()
    normalized_parts = []
    
    for word in name.split():
        word_lower = word.lower()
        
        # Pomiń liczby i jednostki - będą w features
        if re.match(r'\d+(?:\.\d+)?', word) or word_lower in ['mm', 'cm', 'm', 'inch']:
            continue
            
        if word_lower in COMPONENT_ALIASES:
            normalized_parts.append(COMPONENT_ALIASES[word_lower])
        else:
            found = False
            for alias, canonical in COMPONENT_ALIASES.items():
                if alias in word_lower:
                    normalized_parts.append(canonical)
                    found = True
                    break
            if not found and word_lower not in ['i', 'a', 'the', 'der', 'die', 'das']:
                normalized_parts.append(word_lower)
    
    # 4. Zbuduj finalną nazwę: base_name + features
    base = ' '.join(normalized_parts)
    
    if features:
        result = f"{base} {' '.join(features)}"
    else:
        result = base
    
    return result.strip()

def find_best_match(name, existing_names, threshold=85):
    """Znajduje najbardziej podobną nazwę używając fuzzy matching."""
    if not existing_names:
        return None, 0
    
    result = process.extractOne(
        name,
        existing_names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )
    
    if result:
        return result[0], result[1]
    return None, 0
# === EMBEDDINGI I WEKTORY ===
def to_pgvector(vec):
    """Konwertuje listę na format pgvector."""
    if not vec:
        return None
    return '[' + ','.join(f'{float(x):.6f}' for x in vec) + ']'

@st.cache_data(ttl=86400, show_spinner=False)
def get_embedding_ollama(text: str, model: str = EMBED_MODEL) -> list:
    """Pobiera embedding z Ollama."""
    try:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings", 
                         json={"model": model, "prompt": text}, timeout=30)
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Embeddings error: {e}")
        return []

def ensure_project_embedding(cur, project_id: int, description: str):
    """Dodaje embedding do projektu."""
    if not description or len(description.strip()) < 10:
        return
    emb = get_embedding_ollama(description)
    if emb and len(emb) == 768:
        try:
            cur.execute("UPDATE projects SET description_embedding = %s::vector WHERE id=%s",
                       (to_pgvector(emb), project_id))
        except Exception as e:
            logger.warning(f"Embedding failed for project {project_id}: {e}")

def ensure_pattern_embedding(cur, pattern_key: str, dept: str, text_for_embed: str):
    """Dodaje embedding do wzorca."""
    if not text_for_embed or len(text_for_embed.strip()) < 3:
        return
    emb = get_embedding_ollama(text_for_embed)
    if emb and len(emb) == 768:
        try:
            cur.execute("""
                UPDATE component_patterns
                SET name_embedding = %s::vector
                WHERE pattern_key=%s AND department=%s
            """, (to_pgvector(emb), pattern_key, dept))
        except Exception as e:
            logger.warning(f"Embedding failed for pattern {pattern_key}: {e}")


def canonicalize_name(name: str) -> str:
    """Normalizuje nazwę komponentu do porównań i uczenia."""
    if not name:
        return ""
        
    n = name.lower()
    
    # Usuń wymiary z jednostkami
    n = re.sub(r'\b\d+[.,]?\d*\s*(mm|cm|m|kg|t|ton|szt|sztuk|pcs)\b', '', n)
    
    # Usuń samodzielne liczby
    n = re.sub(r'\b\d+[.,]?\d*\b', '', n)
    
    # Mapowanie synonimów
    synonyms = {
        'conveyor': 'przenośnik', 'frame': 'rama', 'jig': 'przyrząd',
        'fixture': 'przyrząd', 'weld': 'spawanie', 'welding': 'spawanie',
        'robotic': 'robot', 'robot': 'robot', 'gripper': 'chwytak',
        'transporter': 'przenośnik', 'carrier': 'nośnik', 'pallet': 'paleta',
        'station': 'stanowisko', 'cell': 'gniazdo', 'assembly': 'montaż',
        'base': 'podstawa'
    }
    
    for eng, pl in synonyms.items():
        n = n.replace(eng, pl)
    
    # Usuń znaki specjalne
    n = re.sub(r'[^a-ząćęłńóśźż\s]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()
    
    return n
def parse_subcomponents_from_comment(comment):
    """
    Parsuje komentarz i wyciąga pod-komponenty z ilościami.
    
    Przykłady:
    "2x Płyta 500x300, 4x Wspornik L100" → [{'quantity': 2, 'name': 'Płyta 500x300'}, ...]
    "Płyta montażowa, Wspornik, 3 śruby M8" → [{'quantity': 1, 'name': 'Płyta montażowa'}, ...]
    """
    if not comment or not isinstance(comment, str):
        return []
    
    subcomponents = []
    
    # Pattern 1: "2x Nazwa", "4 szt Nazwa", "5 Nazwa"
    pattern = r'(\d+)\s*(?:x|szt\.?|sztuk|pcs)?\s*([^,;\n]+?)(?=[,;\n]|$)'
    matches = re.finditer(pattern, comment, re.IGNORECASE)
    
    for match in matches:
        try:
            qty = int(match.group(1))
            name = match.group(2).strip()
            
            # Pomiń jeśli to wymiar (200mm, 5m) lub sama jednostka
            if re.match(r'^\d+\s*(mm|cm|m|kg|ton|h)$', name, re.IGNORECASE):
                continue
            
            # Pomiń bardzo krótkie (prawdopodobnie szum)
            if len(name) < 3:
                continue
            
            subcomponents.append({
                'quantity': qty,
                'name': name
            })
        except (ValueError, IndexError):
            continue
    
    # Pattern 2: Jeśli nie znaleziono liczb, podziel po przecinkach/średnikach
    if not subcomponents:
        separators = r'[,;]'
        parts = [p.strip() for p in re.split(separators, comment) if p.strip()]
        
        for part in parts:
            # Pomiń bardzo krótkie lub same liczby
            if len(part) < 3 or re.match(r'^\d+$', part):
                continue
            
            # Sprawdź czy jest liczba na początku
            num_match = re.match(r'^(\d+)\s*(?:x|szt\.?)?\s*(.+)', part, re.IGNORECASE)
            if num_match:
                try:
                    qty = int(num_match.group(1))
                    name = num_match.group(2).strip()
                    if len(name) >= 3:
                        subcomponents.append({'quantity': qty, 'name': name})
                except:
                    pass
            else:
                # Bez liczby - domyślnie 1 sztuka
                if not re.match(r'^\d+\s*(mm|cm|m|kg|h)$', part, re.IGNORECASE):
                    subcomponents.append({'quantity': 1, 'name': part})
    
    logger.info(f"Parsed {len(subcomponents)} subcomponents from: {comment[:50]}...")
    return subcomponents
def _welford_step(mean, m2, n, x):
    """Algorytm Welforda - aktualizacja średniej i wariancji."""
    if n and n >= 5:
        std = (m2 / max(n - 1, 1)) ** 0.5
        if mean and abs(x - mean) > 2.5 * std:
            return mean, m2, n  # outlier - odrzuć
    n_new = (n or 0) + 1
    delta = x - (mean or 0)
    mean_new = (mean or 0) + delta / n_new
    delta2 = x - mean_new
    m2_new = (m2 or 0) + delta * delta2
    return mean_new, m2_new, n_new

def best_pattern_key(cur, dept: str, key: str, threshold: int = 88) -> str:
    """Fuzzy matching dla pattern_key."""
    cur.execute("SELECT pattern_key FROM component_patterns WHERE pattern_key=%s AND department=%s", (key, dept))
    if cur.fetchone():
        return key
    cur.execute("SELECT DISTINCT pattern_key FROM component_patterns WHERE department=%s AND pattern_key IS NOT NULL", (dept,))
    keys = [r[0] for r in cur.fetchall()]
    if not keys:
        return key
    match, score, _ = process.extractOne(key, keys, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else key

def update_pattern_smart(cur, name, dept, layout_h, detail_h, doc_h, total_h, source='actual'):
    """Welford + outlier + confidence + fuzzy + embedding."""
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
        ml, md, mc, mt, m2l, m2d, m2c, m2t, n = row
        ml, m2l, n = _welford_step(ml, m2l, n, layout_h)
        md, m2d, n = _welford_step(md, m2d, n, detail_h)
        mc, m2c, n = _welford_step(mc, m2c, n, doc_h)
        mt, m2t, n = _welford_step(mt, m2t, n, total)
        
        std_total = (m2t / max(n - 1, 1)) ** 0.5 if n and n > 1 else 0.0
        confidence = min(1.0, n / 10.0) * (1.0 / (1.0 + (std_total / (mt or 1e-6))))
        
        cur.execute("""
            UPDATE component_patterns
            SET avg_hours_3d_layout=%s, avg_hours_3d_detail=%s, avg_hours_2d=%s, avg_hours_total=%s,
                m2_layout=%s, m2_detail=%s, m2_doc=%s, m2_total=%s,
                occurrences=%s, confidence=%s, source=%s,
                last_updated=NOW(),
                last_actual_sample_at=CASE WHEN %s='actual' THEN NOW() ELSE last_actual_sample_at END,
                pattern_key=%s
            WHERE pattern_key=%s AND department=%s
        """, (ml, md, mc, mt, m2l, m2d, m2c, m2t, n, confidence, source, source, key, key, dept))
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
    """Aktualizuje baseline dla kategorii."""
    cur.execute("""
        SELECT mean_layout, mean_detail, mean_doc, m2_layout, m2_detail, m2_doc, occurrences
        FROM category_baselines WHERE department=%s AND category=%s
    """, (dept, category))
    row = cur.fetchone()
    
    if row:
        ml, md, mc, m2l, m2d, m2c, n = row
        ml, m2l, n = _welford_step(ml, m2l, n, layout_h)
        md, m2d, n = _welford_step(md, m2d, n, detail_h)
        mc, m2c, n = _welford_step(mc, m2c, n, doc_h)
        conf = min(1.0, n / 10.0)
        
        cur.execute("""
            UPDATE category_baselines
            SET mean_layout=%s, mean_detail=%s, mean_doc=%s,
                m2_layout=%s, m2_detail=%s, m2_doc=%s,
                occurrences=%s, confidence=%s, last_updated=NOW()
            WHERE department=%s AND category=%s
        """, (ml, md, mc, m2l, m2d, m2c, n, conf, dept, category))
    else:
        cur.execute("""
            INSERT INTO category_baselines (department, category, mean_layout, mean_detail, mean_doc, occurrences, confidence)
            VALUES (%s, %s, %s, %s, %s, 1, 0.1)
        """, (dept, category, layout_h, detail_h, doc_h))

def find_similar_projects_semantic(conn, description, department, limit=5):
    """Semantyczne wyszukiwanie podobnych projektów."""
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
    """Semantyczne wyszukiwanie podobnych komponentów."""
    key = canonicalize_name(name)
    emb = get_embedding_ollama(key)
    if not emb:
        return []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT name, avg_hours_total, confidence, occurrences,
            1 - (name_embedding <-> %s::vector) AS similarity
            FROM component_patterns
            WHERE department=%s AND name_embedding IS NOT NULL
            ORDER BY name_embedding <-> %s::vector
            LIMIT %s
        """, (to_pgvector(emb), department, to_pgvector(emb), limit))
        return cur.fetchall()
        # === MASTER PROMPT ===
MASTER_PROMPT = """Jesteś senior konstruktorem CAD z 20-letnim doświadczeniem w:
- Projektowaniu ram spawalniczych i konstrukcji stalowych
- Automatyce przemysłowej (PLC, robotyka, pozycjonery)
- Systemach wizyjnych i kontroli jakości
- Narzędziach CAD: CATIA V5, SolidWorks, AutoCAD

Odpowiadaj ZAWSZE w języku polskim.

METODYKA SZACOWANIA:
1. ANALIZA WYMAGAŃ (10-15% czasu):
   - Przegląd specyfikacji klienta
   - Analiza norm (ISO, EN, PN)
   - Spotkania techniczne

2. KONCEPCJA I MODELOWANIE (40-50% czasu):
   - Szkice wstępne (3D Layout)
   - Modelowanie 3D głównych podzespołów (3D Detail)
   - Analiza kinematyczna (jeśli ruchome części)
   - Dobór komponentów standardowych

3. OBLICZENIA I WERYFIKACJA (20-30% czasu):
   - Analiza MES/wytrzymałość
   - Sprawdzenie kolizji
   - Optymalizacja masy/kosztów

4. DOKUMENTACJA (15-20% czasu):
   - Rysunki wykonawcze (2D Documentation)
   - Specyfikacja materiałowa (BOM)
   - Instrukcje montażu
   - Dokumentacja techniczna
5. RYZYKA - każde MUSI mieć:
   - "risk": opis
   - "impact": niski/średni/wysoki
   - "mitigation": jak minimalizować

CZYNNIKI KOMPLIKUJĄCE (dodaj czas):
- Spawanie precyzyjne: +20%
- Części ruchome/kinematyka: +30%
- Automatyzacja/PLC: +25%
- Specjalne normy (np. ciśnieniowe): +15%
- Niestandardowe materiały: +10%
- Duże wymiary (>10m): +25%

TWOJA ODPOWIEDŹ MUSI ZAWIERAĆ:
- Lista zadań z czasem każdego (w godzinach)
- Breakdown per faza: Layout / Detail / 2D Documentation
- SUMA dla każdej fazy
- SUMA TOTAL
- Główne założenia
- Potencjalne ryzyka czasowe
"""

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
        sections.append(f"Breakdown z poprzednich projektów:")
        sections.append(f"- 3D Layout: {total_layout:.1f}h")
        sections.append(f"- 3D Detail: {total_detail:.1f}h")
        sections.append(f"- 2D Documentation: {total_2d:.1f}h")
        sections.append(f"\nPrzykładowe komponenty:")

        for comp in limited[:15]:
            if not comp.get('is_summary'):
                sections.append(f"- {comp['name']}: Layout {comp.get('hours_3d_layout',0):.1f}h + Detail {comp.get('hours_3d_detail',0):.1f}h + 2D {comp.get('hours_2d',0):.1f}h")
                
                # Informacja o pod-komponentach
                if comp.get('comment'):
                    sections.append(f"  Uwagi: {comp['comment'][:100]}")

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
            prop_l = p.get('proportion_layout', 0.33)
            prop_d = p.get('proportion_detail', 0.33)
            prop_doc = p.get('proportion_doc', 0.33)

            sections.append(f"{mark} '{p['name']}': Total {total_h:.1f}h (Layout {layout_h:.1f}h/{prop_l:.0%}, Detail {detail_h:.1f}h/{prop_d:.0%}, 2D {doc_h:.1f}h/{prop_doc:.0%}) - {p['occurrences']}x")

    sections.append("""

WYMAGANY FORMAT ODPOWIEDZI - ZWRÓĆ TYLKO CZYSTY JSON:
{
  "components": [
    {"name": "Nazwa", "layout_h": 12.5, "detail_h": 42.0, "doc_h": 28.0}
  ],
  "sums": {"layout": 12.5, "detail": 42.0, "doc": 28.0, "total": 82.5},
  "assumptions": ["Założenie 1"],
  "risks": [
    {
      "risk": "Opis ryzyka",
      "impact": "wysoki/średni/niski",
      "mitigation": "Jak zminimalizować"
    }
  ]
}

WAŻNE: Zwróć WYŁĄCZNIE JSON bez tekstu.""")

    return "\n".join(sections)

# === POMOCNICY BAZY DANYCH ===
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
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            cur.execute('ALTER TABLE projects ADD COLUMN IF NOT EXISTS department VARCHAR(10)')
            
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
                    UNIQUE(name, department)
                )
            ''')
            
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
                ALTER TABLE component_patterns
                ADD COLUMN IF NOT EXISTS proportion_layout FLOAT DEFAULT 0.33,
                ADD COLUMN IF NOT EXISTS proportion_detail FLOAT DEFAULT 0.33,
                ADD COLUMN IF NOT EXISTS proportion_doc FLOAT DEFAULT 0.33,
                ADD COLUMN IF NOT EXISTS std_dev_hours FLOAT DEFAULT 0,
                ADD COLUMN IF NOT EXISTS department VARCHAR(10)
            ''')
            
            cur.execute('CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_projects_department ON projects(department)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_patterns_department ON component_patterns(department)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_component_patterns_name ON component_patterns(name, department)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_versions_project ON project_versions(project_id, created_at DESC)')
            
            conn.commit()
            logger.info("Baza zainicjalizowana")
            return True
    except Exception as e:
        logger.error(f"Błąd inicjalizacji: {e}")
        st.error(f"Błąd inicjalizacji: {e}")
        return False

# === POMOCNICY AI ===
@lru_cache(maxsize=1)
def list_local_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.ok:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except:
        pass
    return []

def model_available(name_prefix: str) -> bool:
    return any(m.startswith(name_prefix) for m in list_local_models())

@st.cache_data(ttl=3600, show_spinner=False)
def query_ollama_cached(_payload_str: str) -> str:
    payload = json.loads(_payload_str)
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
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

def parse_ai_response(text: str, components_from_excel=None):
    """Parsing z priorytetem JSON, fallback na regex."""
    warnings = []
    parsed_components = []
    total_layout = total_detail = total_2d = 0.0
    data = {}  # DODANE - inicjalizacja

    if not text:
        warnings.append("Brak odpowiedzi od AI")
        return {
            "total_hours": 0.0, "total_layout": 0.0, "total_detail": 0.0, "total_2d": 0.0,
            "components": components_from_excel or [], "raw_text": "", "warnings": warnings,
            "analysis": {}, "missing_info": [], "phases": {},
            "risks_detailed": [], "recommendations": []
        }

    # JSON parsing
    try:
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]


        data = json.loads(clean_text)

        if "risks" in data:
            normalized_risks = []
            for risk in data["risks"]:
                if isinstance(risk, str):
                    normalized_risks.append({
                        "risk": risk,
                        "impact": "nieznany",
                        "mitigation": "Do określenia"
                    })
                elif isinstance(risk, dict):
                    normalized_risks.append({
                        "risk": risk.get("risk", "Nieznane ryzyko"),
                        "impact": risk.get("impact", "nieznany"),
                        "mitigation": risk.get("mitigation", "Brak")
                    })
            data["risks"] = normalized_risks





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

        logger.info("✅ JSON parsing success")

    except json.JSONDecodeError:
        logger.warning("JSON failed, fallback to regex")
        warnings.append("Fallback do regex")
        data = {}  # DODANE

        # Regex fallback
        pattern = r"-\s*([^\n]+?)\s+Layout:\s*(\d+[.,]?\d*)\s*h?,?\s*Detail:\s*(\d+[.,]?\d*)\s*h?,?\s*2D:\s*(\d+[.,]?\d*)\s*h?"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                parsed_components.append({
                    "name": match.group(1).strip(),
                    "hours_3d_layout": float(match.group(2).replace(',', '.')),
                    "hours_3d_detail": float(match.group(3).replace(',', '.')),
                    "hours_2d": float(match.group(4).replace(',', '.')),
                    "hours": sum([float(match.group(i).replace(',', '.')) for i in [2,3,4]])
                })
            except:
                pass

    if not parsed_components and components_from_excel:
        warnings.append("Użyto danych z Excel")
        parsed_components = [c for c in components_from_excel if not c.get('is_summary', False)]

    if total_layout == 0 and parsed_components:
        total_layout = sum(c.get('hours_3d_layout', 0) for c in parsed_components)
    if total_detail == 0 and parsed_components:
        total_detail = sum(c.get('hours_3d_detail', 0) for c in parsed_components)
    if total_2d == 0 and parsed_components:
        total_2d = sum(c.get('hours_2d', 0) for c in parsed_components)

    # ZMIENIONY RETURN - dodane nowe pola
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
        "recommendations": data.get("recommendations", [])
    }





    total_layout = total_detail = total_2d = 0.0
    
    if not text:
        warnings.append("Brak odpowiedzi od AI")
        return {
            "total_hours": 0.0, "total_layout": 0.0, "total_detail": 0.0, "total_2d": 0.0,
            "components": components_from_excel or [], "raw_text": "", "warnings": warnings
        }
    
    # JSON parsing
    try:
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        data = json.loads(clean_text)
        
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
        
        logger.info("✅ JSON parsing success")
        
    except json.JSONDecodeError:
        logger.warning("JSON failed, fallback to regex")
        warnings.append("Fallback do regex")
        
        # Regex fallback
        pattern = r"-\s*([^\n]+?)\s+Layout:\s*(\d+[.,]?\d*)\s*h?,?\s*Detail:\s*(\d+[.,]?\d*)\s*h?,?\s*2D:\s*(\d+[.,]?\d*)\s*h?"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                parsed_components.append({
                    "name": match.group(1).strip(),
                    "hours_3d_layout": float(match.group(2).replace(',', '.')),
                    "hours_3d_detail": float(match.group(3).replace(',', '.')),
                    "hours_2d": float(match.group(4).replace(',', '.')),
                    "hours": sum([float(match.group(i).replace(',', '.')) for i in [2,3,4]])
                })
            except:
                pass
    
    if not parsed_components and components_from_excel:
        warnings.append("Użyto danych z Excel")
        parsed_components = [c for c in components_from_excel if not c.get('is_summary', False)]
    
    if total_layout == 0 and parsed_components:
        total_layout = sum(c.get('hours_3d_layout', 0) for c in parsed_components)
    if total_detail == 0 and parsed_components:
        total_detail = sum(c.get('hours_3d_detail', 0) for c in parsed_components)
    if total_2d == 0 and parsed_components:
        total_2d = sum(c.get('hours_2d', 0) for c in parsed_components)
    
    return {
        "total_hours": max(0.0, total_layout + total_detail + total_2d),
        "total_layout": total_layout, "total_detail": total_detail, "total_2d": total_2d,
        "components": parsed_components, "raw_text": text, "warnings": warnings
    }

def parse_cad_project_structured(file_stream):
    """Parser Excel z hierarchią i komentarzami."""
    result = {'project_info': {}, 'multipliers': {}, 'components': [], 'totals': {}, 'statistics': {}}
    df = pd.read_excel(file_stream, header=None)
    
    try:
        result['project_info']['name'] = str(df.iloc[0, 1]) if pd.notna(df.iloc[0, 1]) else ""
        result['project_info']['customer'] = str(df.iloc[2, 1]) if pd.notna(df.iloc[2, 1]) else ""
        result['project_info']['cad'] = str(df.iloc[5, 1]) if pd.notna(df.iloc[5, 1]) else ""
    except:
        pass
    
    multipliers_row = 9
    try:
        result['multipliers']['layout'] = float(df.iloc[multipliers_row, 7]) if pd.notna(df.iloc[multipliers_row, 7]) else 1.0
        result['multipliers']['detail'] = float(df.iloc[multipliers_row, 9]) if pd.notna(df.iloc[multipliers_row, 9]) else 1.0
        result['multipliers']['documentation'] = float(df.iloc[multipliers_row, 11]) if pd.notna(df.iloc[multipliers_row, 11]) else 1.0
    except:
        result['multipliers'] = {'layout': 1.0, 'detail': 1.0, 'documentation': 1.0}
    
    data_start_row = 11
    COL_POS, COL_DESC, COL_COMMENT = 0, 1, 2
    COL_STD_PARTS, COL_SPEC_PARTS = 3, 4
    COL_HOURS_LAYOUT, COL_HOURS_DETAIL, COL_HOURS_DOC = 7, 9, 11
    
    missing_pos_counter = 1
    
    for row_idx in range(data_start_row, df.shape[0]):
        try:
            row_data = df.iloc[row_idx, [COL_POS, COL_DESC, COL_HOURS_LAYOUT, COL_HOURS_DETAIL, COL_HOURS_DOC]]
            if all(pd.isna(row_data)):
                continue
            
            pos = str(df.iloc[row_idx, COL_POS]).strip() if pd.notna(df.iloc[row_idx, COL_POS]) else ""
            name = str(df.iloc[row_idx, COL_DESC]).strip() if pd.notna(df.iloc[row_idx, COL_DESC]) else ""
            
            if not pos or pos in ['nan', 'None', '']:
                has_hours = any([pd.notna(df.iloc[row_idx, col]) for col in [COL_HOURS_LAYOUT, COL_HOURS_DETAIL, COL_HOURS_DOC]])
                if has_hours:
                    pos = f"X.{missing_pos_counter}"
                    missing_pos_counter += 1
                    if not name or name in ['nan', 'None', '']:
                        name = f"[Komponent wiersz {row_idx + 1}]"
                else:
                    continue
            
        
            if name in ['nan', 'None']:
                name = f"[Pozycja {pos}]"
            
            comment = str(df.iloc[row_idx, COL_COMMENT]).strip() if pd.notna(df.iloc[row_idx, COL_COMMENT]) else ""
            if comment in ['nan', 'None']:
                comment = ''
            
            # Parsuj pod-komponenty z komentarza
            subcomponents = parse_subcomponents_from_comment(comment)
            
            std_parts = spec_parts = 0
            try:
                if pd.notna(df.iloc[row_idx, COL_STD_PARTS]):
                    std_parts = int(float(df.iloc[row_idx, COL_STD_PARTS]))
            except:
                pass
            try:
                if pd.notna(df.iloc[row_idx, COL_SPEC_PARTS]):
                    spec_parts = int(float(df.iloc[row_idx, COL_SPEC_PARTS]))
            except:
                pass
            
            hours_layout = hours_detail = hours_doc = 0.0
            try:
                if pd.notna(df.iloc[row_idx, COL_HOURS_LAYOUT]):
                    hours_layout = float(df.iloc[row_idx, COL_HOURS_LAYOUT])
            except:
                pass
            try:
                if pd.notna(df.iloc[row_idx, COL_HOURS_DETAIL]):
                    hours_detail = float(df.iloc[row_idx, COL_HOURS_DETAIL])
            except:
                pass
            try:
                if pd.notna(df.iloc[row_idx, COL_HOURS_DOC]):
                    hours_doc = float(df.iloc[row_idx, COL_HOURS_DOC])
            except:
                pass
            
            total_hours = hours_layout + hours_detail + hours_doc
            # Suma TYLKO jeśli to główny poziom (1.0, 2.0) I ma pod-elementy
           # Suma TYLKO dla głównych numerów bez kropki (1, 2, 3) lub pustych wierszy
            is_summary = (
                pos.replace('.', '').isdigit() and '.' not in pos  # Tylko 1, 2, 3 (bez kropki)
            ) or (
                hours_layout == 0 and hours_detail == 0 and hours_doc == 0  # Puste wiersze
)
            is_summary = pos.endswith('.0') or (pos.replace('.', '').isdigit() and '.' not in pos[1:])
            
            component = {
                'id': pos, 'name': name, 'comment': comment,
                'type': 'assembly' if is_summary else 'part',
                'level': pos.count('.'),
                'parts': {'standard': std_parts, 'special': spec_parts, 'total': std_parts + spec_parts},
                'hours_3d_layout': hours_layout, 'hours_3d_detail': hours_detail, 
                'hours_2d': hours_doc, 'hours': total_hours, 'is_summary': is_summary
            }
            
            result['components'].append(component)
            
        except Exception as e:
            logger.warning(f"Błąd wiersz {row_idx + 1}: {e}")
            continue
    
    parts_only = [c for c in result['components'] if not c.get('is_summary', False)]
    result['totals']['layout'] = sum(c['hours_3d_layout'] for c in parts_only)
    result['totals']['detail'] = sum(c['hours_3d_detail'] for c in parts_only)
    result['totals']['documentation'] = sum(c['hours_2d'] for c in parts_only)
    result['totals']['total'] = sum(c['hours'] for c in parts_only)
    result['statistics']['total_standard_parts'] = sum(c['parts']['standard'] for c in parts_only)
    result['statistics']['total_special_parts'] = sum(c['parts']['special'] for c in parts_only)
    result['statistics']['assemblies_count'] = sum(1 for c in result['components'] if c.get('is_summary', False))
    result['statistics']['parts_count'] = len(parts_only)
    
    logger.info(f"Parser: {len(result['components'])} komponentów ({len(parts_only)} części)")
    return result

def process_excel(file):
    try:
        result = parse_cad_project_structured(file)
        if result['components']:
            st.success(f"✅ {len(result['components'])} komponentów: "
                      f"Layout {result['totals']['layout']:.1f}h + "
                      f"Detail {result['totals']['detail']:.1f}h + "
                      f"2D {result['totals']['documentation']:.1f}h = "
                      f"{result['totals']['total']:.1f}h")
            if result['multipliers']:
                st.info(f"Współczynniki: Layout={result['multipliers']['layout']}, "
                       f"Detail={result['multipliers']['detail']}, "
                       f"Doc={result['multipliers']['documentation']}")
                st.session_state["excel_multipliers"] = result['multipliers']
        return result['components']
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

    #累積 godziny → timeline
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

    fig = px.bar(
        df,
        x='Hours',
        y='Task',
        orientation='h',
        title="Harmonogram realizacji (sekwencyjnie)",
        labels={'Hours': 'Godziny', 'Task': 'Komponent'}
    )
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
    terms = ' & '.join(description.split()[:10])
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, name, client, estimated_hours, actual_hours, department
            FROM projects WHERE department = %s
            AND to_tsvector('simple', coalesce(description,'')) @@ to_tsquery('simple', %s)
            ORDER BY created_at DESC LIMIT %s
        """, (department, terms, limit))
        return cur.fetchall()

def validate_project_input(name, estimated_hours):
    errors = []
    if not name or not name.strip():
        errors.append("Nazwa nie może być pusta")
    if len(name) > 255:
        errors.append("Nazwa zbyt długa")
    if estimated_hours < 0:
        errors.append("Godziny < 0")
    if estimated_hours > 10000:
        errors.append("Godziny > 10000")
    return errors

def clear_project_session():
    for k in ['ai_analysis', 'project_name', 'client', 'description', 'department']:
        if k in st.session_state:
            del st.session_state[k]

def batch_import_excels(files, department):
    results = []
    progress_bar = st.progress(0)
    total = len(files)
    
    for i, file in enumerate(files):
        try:
            progress_bar.progress((i + 1) / total, text=f"Przetwarzanie {file.name} ({i+1}/{total})...")
            components = process_excel(file)
            
            if not components:
                results.append({'file': file.name, 'status': 'error', 'message': 'Brak komponentów'})
                continue
            
            project_name = file.name.replace('.xlsx', '').replace('.xls', '')
            parts_only = [c for c in components if not c.get('is_summary', False)]
            
            if not parts_only:
                results.append({'file': file.name, 'status': 'error', 'message': 'Brak części'})
                continue
            
            total_layout = sum(c.get('hours_3d_layout', 0) for c in parts_only)
            total_detail = sum(c.get('hours_3d_detail', 0) for c in parts_only)
            total_2d = sum(c.get('hours_2d', 0) for c in parts_only)
            total_hours = total_layout + total_detail + total_2d
            
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO projects (name, department, components,
                    estimated_hours_3d_layout, estimated_hours_3d_detail,
                    estimated_hours_2d, estimated_hours)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                """, (project_name, department, json.dumps(parts_only, ensure_ascii=False),
                      total_layout, total_detail, total_2d, total_hours))
                project_id = cur.fetchone()[0]
                
                                
                save_project_version(conn, project_id, "v1.0", parts_only,
                                    total_hours, total_layout, total_detail, total_2d,
                                    "Import historyczny", "Batch Import")
                conn.commit()
            
            results.append({
                'file': file.name, 'status': 'success',
                'project_id': project_id, 'hours': total_hours
            })
            
        except Exception as e:
            results.append({'file': file.name, 'status': 'error', 'message': str(e)})
            logger.error(f"Błąd {file.name}: {e}")
    
    progress_bar.empty()
    return results
def main():
    st.title("🚀 CAD Estimator Pro")
    
    if not init_db():
        st.stop()
    
    # === SIDEBAR ===
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
    
    # === DASHBOARD ===
    if page == "Dashboard":
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
                        AND to_tsvector('simple', name || ' ' || client || ' ' || description) @@ to_tsquery('simple', %s)
                        ORDER BY created_at DESC LIMIT 10
                    """, (search_dept, ' & '.join(search_query.split())))
                else:
                    cur.execute("""
                        SELECT id, name, client, department, estimated_hours, description
                        FROM projects
                        WHERE to_tsvector('simple', name || ' ' || client || ' ' || description) @@ to_tsquery('simple', %s)
                        ORDER BY created_at DESC LIMIT 10
                    """, (' & '.join(search_query.split()),))
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
    
    # === NOWY PROJEKT ===
    elif page == "Nowy projekt":
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
            project_name = st.text_input("Nazwa projektu*", key="project_name")
            client = st.text_input("Klient", key="client")
            description = st.text_area("Opis", height=200, key="description")
        with col2:
            excel_file = st.file_uploader("Excel", type=['xlsx', 'xls'])
            image_files = st.file_uploader("Zdjęcia/Rysunki", type=['jpg', 'png'], accept_multiple_files=True)
            pdf_files = st.file_uploader("PDF", type=['pdf'], accept_multiple_files=True)
        
        if st.button("🤖 Analizuj z AI", use_container_width=True):
            if not description and not excel_file and not image_files and not pdf_files:
                st.warning("Podaj opis lub wgraj pliki")
            else:
                progress_bar = st.progress(0, text="Startuję...")
                try:
                    # Excel
                    components_from_excel = []
                    if excel_file:
                        progress_bar.progress(15, text="Wczytuję Excel...")
                        components_from_excel = process_excel(excel_file)
                    
                    # Obrazy
                    images_b64 = []
                    if image_files:
                        progress_bar.progress(25, text="Analizuję obrazy...")
                        for img in image_files:
                            images_b64.append(encode_image_b64(img))
                    
                    # PDF
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
                    
                    # Prompt
                    prompt = build_analysis_prompt(description, components_from_excel, learned_patterns, pdf_text, department)
                    
                    # Model
                    if images_b64 and model_available("llava"):
                        ai_model = "llava:13b" if model_available("llava:13b") else "llava:latest"
                    else:
                        ai_model = selected_model
                    
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
                    progress_bar.progress(100, text="Gotowe ✅")
                    time.sleep(1)
                    progress_bar.empty()
                    
                except Exception as e:
                    logger.exception("Analiza failed")
                    st.error(f"Błąd: {e}")
        
        if "ai_analysis" in st.session_state:
            analysis = st.session_state["ai_analysis"]
            st.subheader("Wynik analizy")
            
            # with st.expander("Odpowiedź AI", expanded=False):
            #  st.markdown(analysis["raw_text"])
            # === NOWE EXPANDERY ===
            if analysis.get("analysis"):
                with st.expander("📊 Analiza projektu", expanded=True):
                    anal = analysis["analysis"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Typ", anal.get("project_type", "N/A"))
                    col2.metric("Złożoność", anal.get("complexity", "N/A"))
                    if anal.get("estimated_accuracy"):
                        col3.metric("Dokładność", anal["estimated_accuracy"])

            if analysis.get("missing_info"):
                st.warning("⚠️ AI wykryło braki w opisie - uzupełnij dla lepszej wyceny:")
                for info in analysis["missing_info"]:
                    st.write(f"❓ {info}")

                if st.button("📝 Edytuj opis i analizuj ponownie"):
                    st.session_state["reanalyze_mode"] = True

            if st.session_state.get("reanalyze_mode"):
                new_description = st.text_area(
                    "Uzupełnij opis projektu",
                    value=st.session_state.get("description", ""),
                    height=200,
                    key="new_description"
                )
                if st.button("🔄 Analizuj z nowym opisem", type="primary"):
                    st.session_state["description"] = new_description
                    st.session_state["reanalyze_mode"] = False
                    st.rerun()

            if analysis.get("phases"):
                with st.expander("🔧 Szczegóły faz projektu"):
                    for phase_name, phase_data in analysis["phases"].items():
                        st.markdown(f"**{phase_name.upper()} - {phase_data.get('hours', 0):.1f}h**")
                        tasks = phase_data.get("tasks", [])
                        if tasks:
                            for task in tasks:
                                st.write(f"  • {task}")
                        st.divider()
            if analysis.get("risks_detailed"):
                with st.expander("⚠️ Ryzyka i mitygacje"):
                    for risk in analysis["risks_detailed"]:
                        if isinstance(risk, str):
                            # Fallback - nie powinno się zdarzyć po normalizacji
                            st.write(f"⚠️ {risk}")
                            logger.warning(f"Risk jest stringiem: {risk}")
                        else:
                            impact = risk.get("impact", "nieznany")
                            icon = {"niski": "🟢", "średni": "🟡", "wysoki": "🔴"}.get(impact, "⚪")
                            st.markdown(f"{icon} **{risk.get('risk', 'Ryzyko')}** (wpływ: {impact})")
                            st.write(f"  → Mitygacja: {risk.get('mitigation', 'Brak')}")
                        st.divider()



            if analysis.get("recommendations"):
                with st.expander("💡 Rekomendacje"):
                    for rec in analysis["recommendations"]:
                        st.write(f"✓ {rec}")


            with st.expander("🤖 Odpowiedź AI (raw)", expanded=False):
                try:
                    parsed_json = json.loads(analysis["raw_text"])

                    st.markdown("### 📋 Komponenty")
                    for comp in parsed_json.get("components", []):
                        st.write(f"**{comp.get('name')}**: Layout {comp.get('layout_h', 0):.1f}h + Detail {comp.get('detail_h', 0):.1f}h + 2D {comp.get('doc_h', 0):.1f}h")

                    st.markdown("### 📊 Podsumowanie")
                    sums = parsed_json.get("sums", {})
                    st.write(f"- Layout: {sums.get('layout', 0):.1f}h")
                    st.write(f"- Detail: {sums.get('detail', 0):.1f}h")
                    st.write(f"- 2D: {sums.get('doc', 0):.1f}h")
                    st.write(f"- **TOTAL: {sums.get('total', 0):.1f}h**")

                    st.markdown("### 📝 Założenia")
                    for ass in parsed_json.get("assumptions", []):
                        st.write(f"- {ass}")

                    st.markdown("### ⚠️ Ryzyka")
                    for risk in parsed_json.get("risks", []):
                        if isinstance(risk, dict):
                            st.write(f"- {risk.get('risk', risk)}")
                        else:
                            st.write(f"- {risk}")

                except:
                    # Fallback jeśli JSON niepoprawny
                    st.text(analysis["raw_text"])
            for w in analysis.get("warnings", []):
                st.warning(w)
            
            estimated_hours = analysis.get("total_hours", 0.0)
            layout_h = analysis.get("total_layout", 0.0)
            detail_h = analysis.get("total_detail", 0.0)
            doc_h = analysis.get("total_2d", 0.0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Layout", f"{layout_h:.1f}h")
            col2.metric("Detail", f"{detail_h:.1f}h")
            col3.metric("2D", f"{doc_h:.1f}h")
            col4.metric("TOTAL", f"{max(0.0, estimated_hours):.1f}h", delta=f"{(estimated_hours * 150):.0f} PLN")
            
            st.subheader("🔧 Edytuj wycenę")
            final_estimated_hours = estimated_hours
            final_components = analysis.get('components', [])
            
            # Edytor komponentów
            # Edytor komponentów
            if final_components:
                parts_only = [c for c in final_components if not c.get('is_summary', False)]

                st.subheader("📝 Komponenty")
                st.caption(f"ℹ️ Pokazano {len(parts_only)} komponentów (z {len(final_components)} z Excela, pominięto sumy)")

                for i, comp in enumerate(parts_only):


                    # Skróć nazwę w tytule expandera
                    display_name = comp['name'][:50] + "..." if len(comp['name']) > 50 else comp['name']

                    with st.expander(f"{display_name} - {comp.get('hours', 0):.1f}h"):
                        # Pełna nazwa wewnątrz
                        st.markdown(f"**Pełna nazwa:** {comp['name']}")

                        col1, col2, col3 = st.columns(3)
                        new_layout = col1.number_input("Layout", value=float(comp.get('hours_3d_layout', 0)), key=f"l_{i}")
                        new_detail = col2.number_input("Detail", value=float(comp.get('hours_3d_detail', 0)), key=f"d_{i}")
                        new_doc = col3.number_input("2D", value=float(comp.get('hours_2d', 0)), key=f"doc_{i}")
                        
                        comp['hours_3d_layout'] = new_layout
                        comp['hours_3d_detail'] = new_detail
                        comp['hours_2d'] = new_doc
                        comp['hours'] = new_layout + new_detail + new_doc
                        
                        # Pod-komponenty
                        if comp.get('subcomponents'):
                            st.markdown("**Zawiera:**")
                            for sub in comp['subcomponents']:
                                qty = sub.get('quantity', 1)
                                if qty > 1:
                                    st.text(f"  • {qty}x {sub['name']}")
                                else:
                                    st.text(f"  • {sub['name']}")
                        
                        if comp.get('comment'):
                            st.caption(f"💬 {comp['comment']}")
                
                final_estimated_hours = sum(c['hours'] for c in parts_only)
                st.metric("🔢 Suma", f"{final_estimated_hours:.1f}h")
            
            # Współczynniki
            if "excel_multipliers" in st.session_state and final_components:
                st.subheader("📊 Współczynniki z Excela")
                mult = st.session_state["excel_multipliers"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Layout", f"x{mult['layout']:.2f}")
                col2.metric("Detail", f"x{mult['detail']:.2f}")
                col3.metric("Doc", f"x{mult['documentation']:.2f}")
                
                if st.checkbox("Zastosuj współczynniki", value=True, key="apply_mult"):
                    for c in final_components:
                        if not c.get('is_summary'):
                            c['hours_3d_layout'] = c.get('hours_3d_layout', 0) * mult['layout']
                            c['hours_3d_detail'] = c.get('hours_3d_detail', 0) * mult['detail']
                            c['hours_2d'] = c.get('hours_2d', 0) * mult['documentation']
                            c['hours'] = c['hours_3d_layout'] + c['hours_3d_detail'] + c['hours_2d']
                    
                    final_estimated_hours = sum(c.get('hours', 0) for c in final_components if not c.get('is_summary'))
                    st.success(f"✅ Po korekcji: {final_estimated_hours:.1f}h")
            
            # Timeline
            st.subheader("🗂️ Harmonogram")
            show_project_timeline(final_components)
            
            # Podobne projekty
            with get_db_connection() as conn:
                similar = find_similar_projects(conn, st.session_state.get("description"), department)
            
            st.subheader(f"📊 Podobne projekty ({department})")
            if similar:
                for proj in similar:
                    col1, col2, col3 = st.columns([3,1,1])
                    col1.write(f"**{proj['name']}** ({proj['client'] or '-'})")
                    col2.metric("Szacowano", f"{(proj['estimated_hours'] or 0):.1f}h")
                    if proj['actual_hours']:
                        col3.metric("Rzeczywiście", f"{proj['actual_hours']:.1f}h")
            else:
                st.info("Brak podobnych")
            # Semantyczne podobne projekty
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
            
            # Eksport
            st.subheader("📤 Eksport")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Excel", use_container_width=True):
                    excel_data = export_quotation_to_excel({
                        'name': st.session_state.get("project_name"),
                        'client': st.session_state.get("client"),
                        'department': department,
                        'components': final_components,
                        'total_hours': final_estimated_hours
                    })
                    st.download_button("⬇️ Pobierz", excel_data,
                                      file_name=f"wycena_{st.session_state.get('project_name','p')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            # Zapis
            st.subheader("💾 Zapisz projekt")
            col1, col2 = st.columns([3,1])
            with col1:
                change_desc = st.text_input("Opis zmian", placeholder="np. 'Pierwsza wycena'")
            with col2:
                is_approved = st.checkbox("Zatwierdzone", value=False)
            
            if st.button("💾 Zapisz", type="primary", use_container_width=True):
                errors = validate_project_input(st.session_state.get("project_name"), final_estimated_hours)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    try:
                        components_to_save = [c for c in final_components if not c.get('is_summary', False)]
                        
                        with get_db_connection() as conn, conn.cursor() as cur:
                            cur.execute("SELECT id, components FROM projects WHERE name = %s AND department = %s",
                                      (st.session_state.get("project_name"), department))
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
                                      float(analysis.get("total_layout", 0)), float(analysis.get("total_detail", 0)),
                                      float(analysis.get("total_2d", 0)), float(final_estimated_hours),
                                      analysis["raw_text"], project_id))
                                
                                save_project_version(conn, project_id, version_num, components_to_save,
                                                    final_estimated_hours, analysis.get("total_layout", 0),
                                                    analysis.get("total_detail", 0), analysis.get("total_2d", 0),
                                                    change_desc, "System")
                                
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
                                    float(analysis.get("total_layout", 0)), float(analysis.get("total_detail", 0)),
                                    float(analysis.get("total_2d", 0)), float(final_estimated_hours),
                                    analysis["raw_text"]
                                ))
                                project_id = cur.fetchone()[0]
                                
                                # NOWE: Dodaj embedding
                                ensure_project_embedding(cur, project_id, st.session_state.get("description", ""))
                                
                                save_project_version(conn, project_id, "v1.0", components_to_save,
                                                    final_estimated_hours, analysis.get("total_layout", 0),
                                                    analysis.get("total_detail", 0), analysis.get("total_2d", 0),
                                                    change_desc or "Pierwsza wycena", "System")
                                
                                if is_approved:
                                    cur.execute("UPDATE project_versions SET is_approved = TRUE WHERE project_id = %s AND version = 'v1.0'",
                                              (project_id,))
                                
                                conn.commit()
                                st.success(f"✅ Zapisano! ID: {project_id}")
                            
                            logger.info(f"Zapisano: {project_id} - {st.session_state.get('project_name')}")
                            clear_project_session()
                            st.balloons()
                            time.sleep(1.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                        logger.exception("Zapis failed")
    
    # === HISTORIA I UCZENIE ===
    elif page == "Historia i Uczenie":
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
                                        # Ucz się głównego komponentu
                                        update_pattern_smart(
                                            cur, comp.get('name', 'nieznany'), proj['department'],
                                            layout_est * ratio, detail_est * ratio,
                                            doc_est * ratio, total_est * ratio, source='actual'
                                        )
                                        
                                        # Ucz się też pod-komponentów
                                        subcomponents = comp.get('subcomponents', [])
                                        if subcomponents:
                                            total_qty = sum(s.get('quantity', 1) for s in subcomponents)
                                            
                                            for sub in subcomponents:
                                                qty = sub.get('quantity', 1)
                                                weight = qty / total_qty if total_qty > 0 else 1.0 / len(subcomponents)
                                                
                                                # Proporcjonalny czas dla pod-komponentu
                                                sub_layout = layout_est * ratio * weight
                                                sub_detail = detail_est * ratio * weight
                                                sub_doc = doc_est * ratio * weight
                                                sub_total = total_est * ratio * weight
                                                
                                                update_pattern_smart(
                                                    cur, 
                                                    sub['name'], 
                                                    proj['department'],
                                                    sub_layout, sub_detail, sub_doc, sub_total,
                                                    source='subcomponent'
                                                )
                                                
                                                logger.info(f"  └─ Sub: {qty}x {sub['name']} → {sub_total:.1f}h")
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
            # Admin: backfill embeddingów
# Admin: backfill embeddingów
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
                                progress.progress((idx + 1) / total_items, text=f"Projekty: {idx+1}/{len(projects_to_embed)}")
                            
                            for idx, r in enumerate(patterns_to_embed):
                                ensure_pattern_embedding(cur, r['pattern_key'], r['department'], r['name'])
                                progress.progress((len(projects_to_embed) + idx + 1) / total_items, text=f"Wzorce: {idx+1}/{len(patterns_to_embed)}")
                            
                            conn.commit()
                            progress.empty()
                            st.success(f"✅ Przeliczono {len(projects_to_embed)} projektów + {len(patterns_to_embed)} wzorców")
        
        with tab3:
            st.subheader("📦 Batch Import")
            st.info("Import wielu plików Excel naraz")
            
            batch_dept = st.selectbox("Dział dla importu", options=list(DEPARTMENTS.keys()),
                                     format_func=lambda x: f"{x} - {DEPARTMENTS[x]}", key="batch_dept")
            
            excel_files = st.file_uploader("Excel (wiele)", type=['xlsx', 'xls'],
                                          accept_multiple_files=True, key="batch")
            
            if excel_files:
                st.write(f"📁 {len(excel_files)} plików")
                for f in excel_files[:10]:
                    st.write(f"• {f.name}")
                if len(excel_files) > 10:
                    st.write(f"... +{len(excel_files) - 10}")
                
                if st.button("🚀 Importuj", type="primary", use_container_width=True):
                    st.info(f"Import {len(excel_files)} do {batch_dept}...")
                    results = batch_import_excels(excel_files, batch_dept)
                    
                    success = sum(1 for r in results if r['status'] == 'success')
                    errors = sum(1 for r in results if r['status'] == 'error')
                    
                    col1, col2 = st.columns(2)
                    col1.metric("✅ Sukces", success)
                    col2.metric("❌ Błędy", errors)
                    
                    st.subheader("Szczegóły")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    if success > 0:
                        st.success(f"🎉 {success} projektów!")
                    if errors > 0:
                        st.warning(f"⚠️ {errors} błędów")

if __name__ == "__main__":
    main()
