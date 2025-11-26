"""
CAD Estimator Pro - Component Parser

Implementation of ComponentParser protocol for canonicalization and parsing.
"""
import logging
import re
from ...domain.exceptions import ParsingError

logger = logging.getLogger(__name__)

# Component aliases (PL/DE/EN -> EN)
COMPONENT_ALIASES = {
    'wspornik': 'bracket', 'wsporniki': 'bracket', 'halterung': 'bracket',
    'rama': 'frame', 'ramy': 'frame', 'rahmen': 'frame',
    'przenośnik': 'conveyor', 'przenośniki': 'conveyor',
    'płyta': 'plate', 'płyty': 'plate', 'platte': 'plate',
    'profil': 'profile', 'profile': 'profile',
    'adapter': 'adapter', 'adapters': 'adapter',
    'czujnik': 'sensor', 'czujniki': 'sensor', 'sensor': 'sensor',
    'prowadnica': 'guide', 'prowadnice': 'guide', 'führung': 'guide',
    'łożysko': 'bearing', 'łożyska': 'bearing', 'lager': 'bearing',
    'śruba': 'screw', 'śruby': 'screw', 'schraube': 'screw',
    'cylinder': 'cylinder', 'cylinders': 'cylinder', 'siłownik': 'cylinder',
    'podstawa': 'base', 'podstawy': 'base', 'basis': 'base'
}


class CADComponentParser:
    """Component parser for canonicalization and sub-component extraction."""

    def canonicalize_component_name(self, name: str) -> str:
        """
        Canonicalize component name for matching/learning.

        Removes dimensions, numbers, stopwords, applies aliases.
        """
        if not name:
            return ""

        n = name.lower()

        # Remove dimensions and numbers
        n = re.sub(r'\b\d+[.,]?\d*\s*(mm|cm|m|kg|t|ton|szt|pcs|inch|")\b', ' ', n)
        n = re.sub(r'\b\d+[.,]?\d*\b', ' ', n)

        # Tokenize and map aliases
        tokens = re.split(r'[\s\-_.,;/]+', n)
        norm_tokens = []
        stoplist = {'i', 'a', 'the', 'and', 'or', 'der', 'die', 'das', 'und', 'of', 'for'}

        for tok in tokens:
            if not tok or tok in stoplist:
                continue
            mapped = COMPONENT_ALIASES.get(tok, tok)
            norm_tokens.append(mapped)

        # Deduplicate
        seen, out = set(), []
        for t in norm_tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)

        return ' '.join(out).strip()

    def parse_subcomponents_from_comment(self, comment: str) -> list[dict]:
        """Parse sub-components from Excel comment string."""
        if not comment or not isinstance(comment, str):
            return []

        subcomponents = []
        qty_re = re.compile(r'(\d+)\s*(?:x|szt\.?|sztuk|pcs)?\s*[-–—]?\s*([^,;\n]+?)(?=[,;\n]|$)', re.IGNORECASE)

        for m in qty_re.finditer(comment):
            try:
                qty = int(m.group(1))
                name = m.group(2).strip()
                if len(name) >= 3 and not re.match(r'^\d+\s*(mm|cm|m|kg|ton|h)$', name, re.IGNORECASE):
                    subcomponents.append({'name': name, 'quantity': qty})
            except Exception:
                continue

        logger.debug(f"Parsed {len(subcomponents)} subcomponents from comment")
        return subcomponents

    def parse_ai_response(self, ai_text: str, fallback_components=None) -> dict:
        """Parse AI response (stub - full implementation would be complex)."""
        # This would be a large implementation - for now return basic structure
        return {
            'components': fallback_components or [],
            'risks': [],
            'suggestions': [],
            'assumptions': [],
            'warnings': [],
            'overall_confidence': 0.5
        }
