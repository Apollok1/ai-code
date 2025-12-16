"""
CAD Estimator Pro - Component Parser

Implementation of ComponentParser protocol for canonicalization and parsing.
"""
import logging
import re
from cad.domain.exceptions import ParsingError

logger = logging.getLogger(__name__)

# Component aliases (PL/DE/EN -> EN)
# Expanded dictionary with 80+ entries covering common CAD/mechanical components
COMPONENT_ALIASES = {
    # Wsporniki / Brackets
    'wspornik': 'bracket', 'wsporniki': 'bracket', 'wspornika': 'bracket',
    'uchwyt': 'bracket', 'uchwyty': 'bracket', 'uchwytu': 'bracket',
    'konsola': 'bracket', 'konsole': 'bracket', 'konsoli': 'bracket',
    'halterung': 'bracket', 'halter': 'bracket',

    # Ramy / Frames
    'rama': 'frame', 'ramy': 'frame', 'ramie': 'frame', 'ramę': 'frame',
    'konstrukcja': 'frame', 'konstrukcje': 'frame', 'konstrukcji': 'frame',
    'szkielet': 'frame', 'szkielety': 'frame',
    'rahmen': 'frame', 'gestell': 'frame',
    'kadłub': 'frame', 'kadłuby': 'frame',

    # Płyty / Plates
    'płyta': 'plate', 'płyty': 'plate', 'płytę': 'plate', 'płycie': 'plate',
    'blacha': 'plate', 'blachy': 'plate', 'blachę': 'plate',
    'panel': 'plate', 'panele': 'plate', 'panelu': 'plate',
    'platte': 'plate', 'blech': 'plate',
    'tafla': 'plate', 'tafle': 'plate',

    # Osłony / Covers
    'osłona': 'cover', 'osłony': 'cover', 'osłonę': 'cover',
    'pokrywa': 'cover', 'pokrywy': 'cover', 'pokrywę': 'cover',
    'obudowa': 'cover', 'obudowy': 'cover', 'obudowę': 'cover',
    'maska': 'cover', 'maski': 'cover', 'maskę': 'cover',
    'kaptur': 'cover', 'kaptury': 'cover',
    'abdeckung': 'cover', 'gehäuse': 'cover',

    # Napędy / Drives & Motors
    'silnik': 'motor', 'silniki': 'motor', 'silnika': 'motor',
    'napęd': 'drive', 'napędy': 'drive', 'napędu': 'drive',
    'motor': 'motor', 'motoren': 'motor',
    'antrieb': 'drive',

    # Siłowniki / Actuators & Cylinders
    'siłownik': 'actuator', 'siłowniki': 'actuator', 'siłownika': 'actuator',
    'cylinder': 'cylinder', 'cylinders': 'cylinder', 'cylindry': 'cylinder',
    'tłok': 'piston', 'tłoki': 'piston', 'tłoka': 'piston',
    'zylinder': 'cylinder',

    # Prowadnice / Guides & Rails
    'prowadnica': 'guide', 'prowadnice': 'guide', 'prowadnicy': 'guide',
    'szyna': 'rail', 'szyny': 'rail', 'szynę': 'rail',
    'tor': 'rail', 'tory': 'rail', 'toru': 'rail',
    'führung': 'guide', 'schiene': 'rail',

    # Łożyska / Bearings
    'łożysko': 'bearing', 'łożyska': 'bearing', 'łożysku': 'bearing',
    'tuleja': 'bushing', 'tuleje': 'bushing', 'tuleję': 'bushing',
    'panewka': 'bearing', 'panewki': 'bearing',
    'lager': 'bearing', 'buchse': 'bushing',

    # Śruby i łączniki / Fasteners
    'śruba': 'screw', 'śruby': 'screw', 'śrubę': 'screw',
    'wkręt': 'screw', 'wkręty': 'screw', 'wkrętu': 'screw',
    'bolt': 'bolt', 'bolty': 'bolt', 'boltu': 'bolt',
    'nakrętka': 'nut', 'nakrętki': 'nut', 'nakrętkę': 'nut',
    'nit': 'rivet', 'nity': 'rivet', 'nitu': 'rivet',
    'schraube': 'screw', 'mutter': 'nut', 'niet': 'rivet',

    # Wały / Shafts & Axes
    'wał': 'shaft', 'wały': 'shaft', 'wału': 'shaft',
    'oś': 'axis', 'osie': 'axis', 'osi': 'axis',
    'trzpień': 'shaft', 'trzpienie': 'shaft', 'trzpienia': 'shaft',
    'welle': 'shaft', 'achse': 'axis', 'dorn': 'shaft',

    # Koła / Wheels & Rollers
    'koło': 'wheel', 'koła': 'wheel', 'kołu': 'wheel',
    'rolka': 'roller', 'rolki': 'roller', 'rolkę': 'roller',
    'krążek': 'pulley', 'krążki': 'pulley', 'krążka': 'pulley',
    'rad': 'wheel', 'rolle': 'roller', 'scheibe': 'pulley',
    'bęben': 'drum', 'bębny': 'drum', 'bębna': 'drum',

    # Przekładnie / Gears & Gearboxes
    'przekładnia': 'gearbox', 'przekładnie': 'gearbox', 'przekładni': 'gearbox',
    'reduktor': 'reducer', 'reduktory': 'reducer', 'reduktora': 'reducer',
    'koło zębate': 'gear', 'zębatka': 'rack',
    'getriebe': 'gearbox', 'zahnrad': 'gear',

    # Przenośniki / Conveyors
    'przenośnik': 'conveyor', 'przenośniki': 'conveyor', 'przenośnika': 'conveyor',
    'taśmociąg': 'belt_conveyor', 'taśmociągi': 'belt_conveyor',
    'transporter': 'conveyor', 'transportery': 'conveyor',
    'förderer': 'conveyor', 'förderband': 'belt_conveyor',

    # Czujniki / Sensors
    'czujnik': 'sensor', 'czujniki': 'sensor', 'czujnika': 'sensor',
    'sensor': 'sensor', 'sensory': 'sensor', 'sensora': 'sensor',
    'detektor': 'sensor', 'detektory': 'sensor', 'detektora': 'sensor',
    'fühler': 'sensor',

    # Sprężyny / Springs
    'sprężyna': 'spring', 'sprężyny': 'spring', 'sprężynę': 'spring',
    'resor': 'spring', 'resory': 'spring', 'resoru': 'spring',
    'feder': 'spring',

    # Zawory / Valves
    'zawór': 'valve', 'zawory': 'valve', 'zaworu': 'valve',
    'zasuwa': 'valve', 'zasuwy': 'valve', 'zasuwę': 'valve',
    'przepustnica': 'valve', 'przepustnice': 'valve',
    'ventil': 'valve', 'schieber': 'valve',

    # Filtry / Filters
    'filtr': 'filter', 'filtry': 'filter', 'filtru': 'filter',
    'wkład': 'cartridge', 'wkłady': 'cartridge', 'wkładu': 'cartridge',
    'filter': 'filter', 'einsatz': 'cartridge',

    # Złącza / Connectors & Adapters
    'złącze': 'connector', 'złącza': 'connector', 'złączu': 'connector',
    'adapter': 'adapter', 'adaptery': 'adapter', 'adaptera': 'adapter',
    'przejście': 'adapter', 'przejścia': 'adapter', 'przejściu': 'adapter',
    'konektor': 'connector', 'konektory': 'connector',
    'verbinder': 'connector',

    # Podstawy / Bases
    'podstawa': 'base', 'podstawy': 'base', 'podstawę': 'base', 'podstawie': 'base',
    'fundament': 'base', 'fundamenty': 'base', 'fundamentu': 'base',
    'stopa': 'foot', 'stopy': 'foot', 'stopę': 'foot',
    'basis': 'base', 'fuss': 'foot', 'sockel': 'base',

    # Profile / Profiles
    'profil': 'profile', 'profile': 'profile', 'profilu': 'profile',
    'kształtownik': 'profile', 'kształtowniki': 'profile',
    'ceownik': 'channel', 'ceowniki': 'channel',
    'dwuteownik': 'i_beam', 'dwuteowniki': 'i_beam',

    # Rury / Tubes & Pipes
    'rura': 'tube', 'rury': 'tube', 'rurę': 'tube',
    'przewód': 'pipe', 'przewody': 'pipe', 'przewodu': 'pipe',
    'rohr': 'tube', 'leitung': 'pipe',

    # Uchwyty dodatkowe / Additional Mounts
    'mocowanie': 'mount', 'mocowania': 'mount', 'mocowaniu': 'mount',
    'zacisk': 'clamp', 'zaciski': 'clamp', 'zacisku': 'clamp',
    'klamra': 'clamp', 'klamry': 'clamp',
    'befestigung': 'mount', 'klemme': 'clamp',
}

# Component categories for grouping similar components
# Used for category-based fallback estimation when no direct match found
COMPONENT_CATEGORIES = {
    # Structural
    'bracket': 'structural', 'frame': 'structural', 'plate': 'structural',
    'base': 'structural', 'foot': 'structural', 'profile': 'structural',
    'channel': 'structural', 'i_beam': 'structural', 'mount': 'structural',

    # Covers & Housings
    'cover': 'housing', 'housing': 'housing',

    # Drive Systems
    'motor': 'drive', 'drive': 'drive', 'gearbox': 'drive',
    'reducer': 'drive', 'gear': 'drive', 'pulley': 'drive',

    # Motion & Actuation
    'actuator': 'motion', 'cylinder': 'motion', 'piston': 'motion',
    'guide': 'motion', 'rail': 'motion', 'bearing': 'motion',
    'bushing': 'motion', 'shaft': 'motion', 'axis': 'motion',
    'wheel': 'motion', 'roller': 'motion', 'drum': 'motion',

    # Fasteners
    'screw': 'fastener', 'bolt': 'fastener', 'nut': 'fastener',
    'rivet': 'fastener', 'clamp': 'fastener',

    # Conveyors
    'conveyor': 'conveyor', 'belt_conveyor': 'conveyor',

    # Electrical & Control
    'sensor': 'electrical',

    # Mechanical Parts
    'spring': 'mechanical', 'valve': 'mechanical', 'filter': 'mechanical',
    'cartridge': 'mechanical', 'connector': 'mechanical', 'adapter': 'mechanical',

    # Tubes & Pipes
    'tube': 'piping', 'pipe': 'piping',
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

    def get_component_category(self, name: str) -> str | None:
        """
        Get category for component based on canonicalized name.

        Args:
            name: Component name (will be canonicalized)

        Returns:
            Category string or None if not found
        """
        canonical = self.canonicalize_component_name(name)
        if not canonical:
            return None

        # Check each token in canonical name for category match
        tokens = canonical.split()
        for token in tokens:
            if token in COMPONENT_CATEGORIES:
                return COMPONENT_CATEGORIES[token]

        return None

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
