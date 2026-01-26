import json
import logging
import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def _generate(self, prompt: str, json_mode: bool = False, timeout: int = 600) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        r = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def extract_clauses(self, document_text: str, max_chars: int = 12000) -> dict:
        """
        Extract scope items and exclusions from an offer document.

        Returns:
            {
                "scope": ["punkt 1", "punkt 2", ...],
                "exclusions": ["wykluczenie 1", ...],
                "critical_exclusions": ["kluczowe wykluczenie 1", ...]
            }
        """
        text = (document_text or "").strip()
        if not text:
            return {"scope": [], "exclusions": [], "critical_exclusions": []}

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[tekst ucięty]"

        prompt = (
            "Jesteś ekspertem od analizy ofert handlowych i technicznych.\n"
            "Przeanalizuj poniższy dokument ofertowy i wyodrębnij:\n\n"
            "1. **scope** - lista punktów opisujących zakres prac (co oferta zawiera)\n"
            "2. **exclusions** - lista wykluczeń (czego oferta NIE zawiera)\n"
            "3. **critical_exclusions** - z listy wykluczeń wybierz te KRYTYCZNE,\n"
            "   które chronią interes firmy (odpowiedzialność, gwarancja, kary umowne,\n"
            "   warunki płatności, ograniczenia zakresu)\n\n"
            "Odpowiedz WYŁĄCZNIE jako JSON w formacie:\n"
            '{"scope": ["..."], "exclusions": ["..."], "critical_exclusions": ["..."]}\n\n'
            "Każdy punkt powinien być zwięzły (1-2 zdania), po polsku.\n"
            "Jeśli dokument nie zawiera danej sekcji, zwróć pustą listę.\n\n"
            "DOKUMENT:\n"
            f"{text}\n"
        )

        raw = self._generate(prompt, json_mode=True)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                logger.error(f"Failed to parse clauses JSON: {raw[:200]}")
                return {"scope": [], "exclusions": [], "critical_exclusions": []}

        return {
            "scope": data.get("scope", []),
            "exclusions": data.get("exclusions", []),
            "critical_exclusions": data.get("critical_exclusions", []),
        }

    def check_offer_safety(self, selected_clauses: list[str], all_critical: list[str]) -> dict:
        """
        Check if an offer is missing critical exclusions.

        Returns:
            {
                "safe": bool,
                "missing": ["brakujące wykluczenie 1", ...],
                "warnings": ["ostrzeżenie 1", ...]
            }
        """
        if not all_critical:
            return {"safe": True, "missing": [], "warnings": []}

        prompt = (
            "Jesteś ekspertem od weryfikacji ofert.\n"
            "Sprawdź, czy konstruktor nie pominął krytycznych wykluczeń w ofercie.\n\n"
            "WYBRANE KLAUZULE W OFERCIE:\n"
            + "\n".join(f"- {c}" for c in selected_clauses)
            + "\n\nLISTA KRYTYCZNYCH WYKLUCZEŃ (z bazy wiedzy):\n"
            + "\n".join(f"- {c}" for c in all_critical)
            + "\n\nSprawdź, które krytyczne wykluczenia NIE zostały uwzględnione.\n"
            "Odpowiedz jako JSON:\n"
            '{"safe": true/false, "missing": ["..."], "warnings": ["..."]}\n'
        )

        raw = self._generate(prompt, json_mode=True)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                logger.error(f"Failed to parse safety JSON: {raw[:200]}")
                return {"safe": False, "missing": all_critical, "warnings": ["Nie udało się zweryfikować"]}

        return {
            "safe": data.get("safe", False),
            "missing": data.get("missing", []),
            "warnings": data.get("warnings", []),
        }
