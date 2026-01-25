import requests

class OllamaClient:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def summarize_pl(self, text: str, max_chars: int = 12000) -> str:
        text = (text or "").strip()
        if not text:
            return "Brak tekstu do podsumowania."

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Ucięto tekst do limitu w MVP]"

        prompt = (
            "Jesteś asystentem w firmie. Streszcz poniższy tekst po polsku.\n"
            "Wymagania:\n"
            "- najpierw 5–10 punktów najważniejszych informacji,\n"
            "- potem sekcja: 'Ryzyka / niejasności' (jeśli są),\n"
            "- potem: 'Następne kroki' (konkretne).\n\n"
            "TEKST:\n"
            f"{text}\n"
        )

        r = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
