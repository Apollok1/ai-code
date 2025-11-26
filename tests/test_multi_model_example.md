# Multi-Model Pipeline - Test Example

## 📋 Test Data dla Manualnego Testowania

### Przykład 1: Prosta Spawalnia

**Opis projektu:**
```
Projekt: Stacja spawalnicza do elementów stalowych

Wymagania:
- Spawanie elementów stalowych metodą MIG/MAG
- Automatyczny obrotnik z napędem elektrycznym
- Stół roboczy 2000x1000mm z rowkami T
- System wyciągowy dla oparów spawalniczych
- Panel sterowania z przyciskami STOP/START
- Obudowa zabezpieczająca z pleksi

Materiały:
- Konstrukcja: Stal S235JR
- Stół: Stal narzędziowa
- Obudowa: Pleksi 10mm
```

**Oczekiwane Wyniki:**

**Stage 1 (Technical Analysis):**
- Complexity: `medium` lub `high`
- Materials: ["Stal S235JR", "Stal narzędziowa", "Pleksi"]
- Standards: ["EN 1090" (spawanie), "ISO 3834" (jakość spawania)]
- Challenges: ["Precyzja obrotnika", "Wyciąg oparów", "Dostęp serwisowy"]

**Stage 2 (Structure):**
- Root components: ~6-8 głównych zespołów
  - Rama nośna
  - Stół spawalniczy
  - Obrotnik
  - System wyciągowy
  - Panel sterowania
  - Obudowa zabezpieczająca
- Depth: 2-3 poziomy
- Component count: 20-40 elementów

**Stage 3 (Hours):**
- Szacowane godziny: 80-150h
  - Rama i konstrukcja: ~30-40h
  - Stół spawalniczy: ~20-30h
  - Obrotnik z napędem: ~15-25h
  - System wyciągowy: ~10-15h
  - Panel i elektryka: ~8-12h
  - Obudowa: ~12-18h
  - Dokumentacja 2D: ~25-35h

**Stage 4 (Risks):**
- Ryzyka:
  - Medium: "Precyzja obrotnika - może wymagać dodatkowych testów"
  - Low: "Integracja systemu wyciągowego"
- Suggestions:
  - "Rozważ użycie standardowego obrotnika z katalogów"
  - "Panel sterowania - można użyć gotowe rozwiązanie zamiast custom"
- Assumptions:
  - "Dostępność standardowych elementów napędowych"
  - "Klient dostarczy specyfikację systemu spawalniczego"

---

### Przykład 2: Przenośnik Taśmowy

**Opis projektu:**
```
Projekt: Przenośnik taśmowy do transportu paczek

Parametry:
- Długość: 5 metrów
- Szerokość taśmy: 600mm
- Obciążenie: do 50kg/m
- Prędkość: regulowana 0-20m/min
- Napęd: silnik elektryczny 0.75kW
- Sterowanie: falownik + panel dotykowy HMI
- Konstrukcja: aluminium (profile 40x40)
- Rolki nośne: co 300mm

Wymagania dodatkowe:
- Obudowa boczna z blachy aluminiowej
- Czujniki krańcowe
- Lampka sygnalizacyjna LED
```

**Oczekiwane Wyniki:**

**Stage 1:**
- Complexity: `medium`
- Materials: ["Aluminium 40x40", "Taśma PVC", "Stal (rolki)"]
- Standards: ["ISO 5048" (przenośniki), "EN 60204-1" (bezpieczeństwo elektryczne)]
- Challenges: ["Napinanie taśmy", "Regulacja prędkości", "Kalibracja czujników"]

**Stage 2:**
- ~8-10 głównych zespołów
- Depth: 2-3
- Component count: 30-50

**Stage 3:**
- Szacowane: 60-100h
- Profile aluminum: szybsze niż stal (lżejsze, prostsze)
- Rolki: możliwe standardowe z katalogów

**Stage 4:**
- Risk: "Dobór taśmy - może wymagać konsultacji z dostawcą"
- Suggestion: "Użyć standardowych profili aluminiowych zamiast spawanej konstrukcji"
- Assumption: "Falownik i HMI z katalogu (Siemens/Allen-Bradley)"

---

## 🧪 Procedura Testowania

### Krok 1: Uruchom Aplikację
```bash
cd /home/user/ai-code
docker-compose up -d
streamlit run src/cad/presentation/app.py
```

### Krok 2: Konfiguracja w Sidebar
1. Włącz "Multi-Model Pipeline (4 etapy)"
2. Rozwiń "⚙️ Wybór modeli per etap"
3. Wybierz modele (zalecane):
   - Stage 1: `qwen2.5:14b` (jeśli masz) lub `qwen2.5:7b`
   - Stage 2: `qwen2.5:7b`
   - Stage 3: `qwen2.5:7b`
   - Stage 4: `qwen2.5:14b` (jeśli masz) lub `qwen2.5:7b`

### Krok 3: Wprowadź Dane
1. Wybierz dział (np. "131 - Automotive" lub "135 - Special Purpose")
2. Nazwa projektu: "Test - Spawalnia" (lub z Example 2)
3. Opis: Skopiuj opis z Example 1 lub 2
4. Kliknij "🤖 Analizuj z AI"

### Krok 4: Obserwuj Wyniki
Sprawdź czy pojawiają się:
- ✅ Progress (jeśli działa) lub spinner
- ✅ "Multi-Model Pipeline zakończony: Xh, Y komponentów"
- ✅ Sekcja "🎯 Wyniki Multi-Model Pipeline"
  - 1️⃣ Analiza Techniczna (complexity badge, materials)
  - 2️⃣ Struktura Komponentów (component count)
  - 3️⃣ Estymacja Godzin (metrics)
  - 4️⃣ Analiza Ryzyk (risks, suggestions)

### Krok 5: Sprawdź Logi
```bash
docker-compose logs -f streamlit
```

Szukaj linii:
```
INFO - Starting multi-model pipeline execution
INFO - Models: Stage1=..., Stage2=..., Stage3=..., Stage4=...
INFO - Stage 1 complete: Complexity=..., Materials=...
INFO - Stage 2 complete: Components=..., Depth=...
INFO - Stage 3 complete: Estimated ... components, Total hours=...
INFO - Stage 4 complete: Risks=..., Suggestions=...
INFO - Multi-model pipeline execution complete
```

### Krok 6: Porównaj z Oczekiwaniami
- Czy complexity jest sensowny?
- Czy materials są poprawnie zidentyfikowane?
- Czy liczba komponentów jest realistyczna?
- Czy godziny są w rozsądnym przedziale?
- Czy ryzyka mają sens?

---

## 🐛 Troubleshooting

### Problem: "Model not found"
- Sprawdź: `ollama list` - czy model jest pobrany?
- Pobierz: `ollama pull qwen2.5:7b`

### Problem: "JSON parsing error"
- Model zwrócił nieprawidłowy JSON
- Sprawdź logi - będzie fallback extraction
- Spróbuj innego modelu dla tego stage'u

### Problem: "Empty results"
- Model zwrócił puste tablice
- Normalnie - będą defaulty
- Spróbuj bardziej szczegółowego opisu projektu

### Problem: "Stage X failed"
- Sprawdź logi: `docker-compose logs streamlit`
- Zobacz pełny error stack trace
- Możliwe przyczyny:
  - Model timeout (zwiększ timeout w config)
  - Model hallucination (zwrócił zły format)
  - Brak połączenia z Ollama

---

## ✅ Validation Checklist

Po teście sprawdź:

- [ ] Pipeline się wykonał bez crashu
- [ ] Wszystkie 4 stage'y się skończyły
- [ ] Metadata jest wypełniona
- [ ] UI pokazuje wyniki z każdego stage'u
- [ ] Logi pokazują model names i outputs
- [ ] Complexity badge ma kolor
- [ ] Risks mają impact levels
- [ ] Suggestions są sensowne
- [ ] Assumptions są wymienione
- [ ] Lista komponentów jest dostępna

Jeśli wszystko ✅ - **Pipeline działa!** 🎉
