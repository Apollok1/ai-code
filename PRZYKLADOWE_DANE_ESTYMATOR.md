# Przykładowe Dane do Prezentacji CAD Estimator Pro

## DZIAŁ 131 - AUTOMOTIVE

### Przykład 1: Stacja Montażowa - Zespalanie Podwozia (PROSTY)

**Nazwa projektu:** `Stacja montażowa VW - zespalanie komponentów podwozia`

**Departament:** Automotive (131)

**Opis do wklejenia:**
```
Stacja montażowa do zespalania komponentów podwozia dla linii produkcyjnej VW Golf.

WYMIARY OGÓLNE:
- 2000 x 1500 x 1800 mm
- Masa stacji: ~450 kg
- Materiał: stal S235JR

GŁÓWNE KOMPONENTY:

1. RAMA SPAWANA
   - Profile stalowe 80x80x4mm
   - Spawanie MIG/MAG
   - Lakierowanie proszkowe RAL 7035

2. SYSTEM DOCISKOWY
   - 2x cylinder pneumatyczny ISO 15552 Ø63mm, skok 100mm
   - Siła docisku: 5 kN każdy
   - Zasilanie: 6 bar

3. POZYCJONOWANIE
   - Prowadnice liniowe THK HRW21 (długość 800mm)
   - Śruby trapezowe TR16x4
   - Tuleje prowadzące brązowe

4. STEROWANIE
   - PLC Siemens S7-1200 CPU 1214C
   - Panel operatorski HMI 7" KTP700
   - 4x czujniki indukcyjne M12
   - 2x czujniki ciśnienia 0-10 bar
   - Zawory pneumatyczne 5/2 24VDC

5. BEZPIECZEŃSTWO
   - Osłony z poliwęglanu 8mm
   - Wyłącznik bezpieczeństwa kategorii 3
   - Kurtyny świetlne typ 4
   - Sygnalizacja świetlno-dźwiękowa

6. PODSTAWA I TRANSPORT
   - Płyta stołu spawana 10mm
   - 4x kółka skrętne Ø125mm z hamulcem
   - Stopki regulacyjne M16

WYMAGANIA SPECJALNE:
- Norma IATF 16949
- Czas cyklu docelowy: < 45 sekund
- Dokumentacja 2D zgodnie z GD&T ASME Y14.5
- Rysunki spawalnicze zgodnie z ISO 2553
- CE marking required
```

**Oczekiwany wynik:** ~120-140 godzin total

---

### Przykład 2: Oprawa Kontrolna - Karoseria (ŚREDNI)

**Nazwa projektu:** `Oprawa kontrolna BMW - wymiarowanie karoserii F30`

**Departament:** Automotive (131)

**Opis do wklejenia:**
```
Oprawa kontrolna do sprawdzania geometrii karoserii BMW serii 3 (F30).
Kontrola 18 punktów pomiarowych z tolerancją ±0.2mm.

WYMIARY OGÓLNE:
- 3500 x 2000 x 1200 mm
- Konstrukcja stalowa + aluminiowa
- Masa całkowita: ~850 kg

GŁÓWNE ZESPOŁY:

1. RAMA BAZOWA
   - Profile aluminiowe Bosch Rexroth 45x90
   - Połączenia śrubowe M8 klasa 8.8
   - Płyta bazowa aluminiowa 20mm (AW-5083)
   - Regulowane nogi (kompensacja poziomu ±5mm)

2. SYSTEM LOKALIZOWANIA (18 punktów)
   - 12x sworzni lokalizujących Ø12h6 z sygnalizacją
   - 6x piny wycentrowujące stożkowe 1:50
   - Elementy wymienne (wear parts) - stal nierdzewna 1.4301
   - Sprężyny powrotne: siła 50N każda

3. DOCISK PNEUMATYCZNY
   - 8x cylindry zaciskowe SMC MK 40-20
   - Regulacja siły docisku 100-500N
   - Sygnalizacja pozycji (czujniki reed)
   - Dystrybutor powietrza SMC z regulatorami

4. SYSTEM POMIAROWY
   - 6x czujniki laserowe Keyence LK-G80
   - Dokładność: ±5µm
   - Interface: EtherCAT
   - Bracket regulowane 3D (X/Y/Z ±50mm)

5. AUTOMATYKA I WIZUALIZACJA
   - PLC Siemens S7-1500 CPU 1515-2 PN
   - TP1500 Comfort Panel 15.6"
   - Komunikacja z MES przez PROFINET
   - Barcode scanner dla nr VIN
   - Protokół OK/NOK + zapis do bazy danych

6. ERGONOMIA I BEZPIECZEŃSTWO
   - Obszar roboczy podświetlany LED 6000K
   - Mat ergonomiczny 20mm
   - Emergency stop + 2-ręczne sterowanie
   - Acoustic signal przy błędzie pomiaru
   - Light tower 3-kolorowa

WYMAGANIA TECHNICZNE:
- Powtarzalność pomiaru: ±0.05mm
- Czas cyklu kontroli: < 90 sekund
- Kalibrowalne elementy pomiarowe (certyfikat DAkkS)
- Dokumentacja zgodna z VDA 6.3
- PPAP Level 3 documentation
- CMM measurement protocol (initial)

SPECYFIKACJE MATERIAŁOWE:
- Sworznie: stal nierdzewna hartowana 60 HRC
- Powierzchnie robocze: anodowane na twardo
- Powłoka zewnętrzna: Triflex Pro RAL 5012
```

**Oczekiwany wynik:** ~220-280 godzin total

---

### Przykład 3: Fixture Spawalniczy - Rama Podwozia (ZŁOŻONY)

**Nazwa projektu:** `Uchwyt spawalniczy Mercedes - rama podwozia Sprinter`

**Departament:** Automotive (131)

**Opis do wklejenia:**
```
Kompleksowy uchwyt spawalniczy do produkcji ram podwozia Mercedes Sprinter.
Proces: robot spawalniczy + spawanie manualne w 3 stacjach.

WYMIARY I WAGA:
- Wymiary gabarytowe: 4500 x 2800 x 1600 mm
- Masa uchwytu: ~2100 kg (bez detalu)
- Detal: rama stalowa 180 kg

MODUŁ 1: STACJA BAZOWA

1.1 KONSTRUKCJA NOŚNA
   - Profiles stalowe HEA 160 + HEB 120
   - Płyta bazowa 30mm steel S355
   - Spawanie MAG zgodnie z ISO 3834-2
   - Obróbka CNC płyty bazowej (flatness 0.1mm/m)

1.2 SYSTEM LOKALIZOWANIA
   - 24x sworznie lokalizujące Ø16h6 z hardened bushings
   - 8x piny stożkowe 1:50 dla pozycjonowania głównego
   - Tuleje wymienne z brązu CuSn12
   - Kompensacja zużycia: wymienne wkładki co 50k cykli

1.3 DOCISK PNEUMATYCZNY (15 punktów)
   - Cylindry swing clamp Destaco 860-M
   - Siła docisku: 8 kN każdy
   - Czas zacisku: < 2s
   - Monitoring siły przez pressure switches

1.4 DOCISK HYDRAULICZNY (rama główna - 4 punkty)
   - Cylindry hydrauliczne 100/50x200mm
   - Siła: 40 kN każdy
   - Agregat hydrauliczny: pompa 7.5kW, zbiornik 50L
   - Ciśnienie robocze: 160 bar

MODUŁ 2: STACJA ROBOT SPAWALNICZY

2.1 POZYCJONER OBROTOWY
   - Stół obrotowy Ø1200mm, udźwig 500kg
   - Serwo drive z przekładnią precyzyjną 1:50
   - Dokładność pozycjonowania: ±0.05°
   - Zakres obrotu: 0-360° continuous
   - Prędkość: 0.1-10 rpm variable

2.2 SYSTEM WSPÓŁPRACY Z ROBOTEM
   - Robot interface: PROFINET RT
   - Synchronizacja pozycji z trajektorią spawania
   - Anti-collision sensors: 6x laserowe obszarowe
   - Torch cleaning station integration

2.3 SYSTEMY POMOCNICZE
   - Odsysanie spawalnicze: 2x ramiona Ø150mm
   - Uziemienie spawalnicze: 4x punkty Ø25mm
   - Chłodzenie uchwytu: water cooling circuit
   - Temperatura robocza: monitorowana (max 60°C)

MODUŁ 3: STACJA SPAWANIA MANUALNEGO

3.1 MANIPULATOR
   - Tilt unit: zakres -15° do +90°
   - Cylinder hydrauliczny 80/40x300mm
   - Blokada pozycji co 15° (mechanical locks)
   - Sterowanie foot pedal 2-kierunkowe

3.2 STANOWISKO OPERATORA
   - Platforma podnoszona: range 500-1100mm
   - Balustrady bezpieczeństwa zgodne z EN ISO 14122
   - Lighting: 2x LED floodlight 100W 5000K
   - Exhaust arm Ø200mm z damprem

AUTOMATYKA I STEROWANIE:

1. PLC - GŁÓWNY
   - Siemens S7-1500 CPU 1516-3 PN/DP
   - ET200SP distributed I/O (6 modules)
   - Safety relay Pilz PNOZ X3
   - UPS backup 20 minut

2. HMI
   - Panel TP2200 Comfort 22"
   - Visualisation: recipe management (50 wariantów)
   - Trend logs dla sił docisku
   - OEE monitoring

3. KOMUNIKACJA
   - PROFINET z robotem spawalniczym
   - OPC UA do MES
   - Barcode scanner part tracking
   - RFID dla identyfikacji palety

4. BEZPIECZEŃSTWO
   - Safety fence z bramką interlock
   - Light curtains typ 4 (3 strefy)
   - 4x emergency stops (mushroom buttons)
   - Two-hand control dla tiltera
   - Safety relay kategorii 4

SYSTEMY POMOCNICZE:

1. COOLING SYSTEM
   - Water chiller 3kW
   - Flow: 20 L/min
   - Temperature control ±2°C
   - Low flow alarm

2. PNEUMATYKA
   - Ciśnienie zasilania: 6 bar ±0.5
   - Filter-regulator-lubricator unit
   - Manifold distribution 16 sekcji
   - Pressure monitoring każdej strefy

3. HYDRAULIKA
   - Agregat Bosch Rexroth 7.5kW
   - Filtration: 10µm
   - Oil cooler z wentylatorem
   - Level + temperature monitoring

SPECYFIKACJE JAKOŚCIOWE:
- Powtarzalność pozycjonowania: ±0.15mm
- Czas cyklu: 180 sekund (takt produkcji)
- MTBF: > 10,000 cykli
- Dokumentacja: PPAP Level 4
- Welding procedure specifications (WPS)
- CMM measurement report dla uchwytu
- FAT (Factory Acceptance Test) protocol
- Spare parts list z recommended stock
- Maintenance schedule (daily/weekly/monthly)

NORMY I CERTYFIKACJE:
- ISO 3834-2 (welding quality)
- ISO 9606 (welder qualification)
- Mercedes-Benz MBN 10250 (fixture standard)
- CE marking (Machinery Directive 2006/42/EC)
- DGUV Vorschrift 3 (electrical safety inspection)
```

**Oczekiwany wynik:** ~480-620 godzin total

---

## DZIAŁ 135 - SPECIAL PURPOSE MACHINES

### Przykład 1: Linia Pakowania Automatycznego (ŚREDNI)

**Nazwa projektu:** `Linia pakowania automatycznego - produkty spożywcze`

**Departament:** Special Purpose (135)

**Opis do wklejenia:**
```
Automatyczna linia pakowania produktów spożywczych (ciastka) do pudełek kartonowych.
Wydajność: 60 pudełek/minutę.

WYMIARY LINII:
- Długość całkowita: 8500 mm
- Szerokość: 2200 mm
- Wysokość: 2400 mm

MODUŁ 1: ZASILANIE PRODUKTEM

1.1 TRANSPORTER WEJŚCIOWY
   - Taśma modułowa PP 800mm szerokości
   - Napęd: servomotor SEW 0.75kW z przekładnią
   - Prędkość: 0-60 m/min variable
   - Encoder position feedback

1.2 SYSTEM NAKŁADANIA
   - Robot pick-and-place SCARA (zasięg 600mm)
   - Chwytaki próżniowe: 8x przyssawki Ø30mm
   - Generator próżni: ejector SMC ZH13DS
   - Vacuum switch dla kontroli chwytu
   - Cykl: 1 sekunda (60 cykli/min)

MODUŁ 2: KARTONOWANIE

2.1 MAGAZYN PUDELEK
   - Kaseta na flat-packed boxes (300 szt.)
   - Cylinder pneumatyczny dozowania 100/20x50mm
   - Czujnik poziomu LOW/HIGH
   - Signalizacja konieczności uzupełnienia

2.2 SKŁADACZ PUDELEK (ERECTOR)
   - Mechanizm klinowy formujący pudełko
   - 4x cylindry 40/20x100mm synchronizowane
   - Czas cyklu: 0.8 sekundy
   - Podajnik kleju hot-melt (2 linie)

2.3 PRZENOŚNIK PUDELEK
   - Timing belt 50mm x 8000mm
   - Prowadzenie boczne servo-adjustable
   - Zakres pudełek: 100-250mm długość
   - Zmiana formatu: tool-less w < 5 minut

MODUŁ 3: PAKOWANIE

3.1 STACJA NAPEŁNIANIA
   - 3x roboty SCARA w linii
   - Każdy pakuje 20 ciastek (układ 4x5)
   - Vision system dla orientacji produktu
   - Kamera Cognex In-Sight 7000
   - Lighting: LED backlight + front light

3.2 KONTROLA WAGI
   - Waga dynamiczna: zakres 0-500g
   - Dokładność: ±0.5g
   - Reject system: cylinder pusher + skip
   - Interface: EtherCAT do PLC

MODUŁ 4: ZAMYKANIE I ETYKIETOWANIE

4.1 ZAMYKANIE PUDEŁEK
   - Zaklejanie top flap: hot-melt 2 linie
   - Docisk: rolki silikonowe Ø80mm
   - System podgrzewania kleju: 180°C
   - Monitoring temperatury z alarmem

4.2 APLIKATOR ETYKIET
   - Print & Apply system Cab Hermes+
   - Drukarka termotransferowa 203dpi
   - Aplikator pneumatyczny: air-blow
   - Weryfikacja barcode przez scanner
   - Reject przy błędzie druku/aplikacji

MODUŁ 5: TRANSPORT WYJŚCIOWY

5.1 PRZENOŚNIK DO PALETYZACJI
   - Belt conveyor 600mm width
   - Długość: 3000mm
   - Stopper pneumatyczny dla akumulacji
   - Photocells dla zliczania

5.2 REJECT LINE (boczna)
   - Diverter pneumatyczny
   - Kolektor braków z sygnalizacją pełności
   - Segregacja: underweight/overweight/no label

SYSTEM STEROWANIA:

1. PLC
   - Beckhoff CX5240 (IPC)
   - TwinCAT 3 runtime
   - EtherCAT master (30+ slaves)
   - 12x servo drives AX5000

2. HMI
   - Panel PC 21.5" Full HD touchscreen
   - TwinCAT HMI interface
   - Recipe management dla różnych produktów
   - Production statistics (OEE, downtime)
   - Alarm history z timestampami

3. KOMUNIKACJA
   - OPC UA server dla MES
   - MQTT dla cloud analytics (opcja)
   - Modbus TCP dla urządzeń peryferyjnych
   - Barcode data do ERP (SAP interface)

BEZPIECZEŃSTWO:

- Safety PLC: Pilz PSS 4000
- 4x kurtyny świetlne Sick C4000
- Safety gates z interlock: 3 strefy
- Emergency stops: 6 punktów
- Safe motion dla wszystkich serwo
- Kategoria bezpieczeństwa: 3, PLd

WYMAGANIA SPECJALNE:

- Food grade: wszystkie materiały kontaktowe
- IP65 rating dla strefy mokrej (opcja)
- Easy cleaning design (narzędzia: 30 min)
- Zmiana formatu: częściowo automatyczna
- Dokumentacja: rysunki złożeniowe + BOM
- Electrical schematics w EPLAN
- Program PLC dokumentowany + backup
- FAT w zakładzie producenta
- SAT on-site z szkoleniem operatorów (3 dni)

WYDAJNOŚĆ:
- Nominal: 60 boxes/min
- Max: 75 boxes/min (overclocking dla krótkich okresów)
- OEE target: >85%
- MTBF: >200 hours
```

**Oczekiwany wynik:** ~520-680 godzin total

---

### Przykład 2: Maszyna Testująca - Hydraulika (ŚREDNI)

**Nazwa projektu:** `Stanowisko testowe zaworów hydraulicznych - testy ciśnieniowe`

**Departament:** Special Purpose (135)

**Opis do wklejenia:**
```
Automatyczne stanowisko do testów zaworów hydraulicznych.
Test ciśnieniowy, szczelności, flow rate, response time.

WYMIARY:
- 2500 x 1800 x 2000 mm
- Konstrukcja: profil aluminiowy + obudowa blaszana
- Masa: ~600 kg

SEKCJA 1: HYDRAULIKA TESTOWA

1.1 AGREGAT HYDRAULICZNY
   - Pompa: Bosch Rexroth A4VG 12kW
   - Ciśnienie max: 350 bar
   - Flow: 0-60 L/min variable
   - Zbiornik: 120L z chłodnicą
   - Filtracja: 5µm absolute

1.2 MANIFOLD TESTOWY
   - Block aluminium 7075-T6 (custom)
   - 8x złącza testowe G1/4
   - Czujniki ciśnienia: 0-400bar ±0.25% FS
   - Czujniki temperatury: PT100 klasa A
   - Przepływomierz: turbinowy 0.5-60 L/min

1.3 ZAWORY STERUJĄCE
   - 4x zawory proporcjonalne Atos DHZO
   - Sterowanie: ±10V analog
   - Response time: <15ms
   - Amplifier wbudowany

SEKCJA 2: STACJA TESTOWA (uchwyt detalu)

2.1 PRZYRZĄD MOCUJĄCY
   - Quick-change adapter dla różnych zaworów
   - Centralny gwint mocujący + O-ring seal
   - 4x sworznie pozycjonujące
   - Docisk pneumatyczny 3kN
   - Złącza hydrauliczne self-sealing

2.2 DETEKCJA SZCZELNOŚCI
   - Acoustic emission sensor
   - Bariera olejowa z czujnikiem poziomu
   - Visual inspection camera (opcja)
   - Leak rate: detection < 0.1 mL/min

SEKCJA 3: STEROWANIE I POMIARY

3.1 PLC I ACQUISITION
   - Siemens S7-1500 CPU 1515T-2 PN
   - TM Count 2x24V dla enkoderów
   - TM PosInput 1 dla high-speed measurement
   - 4x moduły AI: 16bit resolution

3.2 DATA ACQUISITION
   - NI cDAQ-9178 chassis
   - 2x moduly AI 24-bit dla precision pressure
   - Sample rate: 10 kHz dla dynamiki
   - LabVIEW RT dla real-time logging

3.3 HMI
   - Panel TP1900 Comfort Pro 19"
   - Test sequences programmable
   - Live waveform display (pressure/flow)
   - Pass/Fail indication z alarmem audio
   - Test report generation (PDF + CSV)

SEKWENCJE TESTOWE:

TEST 1: PRESSURE PROOF (30 sekund)
- Ramp-up do 1.5x nominal pressure
- Hold 20 sekund
- Monitoring pressure drop (<2%)
- Criteria: no leaks, stable pressure

TEST 2: FLOW CHARACTERISTICS (60 sekund)
- Sweeping pressure 50-300 bar
- Measurement flow at 10 points
- Compare vs. nominal curve (±5%)
- Hysteresis check (up/down)

TEST 3: RESPONSE TIME (dynamic)
- Step input: 0-100% otwarcia
- Measure: t10-90 (target: <50ms)
- Overshoot check (<10%)
- Settling time (<100ms)

TEST 4: LEAK TEST (internal + external)
- Ciśnienie robocze przez 60s
- Detection acoustic + visual
- Criteria: 0 mL/min external, <0.05 mL/min internal

TEST 5: ENDURANCE SIMULATION (opcja)
- 1000 cykli otwarcie/zamknięcie
- Monitoring wear (flow reduction)
- Temperature monitoring
- Criteria: <5% flow degradation

BEZPIECZEŃSTWO:

- Pressure relief valve: 380 bar setting
- Burst disc backup (420 bar)
- Przezroczysta osłona z poliwęglanu bulletproof
- Door interlock: blokada przy ciśnieniu >50 bar
- Emergency depressurization: <3 sekundy
- Oil temperature alarm >70°C

WYMAGANIA:

- Powtarzalność pomiarów: ±0.5%
- Czas testu kompletnego: <4 minuty
- Traceability: każdy test zapisany z serial number
- Kalibrowalne sensory (roczna kalibracja)
- Data export do QA database
- Dokumentacja zgodna z ISO 17025 (jeśli certyfikacja wymagana)

OPCJE DODATKOWE:
- Vision system dla wear inspection
- Temperature cycling chamber integration
- Multi-valve testing (4 stanowiska równoległe)
- Contamination injection system (test filtracji)
```

**Oczekiwany wynik:** ~280-360 godzin total

---

### Przykład 3: Automat CNC Loading (PROSTY)

**Nazwa projektu:** `Robot załadowczy do centrum obróbczego CNC - automotive`

**Departament:** Special Purpose (135)

**Opis do wklejenia:**
```
Robot do automatycznego załadunku/rozładunku części aluminiowych do centrum obróbczego CNC.
Integracja z istniejącą maszyną Hermle C40U.

WYMIARY:
- 3000 x 2500 x 2600 mm (robot + szafa sterująca)
- Wysięg: 1400 mm

GŁÓWNE KOMPONENTY:

1. ROBOT PRZEMYSŁOWY
   - Fanuc M-20iD/25 (udźwig 25kg)
   - Reach: 1853mm
   - Repeatability: ±0.02mm
   - Kontroler: R-30iB Plus
   - Wersja: wash-down (IP67)

2. CHWYTAK (GRIPPER)
   - Custom design: 2-szczękowy pneumatyczny
   - Opening: 50-250mm
   - Siła chwytu: 500N adjustable
   - Wymiana szczęk tool-less (<2 min)
   - 3 zestawy szczęk dla różnych części

3. SYSTEM WIZYJNY
   - Kamera Cognex In-Sight 8405
   - Mounted na nadgarstku robota
   - Zadania: orientacja części, QC check
   - Lighting: 4x LED strobes 6500K
   - Cycle time: <1 sekunda na zdjęcie

4. MAGAZYNY CZĘŚCI

4.1 INPUT (surówki)
   - Europaleta 1200x800mm
   - Grid: 6x4 pozycje (24 części)
   - Height detection: laser sensor
   - Podnośnik pneumatyczny (lift table)

4.2 OUTPUT (obrobione)
   - Identyczny jak input
   - Segregacja: GOOD / SCRAP
   - Diverter z licznikiem sztuk

5. STACJA CZYSZCZENIA
   - Air blow-off 6 bar (chip removal)
   - 6x dysze ringowe
   - Opcja: ultrasonic bath integration

6. INTEGRACJA Z CNC

6.1 DRZWI AUTOMATYCZNE
   - Retrofit istniejących drzwi Hermle
   - Actuator elektryczny 24VDC
   - Sensor door position: hall effect
   - Safety: interlock z CNC

6.2 KOMUNIKACJA
   - Profinet I/O z CNC Heidenhain TNC640
   - Handshake signals:
     * Part ready to load
     * Part ready to unload
     * Cycle complete
     * Machine alarm
   - Emergency stop chain integration

7. STEROWANIE

7.1 PLC (nadrzędny)
   - Siemens S7-1200 CPU 1215C
   - ET200SP remote I/O przy palecie
   - Komunikacja z robotem przez DI/DO + Profinet
   - Recipe management (10 części różnych)

7.2 HMI
   - KTP1200 Basic PN 12.1"
   - Wybór programu obróbki
   - Counter production/scrap
   - Robot status display
   - Manual jog mode dla teach-in

8. BEZPIECZEŃSTWO

- Ogrodzenie: aluminium profiles + siatka
- 2x bramki interlock z blokadą elektromagnetyczną
- Kurtyna świetlna Sick C4000 Fusion przy wejściu
- Scanner obszarowy Sick S3000 przy palecie
- Emergency stops: 4 lokalizacje
- Category 3 safety architecture
- Audyt TÜV do CE marking

FUNKCJONALNOŚĆ:

- Autonomous mode: 24-hour lights-out
- Alarm przy braku części na palecie input
- Automatic pallet change signal (operator call)
- Vision-based quality check (crack detection)
- Traceability: barcode scan przed obróbką
- Cycle time: 18 sekund (load+unload)
- Availability target: >95%

WYMAGANIA DOKUMENTACJI:

- Layout drawings 2D + 3D STEP
- Electrical schematics EPLAN
- Pneumatic diagrams ISO 1219
- Robot programs backup + annotations
- PLC program (TIA Portal project)
- Risk assessment (EN ISO 12100)
- CE declaration of conformity
- O&M manual (PL + EN)
- Spare parts list z zaleconym zapasem
```

**Oczekiwany wynik:** ~180-240 godzin total

---

## JAK UŻYWAĆ TYCH PRZYKŁADÓW W PREZENTACJI

### STRATEGIA DEMONSTRACJI:

**1. Zacznij od prostego przykładu (131 - Stacja Montażowa)**
   - Pokaże podstawową funkcjonalność
   - Szybka estymacja (~10 sekund)
   - Wyniki łatwe do zrozumienia

**2. Pokaż różnicę między Single vs Multi-Model**
   - Single: użyj "Oprawa Kontrolna" (131)
   - Multi: użyj tego samego projektu w trybie multi-model
   - Porównaj szczegółowość wyników

**3. Demonstracja Special Purpose (135 - Linia Pakowania)**
   - Pokaż że system radzi sobie z inną branżą
   - Inne konteksty, inne complexity factors
   - Różne komponenty w wyniku

**4. Pokaż Import z Excel** (opcja)
   - Jeśli masz czas, stwórz prosty Excel z listą komponentów
   - Pokaż że można importować istniejące struktury

**5. Learning Loop** (finał)
   - Wróć do pierwszego projektu
   - Wprowadź "actual hours" (zmyślone ale realistyczne)
   - Pokaż jak system się uczy

---

## PRZYKŁADOWE "ACTUAL HOURS" DO LEARNING (gdy zademonstrujesz uczenie)

Dla **Stacja Montażowa VW**:

```
Frame Base Assembly:
  Layout 3D: 3h (estymacja) → 3.5h (actual)
  Detail 3D: 12h → 11h (szybsze niż myślano)
  2D Doc: 4h → 5h (więcej wymiarów niż zakładano)

Pneumatic System:
  Layout 3D: 2h → 2h (zgodne)
  Detail 3D: 8h → 10h (więcej pipes niż myślano)
  2D Doc: 3h → 3h (zgodne)

Control System (PLC):
  Layout 3D: 3h → 4h (więcej I/O)
  Detail 3D: 15h → 18h (złożone kable)
  2D Doc: 5h → 6h (electrical schematics)

Total: 127h (estymacja) → 138h (actual)
Accuracy: +8.6% (bardzo dobra estymacja!)
```

---

## WSKAZÓWKI:

✅ **Kopiuj opisy 1:1** - są gotowe do wklejenia
✅ **Zacznij od prostych** - buduj złożoność stopniowo
✅ **Podkreślaj liczby** - 120h, ±20%, confidence HIGH
✅ **Pokazuj ryzyka i sugestie** - to dodaje wartość
✅ **Demonstruj uczenie** - to wyróżnik produktu

Powodzenia! 🎯
