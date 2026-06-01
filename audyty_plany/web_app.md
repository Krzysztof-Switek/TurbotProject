# Plan: konwersja TurbotProject na narzędzie webowe (FastAPI + vanilla JS + Docker)

## Kontekst

Obecnie `Otolits_identyfication_program/` to aplikacja desktop oparta na `cv2.namedWindow` + `cv2.waitKey` + `tkinter` (do pobrania rozmiaru ekranu). Cała interakcja jest lokalna; uruchomienie wymaga GUI, sterowniki ekranu, manualnego ustawienia `cwd` itd. User chce postawić to jako narzędzie webowe na serwerze, wdrażane Dockerem, z wieloma badaczami pracującymi przez przeglądarkę.

Cel: zachować logikę domenową (YOLO detekcja, geometria boxów/wierszy, sortowanie + cropping wg konwencji `A_/B_`) i wystawić ją przez HTTP API + frontend rysujący na HTML5 Canvas. Serwer ma być **w 90% bezstanowy**: stan UI (boxy/wiersze/tryb) trzyma frontend, serwer robi tylko I/O obrazów, detekcję YOLO i finalne cropowanie.

> **Stan wyjściowy (czerwiec 2026):** desktop w branchu `main` (po merge `web_app`) ma już zaimplementowane: numerowanie wierszy "od dołu", walidację konwencji (max 3/wycinek), identyfikację A/B przez largest-gap Y, manual override `compartment_override`, wizualizację etykiet i ramek wycinków, tryb `EDIT_LABEL`, statusbar z licznikami, automatyczny zapis adnotacji YOLO obok zdjęcia (pseudo-labelling loop). Cała ta logika musi być portowana do web — patrz sekcja **"Compartments + numerowanie + pseudo-labelling (port z desktop)"**.

Decyzje uzgodnione z userem:
1. **Single-user dev** — brak auth, serwer za reverse proxy / w LAN.
2. **Źródło obrazów = katalogi na serwerze**, z możliwością nawigacji po drzewie i tworzeniem nowych katalogów (mkdir) z poziomu UI. **Nie ma uploadu z dysku** użytkownika.
3. **Frontend: vanilla JS + HTML5 Canvas** — zero build-step, jeden HTML serwowany przez backend.
4. **Backend: FastAPI + Uvicorn** — async, OpenAPI auto-docs.
5. **Root nawigacji**: jedna zmienna środowiskowa `DATA_ROOT` (domyślnie `/data` w kontenerze), jeden volume w Dockerze. Path traversal blokowany przez `Path.is_relative_to()`.
6. **Numerowanie wierszy "od dołu"** + sztywna walidacja max 3 wiersze/wycinek (oba spełnione w desktop, port 1:1 do web).
7. **Identyfikacja A/B** = automatyczna przez largest-gap Y (`_split_compartments` w `image_cropper.py`) + manual override przez `Row.compartment_override`.
8. **Bbox wycinka** = computed only (bounding-box wszystkich boxów + margines 40 px), bez manualnej edycji.
9. **Pseudo-labelling** = automatyczny zapis adnotacji YOLO `.txt` w lokalizacji **obok zdjęcia źródłowego** (np. `test_images/<name>.txt`). Każdy crop produkuje training data.

---

## Architektura

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (jeden tab = jedna "sesja")                        │
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────┐  │
│  │ FileBrowser  │  │ ImageCanvas          │  │ Toolbar   │  │
│  │ (drzewo +    │  │ (rysowanie boxów,    │  │ (b/l/d/v/r│  │
│  │  mkdir)      │  │  linii, eventy)      │  │  detect,  │  │
│  │              │  │                      │  │  crop)    │  │
│  └──────────────┘  └──────────────────────┘  └───────────┘  │
│         ▲                    ▲                     ▲        │
└─────────┼────────────────────┼─────────────────────┼────────┘
          │ fetch JSON / PNG   │                     │
┌─────────┴────────────────────┴─────────────────────┴────────┐
│  FastAPI (uvicorn) — bezstanowy                             │
│  /api/fs/list, /mkdir                                       │
│  /api/image/preview, /original                              │
│  /api/detect       (re-use AutoDetector, model in memory)   │
│  /api/crop         (re-use ImageCropper z custom output_dir)│
│  /static/*         (HTML/JS/CSS)                            │
└─────────────────────────────────────────────────────────────┘
          ▲
          │ volume mount
┌─────────┴───────────────────────────────────────────────────┐
│  /data  (DATA_ROOT) — obrazy źródłowe + wyjściowe cropy     │
└─────────────────────────────────────────────────────────────┘
```

Stan UI (mode, boxy, wiersze, `compartment_override`, aktualnie zaznaczony element) **w pełni po stronie frontendu**. Serwer dostaje stan dopiero przy crop (jako payload JSON) i wykonuje cropping. Backend trzyma w pamięci tylko: wczytany model YOLO (od startu) i ewentualnie LRU cache wczytanych obrazów (opcjonalne).

**Shared logic helpers** (zaimplementowane w `image_cropper.py`, używane przez **backend i sportowane do JS**):
- `compute_row_labels(rows) -> (labels, error)` — sortowanie, walidacja, split A/B przez largest gap Y, numerowanie "od dołu".
- `compute_compartment_bboxes(rows, margin=40, labels=) -> dict` — bbox wycinka jako bounding-box wszystkich boxów w jego wierszach + margines.

Te dwie funkcje **muszą mieć port JS 1:1** żeby UI mogło rysować etykiety i ramki bez round-tripa do serwera. Serwer używa Pythona wersji przy `/api/crop` (rekonstrukcja, walidacja, zapis adnotacji).

---

## Struktura plików (nowe + zmienione)

```
Otolits_identyfication_program/
├─ web/                                  ← NOWE
│  ├─ __init__.py
│  ├─ app.py                             # FastAPI app, mount routerów + static, startup load YOLO
│  ├─ config.py                          # env: DATA_ROOT, MODEL_PATH, MAX_PREVIEW_PX, ALLOWED_EXTS
│  ├─ routers/
│  │  ├─ __init__.py
│  │  ├─ fs.py                           # GET /api/fs/list, POST /api/fs/mkdir
│  │  ├─ image.py                        # GET /api/image/preview, /original
│  │  ├─ detect.py                       # POST /api/detect
│  │  └─ crop.py                         # POST /api/crop
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ fs_browser.py                   # safe_resolve(rel) + listing + mkdir
│  │  └─ image_service.py                # wrapper na ImageLoader bez Tkinter
│  └─ static/
│     ├─ index.html                      # layout: sidebar + canvas + toolbar
│     ├─ app.js                          # klasy FileBrowser, ImageCanvas, BBox, Row, Api
│     └─ style.css
├─ image_loader.py                       ← ZMIANA: usunąć Tkinter, dodać max_preview_px do init
├─ bounding_box.py                       ← bez zmian (re-use)
├─ row_detector.py                       ← bez zmian (re-use Row, _does_line_intersect_box,
│                                          _line_segments_intersect, _point_in_box w /api/crop)
├─ auto_detector.py                      ← bez zmian (re-use AutoDetector)
├─ image_cropper.py                      ← ZMIANA: ImageCropper.__init__ akceptuje już output_dir,
│                                          ale trzeba upewnić się że ścieżki są pod DATA_ROOT
├─ bounding_box_manager.py               ← DEPRECATED dla web (stan w frontend); zostawić dla desktop
├─ input_handler.py                      ← LEGACY desktop, nieużywany w web
├─ image_window.py                       ← LEGACY desktop, nieużywany w web
├─ main.py                               ← LEGACY desktop entry point — zostaje jako alternative CLI
├─ YOLO/weights/best.pt                  ← model używany w runtime
├─ requirements.txt                      ← NOWE: deps webowe + ML
├─ Dockerfile                            ← NOWE
└─ docker-compose.yml                    ← NOWE (przykładowy)
```

**Decyzja przeniesienia:** trzymamy desktop i web obok siebie. Frontent web nie używa `image_window.py`, `input_handler.py`, `bounding_box_manager.py`, `main.py`. Te pliki pozostają dla developerów którzy wolą uruchamiać lokalnie. Po ustabilizowaniu wersji web można je usunąć osobnym ticketem.

---

## API endpoints

Wszystkie ścieżki w body/query są **relatywne** do `DATA_ROOT`. Backend przepuszcza je przez `safe_resolve()`.

### `GET /api/fs/list?path=<rel>`
- Zwraca podkatalogi i pliki obrazowe (`.png`, `.jpg`, `.jpeg`) w danym katalogu.
- Response:
  ```json
  { "path": "subdir/x", "parent": "subdir",
    "dirs": [{"name":"a"},{"name":"b"}],
    "images": [{"name":"img1.jpg","size":12345}] }
  ```
- Walidacja: path traversal → 400; ścieżka spoza `DATA_ROOT` → 403.

### `POST /api/fs/mkdir`
- Body: `{ "path": "rel/path/new_dir" }`
- `Path(target).mkdir(parents=True, exist_ok=True)` po `safe_resolve()`.
- Response: `{ "path": "rel/path/new_dir", "created": true }`.

### `GET /api/image/preview?path=<rel>`
- Ładuje obraz przez `ImageLoader` (wariant webowy, bez Tkinter), skaluje do `MAX_PREVIEW_PX` (env, default 1920), zwraca PNG.
- Nagłówki: `X-Original-Width`, `X-Original-Height`, `X-Scale` — frontend musi je odczytać, żeby wiedzieć jak przeliczyć współrzędne podglądu na oryginał przy crop.

### `GET /api/image/original?path=<rel>`
- (opcjonalny, do debugowania) zwraca oryginał bez skalowania.

### `POST /api/detect`
- Body: `{ "path": "rel/img.jpg" }`
- Ładuje oryginał przez `ImageLoader`, wywołuje `AutoDetector.detect()` (model trzymany w app state od startup).
- **WAŻNE:** zwraca boxy w **podglądowych** współrzędnych (przeliczonych przez `scale`) — frontend trzyma te liczby, nie martwi się skalą.
- Response: `{ "boxes": [{"x1":12,"y1":34,"x2":56,"y2":78,"label":"auto"}, ...], "scale": 0.5 }`
- Boxy są **bez** przypisania do wierszy — frontend dorysuje linie i sam wyznaczy compartmenty przez port JS `compute_row_labels`.

### `POST /api/crop`
- Body:
  ```json
  {
    "image_path": "rel/img.jpg",
    "output_dir": "rel/out",
    "scale": 0.5,
    "rows": [
      { "line": {"p1":[100,200],"p2":[800,210]},
        "boxes": [{"x1":..,"y1":..,"x2":..,"y2":..}, ...],
        "compartment_override": null
      }
    ],
    "save_annotations": true
  }
  ```
- Backend:
  1. `safe_resolve(image_path)` + `safe_resolve(output_dir)`.
  2. Ładuje oryginał przez `ImageLoader.get_original_image()`.
  3. Tworzy obiekty `BoundingBox` i `RowDetector.Row` z payloadu, przepisuje `compartment_override` na każdy `Row`.
  4. **Walidacja** przez `compute_row_labels(rows)` — jeśli error (>6 wierszy lub >3 w wycinku) → HTTP 400 z komunikatem.
  5. Woła `ImageCropper(output_dir=resolved_output).crop_and_save(...)` (przekazując `image_loader` ze skalą, żeby `scale_coords_to_original` działało).
  6. Jeśli `save_annotations` (default `true`) → `_save_compartment_annotations(original_image, rows)` zapisuje `.txt` **w katalogu zdjęcia źródłowego** (decyzja userska B: razem z obrazem dla LabelImg/CVAT compatibility).
- Response:
  ```json
  {
    "saved": [{"filename":"img_A_2_1.png","row_index":1,"box_index":1}, ...],
    "annotations_path": "rel/img.txt",
    "compartments": {"A": 2, "B": 3}
  }
  ```
- Error response (HTTP 400): `{ "error": "Wycinek A ma 4 wierszy (max 3 na wycinek)..." }`.

---

## Modyfikacje istniejących plików (zachowanie API JAK NAJBLIŻEJ obecnego)

### `image_loader.py`
- Usunąć `get_screen_size()` (Tkinter) i `_SCREEN_SIZE`.
- Konstruktor `ImageLoader(image_dir, max_preview_px=1920)`: zamiast 90% ekranu skala do `max_preview_px`.
- Zachować `load_image()`, `scale_coords_to_original()`, `get_original_image()` — używane przez `/api/detect` i `/api/crop` bez zmian semantyki.
- W razie potrzeby dodać konstruktor alternatywny `from_path(image_path, max_preview_px)` żeby nie trzeba było tworzyć obiektu z katalogiem (web ładuje pojedynczy plik).

### `image_cropper.py`
- `ImageCropper(output_dir=..., image_loader=...)` — już parametryzowane. Sprawdzić tylko że `output_dir` może być absolutną ścieżką poza `cwd` (po `safe_resolve()` daje absolutną).
- Bez zmian logiki sortowania wierszy i konwencji `A_/B_`.
- **Re-use w endpointach** (już istnieją na `main`):
  - `compute_row_labels(rows)` — walidacja + numerowanie, wołane przez `/api/crop` przed crop_and_save.
  - `compute_compartment_bboxes(rows, labels=)` — używane przez `_save_compartment_annotations`.
  - `_save_compartment_annotations(original_image, rows)` — pseudo-labelling, wołane po crop. **Lokalizacja .txt**: `os.path.splitext(image_path)[0] + ".txt"` (obok zdjęcia źródłowego, decyzja userska B).

### `row_detector.py`
- `Row` ma już pole `compartment_override: Optional[str]` (None/'A'/'B'). Backend musi je propagować z payloadu `/api/crop` przy rekonstrukcji obiektów Row.

### YOLO model loading
- W `web/app.py` w `@app.on_event("startup")` jednorazowo ładujemy `AutoDetector(MODEL_PATH)` do `app.state.detector`. `/api/detect` korzysta z tej instancji.

---

## Frontend (`web/static/`)

### `index.html`
Trzy-kolumnowy layout:
- **Lewy panel (sidebar):** drzewo katalogów z breadcrumbami, przyciski "↻ odśwież", "📁 nowy katalog", lista obrazów w aktualnym katalogu (klikalne), pole "Wyjście:" z aktualnym katalogiem wyjściowym i przyciskiem zmiany.
- **Centrum:** `<canvas>` z obrazem + overlay boxy/linie. Pod canvasem status bar (tryb, plik, scale, liczba boxów, liczba wierszy).
- **Prawy panel / topbar:** przyciski trybów (Box / Line / Move / Resize / Delete), przyciski akcji "Auto-detect (YOLO)", "Wytnij i zapisz" (wysyła `/api/crop`).

### `app.js` (jeden plik, modularnie)
Klasy lustrzane do desktopowych:
- `BBox { x1, y1, x2, y2, id, label }` — z metodami `contains`, `getNearestCorner`, `resizeCorner`, `move`.
- `RowLine { p1, p2, id }` — `move`, `distanceTo`.
- `Row { line, boxes, compartmentOverride = null }` — `intersectsBox` (port `_does_line_intersect_box` na JS, identyczna logika cross-product), `centerY()`.
- `ImageCanvas` — `render()`, event handlers `mousedown/mousemove/mouseup`, `keydown`, integracja z `Modes` enum (`ADD_BOX`, `ADD_LINE`, `MOVE`, `RESIZE`, `DELETE`, `EDIT_LABEL`).
- `FileBrowser` — `cdTo(path)`, `mkdir(name)`, `onSelectImage(callback)`, `onSelectOutputDir(callback)`.
- `Api` — `listDir`, `mkdir`, `previewUrl`, `detect`, `crop` (wrappers na `fetch`).

**Shared helpers (port 1:1 z Pythona):**
- `computeRowLabels(rows) -> { labels, error }` — port `compute_row_labels` z `image_cropper.py`. Sortowanie, walidacja, split A/B przez largest gap Y, numerowanie "od dołu". **Logika identyczna** — niedopuszczalne rozjazdy z backendem.
- `splitCompartments(sortedRows) -> { aRows, bRows }` — port `_split_compartments`, respektuje `row.compartmentOverride`.
- `splitAutoByLargestGap(rows) -> [autoA, autoB]` — port `_split_auto_by_largest_gap`.
- `computeCompartmentBboxes(rows, margin=40, labels=null) -> { A: bbox, B: bbox }` — port `compute_compartment_bboxes`.

**Edycja etykiety wiersza (port trybu EDIT_LABEL):**
- Klawisz `e` → `Modes.EDIT_LABEL`.
- Klik na linię → wykryj wiersz (port `get_line_at` z tolerancją 10 px) → otwórz custom modal z dropdown A/B/auto.
- Modal jest **HTML/CSS overlay** w stylu z `style.css` (nie używamy żadnego frameworka). Dropdown to `<select>` z A/B/auto.
- Po wyborze: ustaw `row.compartmentOverride = "A"/"B"/null`, **wywołaj `computeRowLabels` ponownie** — jeśli error (>3 w wycinku), pokaż błąd w modalu i nie zapisuj override.
- Po sukcesie zamknij modal, `render()` aktualizuje etykiety i ramki.

Konwencja kolorów BGR z desktopa staje się RGB w JS:
- Czerwony box (poza wierszem) → `#FF0000`
- Zielony box (w wierszu) → `#00FF00`
- Żółta linia → `#FFFF00`
- **Niebieska ramka wycinka A** → `#0000FF` (BGR `(255,0,0)` na desktop)
- **Fioletowa ramka wycinka B** → `#C800C8` (BGR `(200,0,200)` na desktop)
- Wyznaczane podczas `render()` na podstawie przynależności do `Row` i wyniku `computeRowLabels`.

### Klawiszologia (dokładny port)
| Klawisz | Akcja |
|---------|-------|
| `b`     | ADD_BOX |
| `l`     | ADD_LINE |
| `d`     | DELETE |
| `v`     | MOVE |
| `r`     | RESIZE |
| `e`     | EDIT_LABEL (klik na linię → modal A/B/auto) |
| `Esc`   | reset zaznaczenia / zamknięcie modala |
| `Enter` | wywołanie `/api/crop` |

`n` (następny obraz) — w web zastąpione kliknięciem w sidebarze. Nie potrzeba klawiszologii do nawigacji plików.

### Rendering — co dokładnie rysuje `ImageCanvas.render()`

Kolejność warstw (od najniższej):
1. **Tło**: obraz preview z `/api/image/preview`.
2. **Boxy**: dla każdego `BBox` — prostokąt z kolorem zależnym od przynależności do `Row` (zielony jeśli w wierszu, czerwony jeśli nie). Wyznaczane z `computeRowLabels`.
3. **Linie wierszy**: żółte (`#FFFF00`).
4. **Ramki wycinków**: niebieska A i fioletowa B, prostokąty z `computeCompartmentBboxes`. Bez wypełnienia, grubość 2 px.
5. **Etykiety wycinków**: literowe "A"/"B" w lewym górnym rogu ramki (kolor ramki, biały outline 2 px).
6. **Etykiety wierszy**: tekst `A_1`, `A_2`, `B_3` przy `row.line.p1` z offsetem `+8, -20` px (biały tekst, czarny outline 2 px).
7. **Statusbar**: pod canvasem lub w górnym pasku — `Tryb: ADD_LINE | Wycinek A: 2 wierszy | Wycinek B: 3 wierszy | Skala: 0.5`.
8. **Komunikat walidacji**: gdy `computeRowLabels` zwróci `error` — czerwony tekst w dolnej części canvasu zamiast etykiet (np. `"BŁĄD: 4 wierszy w wycinku A (max 3)"`). Cropowanie blokowane po stronie frontendu (przycisk `Enter` / "Wytnij" disabled).

---

## Bezpieczeństwo (path traversal)

`web/services/fs_browser.py`:
```python
from pathlib import Path
import os

def get_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "/data")).resolve()

def safe_resolve(rel_path: str) -> Path:
    root = get_root()
    target = (root / (rel_path or "")).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"Path '{rel_path}' wychodzi poza DATA_ROOT")
    return target
```

Każdy endpoint dotykający FS zaczyna od `safe_resolve(...)`. W FastAPI łapane przez `except PermissionError → HTTPException(403)`.

---

## Compartments + numerowanie + pseudo-labelling (port z desktop)

Cała ta sekcja opisuje funkcjonalności **już zaimplementowane w desktop** (na `main`) które wymagają portu do web. Zmiany są behawioralnie identyczne — backend re-use Python helpers, frontend ma JS port 1:1.

### Reguła numerowania (sztywna konwencja)
- Najniższy wiersz w wycinku zawsze ma numer `_3`. Każdy kolejny w górę numer o 1 mniejszy:
  - 1 wiersz → `_3`
  - 2 wiersze → `_2`, `_3`
  - 3 wiersze → `_1`, `_2`, `_3`
- Max 3 wiersze per wycinek (A i B osobno). Przekroczenie → blokada zapisu + komunikat.
- Max 6 wierszy globalnie. Przekroczenie → blokada zapisu + komunikat (osobny komunikat niż per-wycinek).

### Identyfikacja A/B
- **Default**: largest-gap Y między wierszami (port `_split_compartments` + `_split_auto_by_largest_gap`).
- **Manual override**: per wiersz `compartment_override: 'A'|'B'|null`. Wiersze z override trafiają do swojego wycinka niezależnie od geometrii. Wiersze auto (override=null) dzielone przez largest gap między sobą.
- Override edytowany w UI przez tryb `EDIT_LABEL` (klawisz `e`) → klik na linię → modal A/B/auto.
- Walidacja przy próbie ustawienia: jeśli override naruszy max 3 per wycinek → blokada + komunikat, override nie zapisuje się.

### Bbox wycinka
- **Computed only** — bounding-box wszystkich boxów w wierszach wycinka + margines (default 40 px).
- Bez manualnej edycji ramki w UI w tej iteracji.
- Renderowany jako prostokąt (niebieski A / fioletowy B) z literowym labelem w lewym górnym rogu.

### Pseudo-labelling (active learning loop)
Każdy pomyślny crop produkuje **plik adnotacji YOLO** obok zdjęcia źródłowego — bez dodatkowej akcji użytkownika.

**Lokalizacja pliku**: katalog zdjęcia źródłowego, ta sama nazwa, rozszerzenie `.txt`.
- Przykład: `<DATA_ROOT>/test_images/FLE_NPZDR_2025_1.jpg` → `<DATA_ROOT>/test_images/FLE_NPZDR_2025_1.txt`.
- Decyzja userska B: razem z obrazem dla bezpośredniej kompatybilności z LabelImg/CVAT (bierzesz katalog jako dataset).

**Format pliku** (YOLO standard):
```
0 <cx> <cy> <w> <h>
0 <cx> <cy> <w> <h>
```
- Klasa `0` = `compartment` (jedna klasa, A/B z pozycji Y post-hoc przy treningu).
- `cx`, `cy`, `w`, `h` znormalizowane do 0–1 wymiarów **oryginalnego** obrazu (przeliczenie przez `scale_coords_to_original`).
- Po jednym wpisie per wycinek (typowo 2 linie: A i B).

**Toggle w UI**: checkbox "Zapisuj adnotacje" w toolbarze (domyślnie ON). Pozwala wyłączyć dla problematycznych zdjęć (active learning quality control).

**Trening (poza scope MVP)**: po nazbieraniu 50–100+ obrazów uruchamia się `YOLO/train_compartments.py` offline, wagi trafiają do `YOLO/weights/compartments.pt`. Plan szczegółowy w `audyty_plany/wycinki_i_numerowanie.md`.

### Shared helpers — re-use w backendzie, port w JS

| Helper | Plik desktop | Backend (Python) | Frontend (JS port) |
|--------|-------------|------------------|---------------------|
| Walidacja + numerowanie | `image_cropper.py:compute_row_labels` | re-use w `/api/crop` | `computeRowLabels` w `app.js` |
| Bbox wycinka | `image_cropper.py:compute_compartment_bboxes` | re-use w `_save_compartment_annotations` | `computeCompartmentBboxes` w `app.js` |
| Split A/B | `image_cropper.py:_split_compartments` | re-use | `splitCompartments` w `app.js` |
| Largest gap | `image_cropper.py:_split_auto_by_largest_gap` | re-use | `splitAutoByLargestGap` w `app.js` |
| Zapis YOLO | `image_cropper.py:_save_compartment_annotations` | re-use w `/api/crop` | brak — backend zapisuje |

**Krytyczne**: JS port musi być algorytmicznie identyczny z Pythonem. Najmniejsze rozjazdy (np. inny tie-break w sortowaniu) dadzą różne etykiety w UI vs nazwy plików.

---

## Docker

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

# OpenCV deps (libgl1 + libglib2 wystarczają dla opencv-python-headless)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Otolits_identyfication_program/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Otolits_identyfication_program /app/Otolits_identyfication_program

ENV DATA_ROOT=/data \
    MODEL_PATH=/app/Otolits_identyfication_program/YOLO/weights/best.pt \
    MAX_PREVIEW_PX=1920 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
WORKDIR /app/Otolits_identyfication_program
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt`
```
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart
opencv-python-headless>=4.9        # WAŻNE: headless wariant (brak GUI deps, mniejszy obraz)
numpy
Pillow
ultralytics>=8.1
exifread
```

> **Uwaga:** desktop używa `opencv-python` (z GUI). Web używa `opencv-python-headless` (bez GUI). Te paczki **konfliktują** — instalować tylko jedną. W kontenerze headless. Jeśli developer chce odpalać desktop lokalnie, robi to poza Dockerem na regular opencv.

### `docker-compose.yml` (przykład)
```yaml
services:
  turbot-web:
    build:
      context: .
      dockerfile: Otolits_identyfication_program/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./host-data:/data          # zmień na właściwą ścieżkę z obrazami
    environment:
      DATA_ROOT: /data
      MAX_PREVIEW_PX: 1920
    restart: unless-stopped
```

Model `best.pt` ląduje w obrazie (~50–200 MB). Jeśli chcesz osobno: dodaj kolejny volume `- ./weights:/weights` i ustaw `MODEL_PATH=/weights/best.pt`.

---

## Kolejność implementacji

**Backend (Python):**
1. **`requirements.txt`** w `Otolits_identyfication_program/`.
2. **`web/config.py`** — odczyt env vars z domyślnymi.
3. **`web/services/fs_browser.py`** — `safe_resolve`, `list_dir`, `make_dir`.
4. **`web/routers/fs.py`** — endpointy `/api/fs/list`, `/api/fs/mkdir`.
5. **`image_loader.py`** — usunięcie Tkinter (`get_screen_size`, `get_tk_root`), parametr `max_preview_px`.
6. **`web/services/image_service.py`** — wrapper na `ImageLoader.load_image_from_path(path)`.
7. **`web/routers/image.py`** — endpoint `/api/image/preview` (zwraca PNG + nagłówki).
8. **`web/routers/detect.py`** — endpoint `/api/detect` (re-use `AutoDetector`).
9. **`web/routers/crop.py`** — endpoint `/api/crop`:
   - rekonstrukcja `BoundingBox`/`Row` z JSON (włącznie z `compartment_override` per Row),
   - walidacja przez `compute_row_labels` (HTTP 400 przy błędach),
   - wywołanie `ImageCropper.crop_and_save` + `_save_compartment_annotations`,
   - response z listą plików + path do `.txt` + liczniki A/B.
10. **`web/app.py`** — FastAPI, mount static, mount routerów, `on_startup` load YOLO.

**Frontend (vanilla JS):**

11. **`web/static/index.html` + `style.css`** — layout 3-kolumnowy + modal CSS dla EDIT_LABEL.
12. **`web/static/app.js` — podstawowe klasy**: `BBox`, `RowLine`, `Row` (z `compartmentOverride`), `Api`, `FileBrowser`.
13. **`app.js` — port shared helpers**: `computeRowLabels`, `splitCompartments`, `splitAutoByLargestGap`, `computeCompartmentBboxes`. **Krytyczne**: porównaj wyniki z Pythonem na kilku scenariuszach (unit-test w przeglądarce / `assert`).
14. **`app.js` — `ImageCanvas`**: render warstw 1–4 (tło, boxy, linie, ramki wycinków), event handlers dla trybów `b/l/d/v/r/Esc/Enter`.
15. **`app.js` — etykiety i statusbar**: render warstw 5–7 (literowe A/B, etykiety A_1/B_3, statusbar).
16. **`app.js` — tryb EDIT_LABEL**: klawisz `e`, modal HTML/CSS, dropdown A/B/auto, walidacja `computeRowLabels` przed zapisem override.
17. **`app.js` — walidacja crop**: blokada przycisku "Wytnij" gdy error w `computeRowLabels`, komunikat na canvasie (warstwa 8).
18. **`app.js` — checkbox "Zapisuj adnotacje"** w toolbarze, propagacja do payloadu `/api/crop`.

**Deploy:**

19. **`Dockerfile` + `docker-compose.yml`**.
20. **Lokalny test bez Dockera:** `uvicorn web.app:app --reload --port 8000` z `DATA_ROOT=./test_images` ustawionym jako env.
21. **Test w Dockerze:** `docker compose up --build`, otwórz `http://localhost:8000`.

---

## Weryfikacja end-to-end

1. **Build i start:**
   ```
   docker compose up --build
   ```
   Logi pokazują "YOLO model loaded from /app/.../best.pt".
2. **UI ładuje się** na `http://localhost:8000` — drzewo `/data` widoczne w lewym sidebarze.
3. **Nawigacja katalogów:** klik w katalog → rozwija; klik `..` → wyżej; klik 📁 → modal z nazwą → mkdir; odśwież listy.
4. **Path traversal regression:**
   - `curl 'http://localhost:8000/api/fs/list?path=../../etc'` → HTTP 403.
   - `curl -X POST -d '{"path":"../foo"}' .../api/fs/mkdir` → HTTP 403.
5. **Klik na obraz** → canvas renderuje preview, X-Scale w nagłówku odczytany przez JS.
6. **Auto-detect:** klik "Auto-detect" → fetch `/api/detect` → boxy YOLO w czerwonym kolorze pojawiają się na canvas.
7. **Tryby ręczne:**
   - `l` + rysuj linię → boxy przecięte zmieniają kolor na zielony.
   - `b` + drag → nowy box.
   - `v` + drag box/linia → przesuwa się, re-assign do wiersza działa.
   - `r` + drag corner → resize.
   - `d` + klik → usuwa.
   - `Esc` → reset.
8. **Wyjście:** wybierz `output_dir` w sidebarze, klik "Wytnij i zapisz" → POST `/api/crop` → odpowiedź z listą plików → na hostcie w `host-data/<output_dir>/` pojawiają się PNG z nazwami `<oryginał>A_1_1.png`, ..., `B_1_1.png` zgodnie z konwencją (sortowanie top-bottom wierszy, lewo-prawo boxów).
9. **Pozytywny smoke:** zmień obraz, narysuj wiersze, wytnij — drugi zestaw plików w tym samym katalogu wyjściowym.
10. **Restart kontenera** — UI nadal działa, dane na volume przeżyły.

### Weryfikacja compartments + pseudo-labelling

11. **Etykiety wierszy w czasie rzeczywistym**: po narysowaniu wiersza etykieta `A_3` widoczna przy lewym końcu linii. Po dodaniu drugiego wiersza w tym samym wycinku → etykiety przeliczają się na `A_2` + `A_3`.
12. **Ramki wycinków**: po dodaniu wierszy w obu wycinkach (np. 2 nad luką, 3 pod) → niebieska ramka A wokół górnych boxów, fioletowa B wokół dolnych. Litery "A"/"B" w lewym górnym rogu każdej ramki.
13. **Statusbar liczników**: pokazuje `Wycinek A: N wierszy | Wycinek B: M wierszy` w czasie rzeczywistym.
14. **Tryb EDIT_LABEL**: wciśnij `e` → klik na linię → modal z dropdown A/B/auto. Wybierz `B` → etykieta wiersza zmienia się natychmiast (np. było `A_2` → po overridzie `B_3` jeśli to jedyny w nowym B). Pozostałe wiersze auto przeliczają się.
15. **Walidacja overflow**: dodaj 4 wiersze w jednym wycinku (lub forsuj 4× override 'A') → przycisk "Wytnij" disabled, czerwony komunikat na canvasie `"BŁĄD: 4 wierszy w wycinku A (max 3)"`. POST `/api/crop` zwraca HTTP 400.
16. **Walidacja global > 6**: dodaj 7 wierszy → analogiczny komunikat `"BŁĄD: wykryto 7 wierszy łącznie (max 6)"`.
17. **Pseudo-labelling**:
    - Po pomyślnym crop sprawdź że plik `.txt` powstał **w katalogu zdjęcia źródłowego** (`DATA_ROOT/test_images/<name>.txt`, nie w `output_dir/`).
    - Otwórz `.txt`: 1-2 linie w formacie `0 cx cy w h`, wartości w [0, 1].
    - Wyłącz checkbox "Zapisuj adnotacje" → crop nadal działa, ale `.txt` **nie powstaje**.
18. **Spójność JS ↔ Python**:
    - Sprawdź dla 5 wierszy (2 w A, 3 w B) że nazwy plików w response (`A_2_1.png`, `B_1_1.png` etc.) odpowiadają etykietom widocznym w UI **przed** kliknięciem "Wytnij".
    - Override prawym przyciskiem... err, klawiszem `e` na losowym wierszu → ponowny crop → nazwy plików zgodne z nowymi etykietami w UI.

---

## Co świadomie pomijamy w tej iteracji

- **Auth / multi-user** — single-user dev z decyzji usera.
- **Upload z dysku użytkownika** — tylko nawigacja po katalogach serwera.
- **WebSockety / live preview** — przegląd jest synchroniczny (fetch → render).
- **Undo/redo** — frontend stan może być rozszerzony o historię, ale nie w tej iteracji.
- **Edycja istniejących cropów** — out of scope (cropping jest jednokierunkowy: obraz → pliki w output_dir).
- **HTTPS / reverse proxy** — zakładamy że stoi za nginx/traefik dodanym osobno.
- **Usuwanie desktop legacy** (`image_window.py`, `input_handler.py`, `bounding_box_manager.py`, `main.py`) — zostaje obok web wersji, do osobnego cleanupu po ustabilizowaniu wersji webowej.
- **Walidacja `output_dir` non-empty** — jeśli user nie wybierze, blokujemy crop po stronie frontendu.
- **Auto-detekcja wierszy** (klastrowanie po Y / RANSAC) — wymagana ręczna detekcja linii w MVP. Auto-detekcja w osobnej iteracji (plan w `audyty_plany/wycinki_i_numerowanie.md` Etap B).
- **Trening modelu YOLO compartments** — pseudo-labelling produkuje training data, ale faktyczny trening uruchamiany offline poza scope MVP. Wymiana modelu z heurystyki na YOLO compartments w osobnej iteracji (`wycinki_i_numerowanie.md` Etapy E+F).
- **Manualna edycja bboxa wycinka** — bboxy są computed only z marginesem 40 px, bez resize'u w UI. Manual edit (drag krawędzi) w przyszłej iteracji.
- **Dropdown UX dialogu EDIT_LABEL po stronie desktop** — desktop ma teraz `simpledialog.askstring` (text input). Web będzie miał natywny dropdown w HTML/CSS od razu.