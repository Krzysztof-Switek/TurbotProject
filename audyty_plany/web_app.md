# Plan: konwersja TurbotProject na narzędzie webowe (FastAPI + vanilla JS + Docker)

## Kontekst

Obecnie `Otolits_identyfication_program/` to aplikacja desktop oparta na `cv2.namedWindow` + `cv2.waitKey` + `tkinter` (do pobrania rozmiaru ekranu). Cała interakcja jest lokalna; uruchomienie wymaga GUI, sterowniki ekranu, manualnego ustawienia `cwd` itd. User chce postawić to jako narzędzie webowe na serwerze, wdrażane Dockerem, z wieloma badaczami pracującymi przez przeglądarkę.

Cel: zachować logikę domenową (YOLO detekcja, geometria boxów/wierszy, sortowanie + cropping wg konwencji `A_/B_`) i wystawić ją przez HTTP API + frontend rysujący na HTML5 Canvas. Serwer ma być **w 90% bezstanowy**: stan UI (boxy/wiersze/tryb) trzyma frontend, serwer robi tylko I/O obrazów, detekcję YOLO i finalne cropowanie.

Decyzje uzgodnione z userem:
1. **Single-user dev** — brak auth, serwer za reverse proxy / w LAN.
2. **Źródło obrazów = katalogi na serwerze**, z możliwością nawigacji po drzewie i tworzeniem nowych katalogów (mkdir) z poziomu UI. **Nie ma uploadu z dysku** użytkownika.
3. **Frontend: vanilla JS + HTML5 Canvas** — zero build-step, jeden HTML serwowany przez backend.
4. **Backend: FastAPI + Uvicorn** — async, OpenAPI auto-docs.
5. **Root nawigacji**: jedna zmienna środowiskowa `DATA_ROOT` (domyślnie `/data` w kontenerze), jeden volume w Dockerze. Path traversal blokowany przez `Path.is_relative_to()`.

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

Stan UI (mode, boxy, wiersze, aktualnie zaznaczony element) **w pełni po stronie frontendu**. Serwer dostaje stan dopiero przy crop (jako payload JSON) i wykonuje cropping. Backend trzyma w pamięci tylko: wczytany model YOLO (od startu) i ewentualnie LRU cache wczytanych obrazów (opcjonalne).

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

### `POST /api/crop`
- Body:
  ```json
  {
    "image_path": "rel/img.jpg",
    "output_dir": "rel/out",
    "scale": 0.5,
    "rows": [
      { "line": {"p1":[100,200],"p2":[800,210]},
        "boxes": [{"x1":..,"y1":..,"x2":..,"y2":..}, ...] }
    ]
  }
  ```
- Backend:
  1. `safe_resolve(image_path)` + `safe_resolve(output_dir)`.
  2. Ładuje oryginał przez `ImageLoader.get_original_image()`.
  3. Tworzy obiekty `BoundingBox` i `RowDetector.Row` ze wszystkich wierszy z payloadu.
  4. Woła `ImageCropper(output_dir=resolved_output).crop_and_save(...)` (przekazując `image_loader` ze skalą, żeby `scale_coords_to_original` działało).
  5. Zwraca listę zapisanych plików.
- Response: `{ "saved": [{"filename":"img_A_1_1.png","row_index":1,"box_index":1}, ...] }`

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
- `Row { line, boxes }` — `intersectsBox` (port `_does_line_intersect_box` na JS, identyczna logika cross-product).
- `ImageCanvas` — `render()`, event handlers `mousedown/mousemove/mouseup`, `keydown`, integracja z `Modes` enum (`ADD_BOX`, `ADD_LINE`, `MOVE`, `RESIZE`, `DELETE`).
- `FileBrowser` — `cdTo(path)`, `mkdir(name)`, `onSelectImage(callback)`, `onSelectOutputDir(callback)`.
- `Api` — `listDir`, `mkdir`, `previewUrl`, `detect`, `crop` (wrappers na `fetch`).

Konwencja kolorów BGR z desktopa staje się RGB w JS:
- Czerwony box (poza wierszem) → `#FF0000`
- Zielony box (w wierszu) → `#00FF00`
- Żółta linia → `#FFFF00`
- Wyznaczane podczas `render()` na podstawie przynależności do `Row`.

### Klawiszologia (dokładny port)
| Klawisz | Akcja |
|---------|-------|
| `b`     | ADD_BOX |
| `l`     | ADD_LINE |
| `d`     | DELETE |
| `v`     | MOVE |
| `r`     | RESIZE |
| `Esc`   | reset zaznaczenia |
| `Enter` | wywołanie `/api/crop` |

`n` (następny obraz) — w web zastąpione kliknięciem w sidebarze. Nie potrzeba klawiszologii do nawigacji plików.

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

1. **`requirements.txt`** w `Otolits_identyfication_program/`.
2. **`web/config.py`** — odczyt env vars z domyślnymi.
3. **`web/services/fs_browser.py`** — `safe_resolve`, `list_dir`, `make_dir`.
4. **`web/routers/fs.py`** — endpointy `/api/fs/list`, `/api/fs/mkdir`.
5. **`image_loader.py`** — usunięcie Tkinter, parametr `max_preview_px`.
6. **`web/services/image_service.py`** — wrapper na `ImageLoader.load_image_from_path(path)`.
7. **`web/routers/image.py`** — endpoint `/api/image/preview` (zwraca PNG + nagłówki).
8. **`web/routers/detect.py`** — endpoint `/api/detect` (re-use `AutoDetector`).
9. **`web/routers/crop.py`** — endpoint `/api/crop` (rekonstrukcja `BoundingBox`/`Row` z JSON, wywołanie `ImageCropper`).
10. **`web/app.py`** — FastAPI, mount static, mount routerów, `on_startup` load YOLO.
11. **`web/static/index.html` + `style.css`** — layout.
12. **`web/static/app.js`** — port logiki z desktop na JS (canvas, eventy, klasy).
13. **`Dockerfile` + `docker-compose.yml`**.
14. **Lokalny test bez Dockera:** `uvicorn web.app:app --reload --port 8000` z `DATA_ROOT=./test_images` ustawionym jako env.
15. **Test w Dockerze:** `docker compose up --build`, otwórz `http://localhost:8000`.

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