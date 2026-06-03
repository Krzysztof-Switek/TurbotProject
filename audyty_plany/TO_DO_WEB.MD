# TO_DO_WEB — stan sesji + plan dalszych kroków

> **Branch aktywny:** `web_service` (lokalnie + origin sync, 20 commitów ahead of main).
> **Stan po sesji (czerwiec 2026):** backend kompletny, frontend kompletny, deploy + testy zostały.

---

## Wykonane (kroki 1–18) ✅

### Backend (Python / FastAPI)

| # | Plik | Co zostało zaimplementowane |
|---|------|---|
| 1 | `Otolits_identyfication_program/requirements.txt` | fastapi, uvicorn[standard], python-multipart, opencv-python-headless, numpy, Pillow, ultralytics, exifread |
| 2 | `web/__init__.py` + `web/config.py` | env vars: DATA_ROOT, MODEL_PATH, MAX_PREVIEW_PX, ALLOWED_EXTS + `describe()` dla logów |
| 3 | `web/services/fs_browser.py` | `safe_resolve` (path traversal guard), `to_rel`, `list_dir`, `make_dir`, dataclasses `ListResult` |
| 4 | `web/routers/fs.py` | `GET /api/fs/list?path=`, `POST /api/fs/mkdir`, Pydantic schemas, error mapping (403/404/400/409) |
| 5 | `image_loader.py` refactor + `input_handler.py` | usunięty tkinter (`get_screen_size`/`get_tk_root`); dodany `max_preview_px` param + `ImageLoader.from_path()` classmethod; `_get_tk_root()` przeniesiony lokalnie do `input_handler` |
| 6 | `web/services/image_service.py` | `load(rel_path)` → `LoadedImage`, `encode_png(image)` → bytes, `get_preview_meta()` → dict (X-Original-Width/Height/Scale) |
| 7 | `web/routers/image.py` | `GET /api/image/preview?path=` zwraca PNG + nagłówki X-Scale (frontend przelicza coords) |
| 8 | `web/routers/detect.py` | `POST /api/detect` — re-use `AutoDetector` z `app.state`, detect na pełnej rozdzielczości, przeliczenie do przestrzeni preview (× scale) |
| 9 | `web/routers/crop.py` | `POST /api/crop` — rekonstrukcja `BoundingBox`/`Row` z JSON (z `compartment_override`), walidacja przez `compute_row_labels` (400 przy >6 wierszy / >3 per wycinek), `ImageCropper.crop_and_save`, opcjonalny `_save_compartment_annotations`, response z `compartments` (liczniki A/B) |
| 10 | `web/app.py` | FastAPI app + `lifespan` (startup YOLO load → `app.state.detector`), mount routerów `fs/image/detect/crop`, mount `web/static/` na `/` z `html=True` |

### Frontend (vanilla JS / HTML / CSS)

| # | Plik | Co zostało zaimplementowane |
|---|------|---|
| 11 | `web/static/index.html` + `style.css` | Layout 3-kolumnowy (sidebar 280px + main grid), dark theme, sidebar z breadcrumbs + dirs[] + images[] + output_dir, topbar z trybami + actions, canvas-wrap, statusbar; 2 modale (EDIT_LABEL + mkdir) |
| 12 | `web/static/app.js` — podstawy | `Modes` (enum), `BBox` / `RowLine` / `Row` (port klas domeny z `toJSON`), `Api` wrappery na `fetch` (`listDir`/`mkdir`/`previewUrl`/`detect`/`crop`) + `ApiError`, `FileBrowser` z nawigacją + mkdir prompt |
| 13 | `web/static/app.js` — shared helpers | **Port 1:1 z `image_cropper.py`**: `splitAutoByLargestGap`, `splitCompartments`, `computeRowLabels`, `computeCompartmentBboxes` + smoke testy 5 scenariuszy (3+3, 2+luka, 2 forced A, 4 forced A, 7 global) |
| 14 | `web/static/app.js` — ImageCanvas | Klasa z stanem (boxes/rows/mode/selection), `loadImage(path)` (fetch preview + nagłówki), event handlers per Mode (`b/l/d/v/r`), 4 warstwy render (tło, boxy, linie, ramki wycinków), `_reassignAllBoxesToRows` + `_findBoxAt/_findRowByLineAt`; bootstrap: mode buttons, auto-detect, crop, klawisze, KEY_TO_MODE map |
| 15 | `web/static/app.js` — etykiety + statusbar | Cache `lastLabels`/`lastCounts`/`lastError` w ImageCanvas; warstwa 5 (literowe "A"/"B" w rogu ramki, bold 18px), warstwa 6 (etykiety wierszy `A_1`/`B_3` przy `row.line.p1` z offsetem); statusbar `Wycinek A: N \| Wycinek B: M` + error |
| 16 | `web/static/app.js` — EDIT_LABEL modal | `openEditLabelModal(row)` w bootstrap, `<select>` A/B/auto, walidacja przez `computeRowLabels` przed zapisem (rollback gdy błąd, error w modalu, nie zamyka); wpięcie do `imageCanvas._openEditLabelModal` |
| 17 | `web/static/app.js` — walidacja crop | Warstwa 8 (czerwony komunikat na canvasie gdy `lastError`); `$btnCrop` disabled gdy brak image/output_dir/wierszy/walidacja failed; dynamiczny `title` tooltip; `FileBrowser.setOutputDir` wywołuje `onSelectOutputDir` → `updateActionButtons` |
| 18 | `web/static/app.js` + `index.html` | Checkbox "Zapisuj adnotacje" z **localStorage persistence** (`turbot.saveAnnotations`), rozszerzony tooltip z opisem YOLO format, 3-stanowy alert po crop (zapisane / pominięte bez bboxów / wyłączone) |

### Dokumentacja

- `audyty_plany/web_app.md` — zaktualizowany w 10 sekcjach o port compartments z desktop (commit `6df8ad3`).

---

## W trakcie 🟡

### Krok 20: lokalny test backendu

**Status:** dedykowany venv + deps gotowe; pozostało uruchomić uvicorn i przejść smoke test endpointów.

**Wykonane w kroku 20:**
- ✅ Sprawdzony Python 3.13.12 globalny.
- ✅ Stworzony `.venv-web/` (w `C:\Users\kswitek\Documents\TurbotProject\.venv-web`).
- ✅ Zainstalowane wszystkie zależności z `requirements.txt`: fastapi 0.136.3, uvicorn 0.48.0, python-multipart 0.0.30, opencv-python-headless 4.13.0.92, ultralytics 8.4.58, torch 2.12.0, numpy 2.4.6, Pillow 12.2.0, exifread 3.5.1.
- ⚠️ **Uwaga:** `ultralytics` ciągnie też `opencv-python` (4.13.0.92) — w venv są **obie wersje cv2** (regular + headless). W praktyce import `cv2` weźmie pierwszą znalezioną; może powodować subtelne problemy. Do sprawdzenia w teście.

**Do zrobienia w kroku 20:**
1. Uruchomić uvicorn: `cd Otolits_identyfication_program && DATA_ROOT=. ../.venv-web/Scripts/python.exe -m uvicorn web.app:app --port 8000` (lub z `--reload`).
2. Sprawdzić logi startup:
   - Czy `config.describe()` poprawnie pokazuje DATA_ROOT.
   - Czy YOLO model się załadował z `MODEL_PATH=YOLO/weights/best.pt`.
3. Smoke test endpointów (`curl`/przeglądarka):
   - `GET /docs` → Swagger UI (FastAPI auto-generuje).
   - `GET /api/fs/list?path=` → lista katalogów + obrazów w DATA_ROOT.
   - `GET /api/fs/list?path=test_images` → 10 JPGów (`FLE_NPZDR_2025_*.jpg`).
   - `GET /api/image/preview?path=test_images/FLE_NPZDR_2025_1.jpg` → PNG + nagłówki X-Scale.
   - `POST /api/detect` z `{"path":"test_images/FLE_NPZDR_2025_1.jpg"}` → lista boxów.
   - Path traversal test: `GET /api/fs/list?path=../../etc` → 403.
4. Otworzyć `http://localhost:8000/` w przeglądarce:
   - Sprawdzić czy frontend ładuje się.
   - Konsola: smoke testy `[smoke] computeRowLabels OK (5/5)`.
   - Nawigacja do `test_images/`, klik na obraz → render na canvasie.
   - Auto-detect → boxy YOLO.
   - Rysuj linię `l` → wiersze + etykiety + ramki.
   - `e` + klik na linię → modal A/B/auto.
   - Ustaw output_dir w sidebarze, Enter → crop, sprawdź pliki PNG + `.txt`.
5. Naprawić ewentualne bugi które wyjdą podczas testów (typowo: typos w endpointach, źle obsłużone błędy, Pydantic schema mismatch).

---

## Do zrobienia 📋

### Krok 19: `Dockerfile` + `docker-compose.yml`

**Co trzeba:**
- `Otolits_identyfication_program/Dockerfile`:
  - FROM `python:3.11-slim` (lub 3.13 jeśli stable).
  - `apt-get install libgl1 libglib2.0-0` (dla OpenCV).
  - `pip install -r requirements.txt`.
  - COPY kodu + ENV (`DATA_ROOT=/data`, `MODEL_PATH=/app/.../best.pt`, `MAX_PREVIEW_PX=1920`).
  - `CMD uvicorn web.app:app --host 0.0.0.0 --port 8000`.
- `Otolits_identyfication_program/docker-compose.yml` (lub w root):
  - Service `turbot-web`, port `8000:8000`, volume `./host-data:/data`.
- ⚠️ Sprawdzić czy model `best.pt` ląduje w obrazie (~50–200 MB) — może wartownej rozmiarowo opcja z osobnym volumem.

### Krok 21: test w Dockerze

**Co trzeba:**
- `docker compose up --build`.
- Sprawdzić logi startup (jak w kroku 20).
- Otworzyć `http://localhost:8000/` z host machine.
- Wszystkie testy E2E z planu `web_app.md` (sekcja "Weryfikacja end-to-end" + "Weryfikacja compartments + pseudo-labelling"):
  1. UI ładuje się, drzewo `/data` widoczne.
  2. Nawigacja katalogów + mkdir.
  3. Path traversal regresja (403).
  4. Klik obraz → render.
  5. Auto-detect → boxy YOLO.
  6. Tryby ręczne `l/b/v/r/d`.
  7. Tryb `e` (EDIT_LABEL) + modal.
  8. Crop → pliki PNG + `.txt` adnotacje.
  9. Walidacja overflow (>6, >3 per wycinek).
  10. Pseudo-labelling: plik `.txt` obok zdjęcia źródłowego, format YOLO.
  11. Spójność JS ↔ Python: etykiety w UI = nazwy plików w crop response.

---

## Decyzje projektowe (do zapamiętania)

1. **Single-user dev**, bez auth (server za reverse proxy / LAN).
2. **Źródło obrazów** = katalogi serwerowe (`DATA_ROOT`), nie upload. Nawigacja + mkdir przez UI.
3. **Frontend**: vanilla JS + HTML5 Canvas, zero build-step.
4. **Backend**: FastAPI + Uvicorn.
5. **Reguła "od dołu"**: najniższy wiersz = `_3`, kolejne w górę `_2`, `_1`. Max 3 wiersze/wycinek, max 6 globalnie.
6. **Identyfikacja A/B** = largest-gap Y + manual override przez `compartment_override`.
7. **Bbox wycinka** = computed only (boxy + margin 40 px), bez manualnej edycji.
8. **Pseudo-labelling** = `.txt` YOLO format **obok zdjęcia źródłowego** (decyzja userska B: dla LabelImg/CVAT compatibility).
9. **`opencv-python-headless`** w Dockerze (NIE regular). Desktop dev ma osobny venv z regular opencv.
10. **Push do origin**: SSL `schannel` (per-command override `git -c http.sslBackend=schannel push ...` — globalnego configu git nie zmieniamy).

---

## Stan repo

- **`main`** = `60d3aaf` Merge `web_app` (sync z origin).
- **`web_service`** = `ab7e749` krok 18, **20 commitów ahead** of main + 1 dokumentacyjny (web_app.md update).
- Stare branche (`web_app`, `row_detector_ransac`, `feature/standalone-windows-app`) usunięte z origin + lokalnie.
- `.venv-web/` w root projektu (gitignored) — pełen stack do uruchomienia web.

---

*Plik wygenerowany w trakcie sesji — aktualizuj w trakcie postępu kroków 19–21.*
