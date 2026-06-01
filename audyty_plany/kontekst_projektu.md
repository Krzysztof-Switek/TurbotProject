# TurbotProject — Kontekst projektu (dokument referencyjny)

Dokument przygotowany do użycia jako kontekst w kolejnych sesjach. Opisuje całość kodu w `C:\Users\kswitek\Documents\TurbotProject`, pełne działanie programu, listę plików, funkcji oraz zależności między modułami. Ma ułatwiać szybkie zlokalizowanie fragmentu do modyfikacji.

> **Aktualizacja 2026-05-28:** dokument odzwierciedla stan po cleanupie wg planu `~/.claude/plans/twoim-celem-jest-teraz-nested-planet.md`. Zmiany podsumowane w sekcji 11.

---

## 1. Cel projektu

Aplikacja desktop (OpenCV + PIL + YOLO) do **identyfikacji i wycinania otolitów** z fotografii ryb. Użytkownik:

1. Ładuje obrazy z katalogu `test_images/`.
2. YOLO (`AutoDetector`) auto-wykrywa otolity → tworzy boxy (czerwone).
3. Użytkownik rysuje **linie wierszy** (`l`) — boxy przecięte linią są przypisane do wiersza i robią się zielone.
4. Można dodawać/przesuwać/usuwać/zmieniać rozmiar boxów i linii.
5. `Enter` → wycinanie zawartości boxów do `output_crops/` z nazwami typu `<oryginał>A_<rzad>_<box>.png` (wiersze 1–3) lub `B_<n>_<box>.png` (wiersze 4–6).
6. `n` → następny obraz.

Pomocniczy skrypt `Picks_modification_scripts/Resize.py` zmniejsza fotografie wsadowe poniżej 1 MB.

---

## 2. Struktura katalogów (top-level)

```
TurbotProject/
├─ Otolits_identyfication_program/    # główny program
│  ├─ main.py                         # entry point
│  ├─ image_loader.py                 # wczytywanie + skalowanie obrazów + EXIF
│  ├─ bounding_box.py                 # klasa BoundingBox (geometria, rysowanie)
│  ├─ bounding_box_manager.py         # kolekcja boxów (CRUD + cache)
│  ├─ row_detector.py                 # RowLine, Row, RowDetector
│  ├─ input_handler.py                # tryby pracy, obsługa klawiatury/myszy
│  ├─ image_window.py                 # główna pętla okna OpenCV + rendering
│  ├─ image_cropper.py                # wycinanie wg wierszy/kolumn → PNG
│  ├─ auto_detector.py                # adapter YOLO (ultralytics)
│  ├─ YOLO/                           # trening i wagi modelu (legacy/standalone)
│  │  ├─ config.yaml                  # ścieżki dataset + mapowanie klas (0=mark, 1=otolith)
│  │  ├─ train.py                     # uruchomienie treningu (yolov8n, 90 epok)
│  │  ├─ predict.py                   # standalone predykcja (legacy, ścieżka mtiurina)
│  │  ├─ predict_file.py
│  │  ├─ yolo_trainer.py
│  │  ├─ weights/best.pt              # docelowy model używany w runtime
│  │  └─ yolo11l.pt / yolo11n.pt / yolo11x.pt / yolov8n.pt
│  ├─ tests/
│  │  ├─ bounding_box_manager_test.py
│  │  └─ image_loader_test.py
│  ├─ test_images/                    # wejściowe obrazy (FLE_NPZDR_2025_*.jpg)
│  └─ output_crops/                   # wyjście — wycięte otolity
├─ Picks_modification_scripts/
│  └─ Resize.py                       # batch-resize do <1 MB JPG
├─ audyty_plany/                      # ten dokument
├─ runs/                              # artefakty treningu YOLO
├─ pyvenv.cfg                         # UWAGA: wskazuje na nieistniejący Python 3.10
└─ README.md                          # tylko nagłówek "Turbot"
```

> Usunięto w cleanupie: `row_editor.py` (pusty), `row_manager.py` (szkielet), `gui.py` (szkielet).

Aktualna gałąź git: `row_detector_ransac`. Branża główna: `main`.

---

## 3. Główny przepływ wykonania

Sekwencja od uruchomienia `main.py` aż do wyjścia:

```
main.py
 └─ ImageLoader(image_dir="test_images")
 └─ first_image = loader.load_image()            # skalowanie do 90% ekranu
 └─ BoundingBoxManager()
 └─ InputHandler(bbox_manager, None)             # row_detector zostanie nadpisany przez ImageWindow
 └─ AutoDetector(model_path="yolo/weights/best.pt")
 │    └─ jeśli model istnieje: detect() → bbox_manager.add_box(... label="auto")
 └─ ImageWindow(image_loader, bbox_manager, input_handler, auto_detector).show_image()
       │
       ├─ __init__ tworzy RowDetector(bbox_manager) i wpina go w input_handler.row_detector
       ├─ cv2.namedWindow + setMouseCallback(_handle_mouse_event)
       └─ pętla while True:
              update_display()
              key = cv2.waitKey(1)
              if key == 'q' / okno zamknięte → break
              if key == Enter → _handle_crop_boxes() → ImageCropper.process_cropping
              if key == 'n'   → _handle_next_image()
              else            → input_handler.keyboard_callback(key)
```

Każda zmiana stanu ustawia `self.dirty = True` w `ImageWindow`, co wymusza re-render w następnej iteracji.

**Ścieżka modelu YOLO:** `main.py` przekazuje `'yolo/weights/best.pt'`. Rzeczywiste wagi leżą w `Otolits_identyfication_program/YOLO/weights/best.pt`. Program musi być uruchamiany z katalogu `Otolits_identyfication_program/` (cwd), a ścieżka jest case-insensitive na Windows — działa, ale jest krucha (potencjalne miejsce do refactoru).

---

## 4. Tryby pracy i sterowanie

Definicje w `input_handler.py`:

- `WorkMode`: `AUTO` (mysz wyłączona), `MANUAL` (aktywna interakcja)
- `ManualMode`: `ADD_BOX`, `ADD_LINE`, `DELETE`, `MOVE`, `RESIZE`

Klawiszologia (`InputHandler.key_bindings`):

| Klawisz | Akcja |
|---------|-------|
| `a`     | tryb AUTO |
| `m`     | tryb MANUAL |
| `b`     | MANUAL → ADD_BOX |
| `l`     | MANUAL → ADD_LINE |
| `d`     | MANUAL → DELETE |
| `v`     | MANUAL → MOVE |
| `r`     | MANUAL → RESIZE |
| `Esc`   | reset zaznaczenia |
| `Enter` | wycięcie boxów i zapis PNG (obsługa w `ImageWindow`) |
| `n`     | następne zdjęcie (obsługa w `ImageWindow`) |
| `q`     | wyjście (obsługa w `ImageWindow`) |

W `ImageWindow.show_image()` klawisz jest konwertowany przez `chr(key).lower()` → wielkie i małe litery działają tak samo.

---

## 5. Pliki — szczegółowy spis funkcji

### 5.1 `main.py`
- skrypt entry-pointowy, brak funkcji top-level
- importuje: `ImageLoader`, `BoundingBoxManager`, `ImageWindow`, `InputHandler`, `AutoDetector` (już NIE importuje `RowDetector` — tworzy go `ImageWindow`)
- inicjalizuje komponenty, robi pierwsze auto-detect, uruchamia `ImageWindow`
- `InputHandler` dostaje `row_detector=None`, faktyczna instancja powstaje w `ImageWindow.__init__`

### 5.2 `image_loader.py`
- `get_screen_size() -> (w, h)` — cache rozmiaru ekranu przez Tkinter
- `class ImageLoader`
  - `__init__(image_dir)` — sortowana lista plików `.png/.jpg/.jpeg`
  - `load_image() -> np.ndarray` — walidacja (rozmiar pliku ≤100 MB, wymiary ≤50k×50k), `cv2.imread`, korekta EXIF dla JPG, skalowanie do 90% ekranu
  - `next_image() -> np.ndarray` — `current_index += 1` → `load_image()`
  - `_resize_to_screen(img)` — INTER_AREA przy zmniejszaniu, INTER_LINEAR przy powiększaniu, ale `scale = min(..., 1.0)` blokuje powiększanie
  - `_get_exif_orientation() -> int` — odczyt przez `exifread`
  - `_apply_orientation(img, orient)` — obroty 90/180/270
  - `get_original_image(copy=True)` — pełna rozdzielczość do croppingu
  - `scale_coords_to_original(x1,y1,x2,y2)` — transformacja podglądu → oryginał (używana przez `ImageCropper`)
  - `clear()`, `current_image_path` (property)
- moduł-level: `_SCREEN_SIZE` (cache)

### 5.3 `bounding_box.py`
- `class BoundingBox`
  - `__init__(x1,y1,x2,y2,label=None,is_temp=False)` — `is_temp=True` pomija walidację (potrzebne przy rysowaniu)
  - `_validate_coordinates()`, `_normalize_coords()`, `_invalidate_cache()`
  - `move(dx, dy)`
  - `resize_corner(corner_idx, x, y)` — czytelny `if/elif`: 0=LU, 1=PG (prawy górny), 2=LD (lewy dolny), 3=PD
  - `contains(x, y, tolerance=0)` — point-in-box
  - `intersects(other)`, `distance_to(x,y)`
  - `get_center()`, `get_corners()` (cache), `get_nearest_corner(x, y)`
  - `area()`, `width()`, `height()`, `aspect_ratio()`
  - `draw(image, color=None, thickness=1)` — selected → czerwony
  - `to_dict()`, `from_dict()` (class method) — serializacja
  - `copy()`, `release()`, `__str__`, `__repr__`
- pole `self.color = (0, 0, 255)` — domyślny czerwony BGR; finalny kolor wyznacza `ImageWindow.update_display` na podstawie przynależności do wiersza (zielony `(0, 255, 0)` w wierszu, czerwony `(0, 0, 255)` poza)

### 5.4 `bounding_box_manager.py`
- `class BoundingBoxManager`
  - `boxes: List[BoundingBox]`, `_boxes_cache`
  - `add_box(x1_or_box, y1=None, x2=None, y2=None, label=None)` — overload: akceptuje `BoundingBox` LUB cztery liczby
  - `remove_box(box) -> bool`
  - `update_box(box, x1, y1, x2, y2)` — bezpośrednie przypisanie współrzędnych + `box._invalidate_cache()` + `invalidate_cache()`. (Wcześniej wołał nieistniejące `box.update(...)`.)
  - `get_boxes()` (cache)
  - `get_box_at(x, y, tolerance=5.0)` — iteracja `reversed` (od najnowszych)
  - `clear_all()` — `gc.collect()`
  - `to_list() / from_list()` — serializacja
  - `invalidate_cache()`, `__del__`

### 5.5 `row_detector.py`
- `@dataclass RowLine`
  - pola: `p1`, `p2`, `id`, `color=(0,255,255)` (żółty BGR), `thickness=1`
  - `move(dx, dy)`
- `class RowDetector`
  - zagnieżdżony `@dataclass Row`
    - pola: `id`, `line: RowLine`, `boxes: List[BoundingBox]`
    - `add_box(box)` — tylko gdy linia przecina box; sortuje od lewej do prawej po środku boxa
    - `remove_box(box)`
    - `_does_line_intersect_box(line, box)` — sprawdza 4 krawędzie + zawieranie punktu
    - `_line_segments_intersect(p1,p2,q1,q2)` — klasyczna metoda cross-product
    - `_point_in_box(point, box)`
  - `__init__(bbox_manager)`
  - `start_new_line(x, y)` — tworzy `RowLine` z `p1=p2`
  - `update_line_end(x, y)` — przy ruchu myszy w ADD_LINE
  - `finish_line()` — minimalna długość 10 px, w przeciwnym razie odrzucone; potem `_assign_boxes_to_line()`
  - `_reset_drawing_state()`
  - `draw_rows(image)` — wszystkie ukończone linie + aktualnie rysowana (LINE_AA)
  - `clear_rows()`
  - `remove_line(line)` — uproszczone (brak zbędnego `try/except`)
  - `get_line_at(x, y, tolerance=10.0)` — sprawdza dystans od p1, p2 i od odcinka
  - `_assign_boxes_to_line()` — filtr bounding-rect (margin 50 px) → dokładne `_does_line_intersect_box`; pomija boxy już przypisane do innych wierszy. **Nie ustawia już koloru boxa** (kolor wyznacza render).
  - `_distance_to_line(p1, p2, point)` — wzór punkt-prosta

### 5.6 `input_handler.py`
- `WorkMode`, `ManualMode` (Enum)
- `@dataclass SelectionContext` — `element`, `drag_start`, `is_drawing`, `corner_idx` (int dla boxów, `'p1'`/`'p2'` dla linii)
- `class InputHandler`
  - `__init__(bbox_manager, row_detector)` — domyślnie `MANUAL` + `ADD_LINE`. Akceptuje `row_detector=None` (faktyczna instancja wpinana przez `ImageWindow.__init__`).
  - `get_mode_info() -> str`, `get_key_bindings_info() -> dict` — używa `base_keys.copy()` żeby nie mutować lokalu (bez zmiany działania)
  - `keyboard_callback(key)` — zwraca True jeśli klawisz znany
  - `mouse_callback(event, x, y)` — dispatch do `_handle_left_down/_handle_mouse_move/_handle_left_up`
  - `_handle_left_down(x, y)` — różne ścieżki dla każdego `ManualMode` (po cleanupie usunięty redundantny `isinstance(line, RowLine)`)
  - `_handle_mouse_move(x, y)` — rysowanie temp_boxa / aktualizacja linii / przeciąganie elementu; po edycji RowLine wywołuje `_update_row_boxes(row)` dla każdego wiersza
  - `_handle_left_up(x, y)` — finalizacja: dodanie boxa do managera (po normalizacji `min/max`) albo `row_detector.finish_line()`
  - `_reset_selection()`, `_set_work_mode(mode)`, `_set_manual_mode(mode)`, `reset_to_defaults()`
  - `_update_row_boxes(row)` — przelicza przynależność boxów do wiersza po przesunięciu/resize linii. **Nie ustawia już kolorów** (kolor wyznacza render).

### 5.7 `image_window.py`
- importuje też `WorkMode`, `ManualMode` z `input_handler` (po cleanupie — wcześniej crash na `n` przy aktywnym YOLO)
- `class ImageWindow`
  - `__init__(image_loader, bbox_manager, input_handler, auto_detector=None)` — tworzy `ImageCropper(image_loader=...)`, **przypisuje** `input_handler.row_detector = RowDetector(bbox_manager)` (jedyna inicjalizacja — `main.py` już go nie tworzy)
  - `_prepare_display_image()` — konwersja koloru, ograniczenie cache do 5 obrazów (`_release_resources()`)
  - `_release_resources()` — `self._cached_images.clear()` + `gc.collect()` (usunięta no-op pętla)
  - `update_display()` — rysuje wszystkie boxy (zielony jeśli w wierszu, czerwony jeśli nie — kolor wyliczany tu, nie cache'owany w boxie), `temp_box`, linie (`row_detector.draw_rows`), nakładka tekstu (mode info, key bindings) przez PIL
  - `show_image()` — główna pętla; `cv2.waitKey(1)`; obsługuje `q`, `Enter`, `n` osobno, resztę przez `input_handler.keyboard_callback`
  - `mark_dirty()`
  - `_handle_mouse_event(event, x, y, flags, param)` — wrapper na `input_handler.mouse_callback`
  - `_handle_next_image()` — czyści boxy/wiersze, ładuje następny obraz, ponawia detekcję YOLO; po cleanupie używa `_set_work_mode(WorkMode.MANUAL)` i `_set_manual_mode(ManualMode.ADD_LINE)` (zamiast nieistniejących `set_*`)
  - `_handle_crop_boxes()` — wywołuje `image_cropper.process_cropping(...)`
  - `_cleanup()` — w `finally`, niszczy okna, czyści wiersze/boxy

### 5.8 `image_cropper.py`
- `@dataclass CropResult` — `image`, `box_index`, `row_index`, `original_coords`, `filename`
- `class ImageCropper`
  - `__init__(output_dir="output_crops", image_loader=None)` — tworzy katalog wyjściowy
  - `crop_and_save(original_image, rows, boxes) -> List[CropResult]`
    - sortuje wiersze top→bottom po `min(b.y1 for b in row.boxes)`
    - **konwencja nazewnictwa:** wiersze 1–3 dostają prefix `A_<i>`, wiersze 4–6 dostają `B_<i-3>`
    - sortuje boxy w wierszu lewo→prawo po środku
    - transformuje współrzędne z podglądu na oryginał przez `image_loader.scale_coords_to_original(...)`
    - clipping do granic obrazu, pomija puste/odwrócone boxy
    - zapis `cv2.imwrite(<output_dir>/<original_filename><prefix>_<box_idx>.png, cropped)`
  - `process_cropping(bbox_manager, input_handler)` — pobiera oryginał, woła `crop_and_save` z `input_handler.row_detector.rows` i `bbox_manager.boxes`, loguje listę plików
- `TYPE_CHECKING` zawiera: `ImageLoader`, `RowLine`, `BoundingBox` (z `bounding_box`, nie `bounding_box_manager`), `BoundingBoxManager`, `InputHandler`

### 5.9 `auto_detector.py`
- `class AutoDetector`
  - `__init__(model_path='yolo/weights/best.pt')` — `YOLO(...)` jeśli plik istnieje, inaczej `self.model = None`
  - `conf_threshold = 0.2`, `iou_threshold = 0.6`
  - `detect(image) -> List[(x1,y1,x2,y2)]`
  - `_filter_detections(boxes_data)` — bierze tylko klasę 1 (otolit), próg pewności 0.2
  - `_remove_overlapping_boxes(boxes)` — kasuje nakładające się (IoU > 0.6), zostawia większe
  - `_calculate_iou(box1, box2)` (static), `_box_area(box)` (static)

### 5.10 `YOLO/` (trening)
- `train.py` — `YOLO("yolov8n.yaml")` + `model.train(data="config.yaml", epochs=90)`
- `config.yaml` — dataset path (twardo zakodowany na `mtiurina`), klasy `0: mark`, `1: otolith`
- `predict.py` — standalone batch predykcja (legacy z hardkodami `mtiurina`)
- `predict_file.py`, `yolo_trainer.py`
- `weights/best.pt` — używane przez runtime

### 5.11 `Picks_modification_scripts/Resize.py`
- `resize_image(input, output)` — pętla z malejącą jakością JPG (–5) i scale × 0.95 aż plik <1 MB lub quality < 30
- `process_images()` — przechodzi przez `INPUT_DIR = C:\Users\kswitek\Documents\Turbot\TUR_images`, zapisuje do `TUR_resized/`

### 5.12 `tests/`
- `bounding_box_manager_test.py`, `image_loader_test.py` — testy jednostkowe (do uruchomienia osobno, nieobjęte runtimem)

---

## 6. Mapa zależności (importy)

```
main.py
 ├─ image_loader.ImageLoader
 ├─ bounding_box_manager.BoundingBoxManager
 ├─ image_window.ImageWindow
 ├─ input_handler.InputHandler
 └─ auto_detector.AutoDetector
 (NIE importuje już row_detector — tworzy go ImageWindow)

image_loader.py        → os, cv2, numpy, exifread, tkinter (lazy)
bounding_box.py        → uuid, math, cv2, numpy
bounding_box_manager.py→ numpy, gc, bounding_box.BoundingBox
row_detector.py        → numpy, cv2, dataclasses, uuid, math, bounding_box.BoundingBox
input_handler.py       → enum, cv2, dataclasses, math.hypot, row_detector.RowLine, bounding_box.BoundingBox
image_window.py        → cv2, gc, traceback, row_detector.RowDetector, image_cropper.ImageCropper,
                         input_handler.WorkMode + ManualMode, PIL (Image/ImageDraw/ImageFont), numpy
image_cropper.py       → os, cv2, numpy, dataclasses (TYPE_CHECKING: image_loader.ImageLoader,
                         row_detector.RowLine, bounding_box.BoundingBox,
                         bounding_box_manager.BoundingBoxManager, input_handler.InputHandler)
auto_detector.py       → os, ultralytics.YOLO
```

Zależności **runtime → externals**: `opencv-python`, `numpy`, `Pillow`, `ultralytics`, `exifread`, `tkinter` (stdlib).

---

## 7. Kolory boxów (BGR)

Po cleanupie **kolory są wyznaczane wyłącznie podczas renderu** w `image_window.update_display`. `box.color` (default `(0, 0, 255)`) nie jest już mutowany w RowDetector/InputHandler.

| Stan          | Kolor          | Gdzie ustawiany |
|---------------|----------------|-----------------|
| Box w wierszu | (0, 255, 0)    | `image_window.update_display` (per frame, na podstawie `in_row`) |
| Box poza wierszem | (0, 0, 255) | `image_window.update_display` (per frame) |
| Linia (RowLine)| (0, 255, 255) | `RowLine` dataclass default |
| Box `selected` | (0, 0, 255)   | `BoundingBox.draw` (jeśli `self.selected`) |

---

## 8. Konwencja nazw plików wyjściowych

`<oryginalna_nazwa><prefix>_<box_idx>.png`, gdzie:

- `<prefix>` = `A_1`, `A_2`, `A_3` dla wierszy 1–3 (sortowanie top→bottom po `min y1` boxów)
- `<prefix>` = `B_1`, `B_2`, `B_3` dla wierszy 4–6
- `<box_idx>` zaczyna się od 1, sortowanie lewo→prawo po środku boxa

Pliki w `output_crops/`. Stare crops (`TUR_BITS_2015_*`) są w `git status` jako `D` (usunięte z indeksu).

---

## 9. Znane luki i miejsca do poprawy (po cleanupie)

> Sekcja po cleanupie. Listę wcześniejszych bugów-blokerów (crash po `n`, `box.update`, podwójna inicjalizacja RowDetector, martwy kod kolorów, pliki-szkielety) — patrz sekcja 11.

1. **`config.yaml` ma twardą ścieżkę użytkownika `mtiurina`** — nieprzenośne między maszynami. Świadomie zostawione (legacy/standalone).
2. **`predict.py` w `YOLO/`** zawiera stary kod legacy z hardkodami `C:\Users\mtiurina\...`. Świadomie zostawione (nieużywane w runtime).
3. **Ścieżka modelu YOLO** `'yolo/weights/best.pt'` (małe `yolo`) vs katalog `YOLO/` — case-insensitive na Windows, ale na Linux/CI by się sypnęło.
4. **`image_window.py:show_image` woła `mark_dirty()` w każdej iteracji** — flaga `dirty` jest dziś zawsze `True`. Granica behawioralnej neutralności (usunięcie zmieniłoby rytm renderowania), więc zostawione.
5. **`BoundingBox.from_dict`** nie odtwarza `id` z dict (round-trip generuje nowe id). Serializacja nieużywana w runtime.
6. **`image_loader.py:clear`** robi `del self.scale` zamiast `self.scale = 1.0` — niegroźne, ale niezgrabne.
7. **`bounding_box.py:aspect_ratio`** przy `height()==0`, **`auto_detector.detect`** przy pustej liście wyników — edge-case'y bez guard'ów; świadomie zostawione (zmieniłyby zachowanie brzegowe).
8. **`BoundingBoxManager.__del__` z `gc.collect()`** — typowa pułapka destruktorów (`gc` może być już ubity przy shutdown Pythona), ale w praktyce niegroźne.
9. **`.venv` w katalogu projektu wskazuje na nieistniejący Python 3.10** — środowisko zepsute, do odtworzenia osobno.
10. **`get_key_bindings_info` w InputHandler** używa już `base_keys.copy()` (cleanup), więc nawet gdyby ktoś trzymał referencję, nie zostanie zmutowana.

---

## 10. Szybka nawigacja — co edytować, gdy chcesz...

| Zadanie | Plik:funkcja |
|---------|-------------|
| Zmienić listę formatów wejściowych | `image_loader.py:ImageLoader.__init__` |
| Zmienić skalowanie do ekranu | `image_loader.py:_resize_to_screen` |
| Dodać nowy tryb manualny | `input_handler.py:ManualMode` + `key_bindings` + dispatch w `_handle_left_down/_handle_mouse_move/_handle_left_up` |
| Zmienić logikę przypisywania boxów do wiersza | `row_detector.py:_assign_boxes_to_line` i `_does_line_intersect_box` |
| Zmienić próg pewności YOLO | `auto_detector.py:AutoDetector.__init__` (`conf_threshold`, `iou_threshold`) |
| Zmienić konwencję nazw wycinków | `image_cropper.py:crop_and_save` (sekcja `prefix = ...`) |
| Zmienić sterowanie | `input_handler.py:key_bindings` + ew. obsługa specjalna w `image_window.py:show_image` (Enter/n/q) |
| Modyfikować render (tekst, kolory boxów) | `image_window.py:update_display` — to JEDYNE miejsce gdzie wyznacza się kolor boxa |
| Zmienić obsługę EXIF | `image_loader.py:_get_exif_orientation`, `_apply_orientation` |
| Trenować model | `YOLO/train.py` + `YOLO/config.yaml` |
| Batch-zmniejszanie obrazów wsadowych | `Picks_modification_scripts/Resize.py` |

---

## 11. Historia zmian — cleanup z 2026-05-28

Przeprowadzony cleanup zgodnie z planem `~/.claude/plans/twoim-celem-jest-teraz-nested-planet.md`. Zasada: zero zmiany działania użytkowego, naprawa rzeczywistych crashy + usunięcie martwego kodu.

**Naprawione crashe:**
- `image_window.py:_handle_next_image` — wywołania `set_work_mode`/`set_manual_mode` zamienione na `_set_work_mode`/`_set_manual_mode`; dodany import `WorkMode`, `ManualMode` z `input_handler`. Przedtem `AttributeError` po `n` przy aktywnym YOLO.
- `bounding_box_manager.py:update_box` — wołał nieistniejące `box.update(...)`. Zamienione na bezpośrednie przypisanie + `_invalidate_cache()`.

**Usunięty martwy kod:**
- Pliki `row_editor.py`, `row_manager.py`, `gui.py` — szkielety bez użyć w runtime.
- `row_detector.py:Row.slope` (property) — nigdzie nie używane.
- `row_detector.py:_assign_boxes_to_line` — przypisanie `box.color = (0, 255, 0)` (nadpisywane per frame).
- `row_detector.py:remove_line` — zbędny `try/except ValueError` wokół list comprehension.
- `input_handler.py:_update_row_boxes` — przypisania kolorów `(0, 255, 0)` / `(0, 0, 255)` (nadpisywane per frame).
- `input_handler.py:_handle_left_down` — redundantne `isinstance(line, RowLine)`.
- `image_window.py:_release_resources` — pętla `for img in self._cached_images: img = None` (no-op, lokalna zmienna).

**Eliminacja powieleń:**
- `main.py` — usunięty import i tworzenie `RowDetector(bbox_manager)`; `InputHandler` dostaje `row_detector=None`. `ImageWindow.__init__` jest jedynym właścicielem `RowDetector` (przedtem tworzony dwukrotnie i natychmiast nadpisywany).

**Stylowe poprawki (zero zmiany działania):**
- `bounding_box.py:resize_corner` — czytelny `if/elif` zamiast idiomu `setattr-or-setattr` w lambdach.
- `input_handler.py:get_key_bindings_info` — `base_keys.copy()` zamiast referencji.
- `image_cropper.py` — `TYPE_CHECKING` dopełniony o `BoundingBoxManager`, poprawiona błędna ścieżka importu `BoundingBox` (z `bounding_box_manager` na `bounding_box`).

**Świadomie nie ruszane:**
- Hardkody `mtiurina` w `YOLO/predict.py` i `YOLO/config.yaml` (legacy/standalone, poza runtime).
- Edge-case'y dodające nowe ścieżki kodu (np. `aspect_ratio` przy `h=0`, pusta lista wyników YOLO).
- `mark_dirty()` w pętli (granica behawioralnej neutralności).

**Weryfikacja statyczna:** wszystkie zmodyfikowane pliki przeszły `python -m py_compile`. Pełny import zablokowany przez zepsute `.venv` projektu — wymaga rekonstrukcji środowiska.

---

## 12. Branch i commit context

- Aktualna branża: `row_detector_ransac` (sugeruje że jest planowane wprowadzenie RANSAC do detekcji wierszy — obecnie w kodzie nie ma RANSAC, użyte jest geometryczne przecięcie odcinek-prostokąt).
- Ostatnie commity przed cleanupem: `convert to small letters`, `starting with manual mode part 2` (×2), `starting with manual mode`, `display working mode on picture`.
- Modyfikacje robocze: `.idea/`, masa usuniętych plików z `output_crops/TUR_BITS_2015_*`, plus zmiany z cleanupu w sekcji 11 (do scommitowania osobno).

---

*Dokument wygenerowany 2026-05-28, zaktualizowany po cleanupie tego samego dnia. Aktualizuj po większych refaktorach.*
