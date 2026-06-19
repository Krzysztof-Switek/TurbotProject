# 02 — Code Map

All Python modules with classes, methods and functions (including `_private`), signatures and docstrings.

## `Otolits_identyfication_program/YOLO/datasets/split_dataset.py`

- `def get_difference_from_2_list(list1, list2)`
- `def get_split_data(list_id)`
- `def make_folder()`
- `def copy_image(file, id_folder)`

## `Otolits_identyfication_program/YOLO/predict.py`

- `def get_overlap_percentage(rect1, rect2)`
  - Compute the Intersection over Union (IoU) of two rectangles.

## `Otolits_identyfication_program/YOLO/predict_file.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/YOLO/train.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/YOLO/yolo_trainer.py`

**Has `__main__` entry point.**

### class `YoloTrainer`

- `def __init__(self, model, data, imgsz=640, device='cpu', workers=0, batch=4, epochs=200, patience=50, name='turbot_results', amp=False, single_cls=True, bounding_box_manager=None)`
- `def train(self)`
- `def detect_objects(self, image_path)`

## `Otolits_identyfication_program/__init__.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/auto_detector.py`

### class `AutoDetector`

- `def __init__(self, model_path='yolo/weights/best.pt')`
  - Inicjalizacja detektora YOLO
- `def detect(self, image) -> List[Tuple[int, int, int, int]]`
  - Główna metoda wykrywająca otolity na obrazie
- `def _filter_detections(self, boxes_data: list) -> List[Tuple[float, float, float, float]]`
  - Filtruje detekcje według progu pewności i klasy
- `def _remove_overlapping_boxes(self, boxes: list) -> List[Tuple[int, int, int, int]]`
  - Usuwa nakładające się bboxy, zachowując większe
- `def _calculate_iou(box1, box2) -> float`
  - Oblicza Intersection over Union dla dwóch bboxów
- `def _box_area(box) -> float`
  - Oblicza powierzchnię boxa

## `Otolits_identyfication_program/bounding_box.py`

### class `BoundingBox`

- `def __init__(self, x1: float, y1: float, x2: float, y2: float, label: Optional[str]=None, is_temp: bool=False)`
  - Inicjalizacja boxa
- `def _validate_coordinates(self) -> None`
  - Walidacja współrzędnych boxa
- `def move(self, dx: float, dy: float) -> None`
  - Przesuwa box o wektor (dx, dy)
- `def resize_corner(self, corner_idx: int, x: float, y: float) -> None`
  - Zmienia rozmiar poprzez przeciąganie wybranego narożnika
- `def _normalize_coords(self) -> None`
  - Upewnia się że x1 < x2 i y1 < y2
- `def _invalidate_cache(self)`
  - Unieważnia cache po zmianie współrzędnych
- `def contains(self, x: float, y: float, tolerance: float=0) -> bool`
  - Sprawdza czy punkt (x,y) jest w boxie z tolerancją
- `def intersects(self, other: 'BoundingBox') -> bool`
  - Sprawdza czy boxy się nakładają
- `def distance_to(self, x: float, y: float) -> float`
  - Oblicza minimalną odległość punktu od boxa
- `def get_center(self) -> Tuple[float, float]`
  - Zwraca środek boxa (x, y)
- `def get_corners(self) -> List[Tuple[float, float]]`
  - Zwraca współrzędne wszystkich rogów z cache'owaniem
- `def get_nearest_corner(self, x: float, y: float) -> int`
  - Zwraca indeks najbliższego narożnika (0-3)
- `def area(self) -> float`
- `def width(self) -> float`
- `def height(self) -> float`
- `def aspect_ratio(self) -> float`
- `def draw(self, image: np.ndarray, color: Optional[Tuple[int, int, int]]=None, thickness: int=1) -> None`
  - Rysuje box na obrazie
- `def to_dict(self) -> Dict[str, Any]`
  - Konwertuje do słownika (do zapisu)
- `def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox'`
  - Tworzy box ze słownika
- `def __str__(self) -> str`
- `def __repr__(self) -> str`
- `def copy(self) -> 'BoundingBox'`
  - Tworzy dokładną kopię boxa
- `def release(self)`
  - Zwolnienie zasobów (do użycia w BoundingBoxManager.clear_all())

## `Otolits_identyfication_program/bounding_box_manager.py`

### class `BoundingBoxManager`

- `def __init__(self)`
- `def add_box(self, x1_or_box, y1=None, x2=None, y2=None, label=None) -> BoundingBox`
  - Dodaje nowy bounding box - akceptuje współrzędne lub obiekt BoundingBox
- `def remove_box(self, box: BoundingBox) -> bool`
  - Usuwa box i zwraca status operacji
- `def update_box(self, box: BoundingBox, x1: float, y1: float, x2: float, y2: float) -> None`
  - Aktualizuje współrzędne istniejącego boxa
- `def get_boxes(self) -> List[BoundingBox]`
  - Zwraca cache'owaną kopię listy boxów
- `def get_box_at(self, x: float, y: float, tolerance: float=5.0) -> Optional[BoundingBox]`
  - Znajduje box zawierający punkt (x,y) z tolerancją
- `def clear_all(self) -> None`
  - Usuwa wszystkie boxy z dodatkowym czyszczeniem zasobów
- `def to_list(self) -> List[dict]`
  - Eksport boxów do listy słowników
- `def from_list(self, boxes_data: List[dict]) -> None`
  - Import boxów z listy słowników
- `def invalidate_cache(self)`
  - Unieważnia cache po modyfikacjach
- `def __del__(self)`
  - Destruktor - dodatkowe czyszczenie

## `Otolits_identyfication_program/image_cropper.py`

- `def compute_row_labels(rows, swap: bool=False) -> Tuple[Dict[int, str], Optional[str]]`
  - Wyznacza etykiety (np. "A_1", "B_3") dla wszystkich wierszy.
- `def compute_compartment_bboxes(rows, margin: int=40, labels: Optional[Dict[int, str]]=None) -> Dict[str, Tuple[float, float, float, float]]`
  - Wyznacza bbox każdego wycinka jako bounding-box wszystkich boxów w jego
- `def _split_compartments(sorted_rows)`
  - Podział wierszy na wycinek A (górny) i B (dolny).
- `def _split_auto_by_largest_gap(sorted_rows)`
  - Dzieli wiersze (bez override) przez największą lukę Y między centroidami.
- `def _pick_nice_um(width_px: int, um_per_px: float) -> float`
  - Wybiera 'ładną' długość paska skali (μm) dla wycinka.
- `def draw_scale_bar(image: np.ndarray, um_per_px: float) -> np.ndarray`
  - Rysuje pasek skali w prawym dolnym rogu wycinka.

### class `CropResult`


### class `ImageCropper`

- `def __init__(self, output_dir: str='output_crops', image_loader: 'ImageLoader'=None)`
- `def crop_and_save(self, original_image: np.ndarray, rows: List['RowLine'], boxes: List['BoundingBox'], um_per_px: Optional[float]=None, swap: bool=False) -> List[CropResult]`
- `def process_cropping(self, bbox_manager: 'BoundingBoxManager', input_handler: 'InputHandler')`
  - Handles cropping boxes and saving them to disk
- `def _save_compartment_annotations(self, original_image: np.ndarray, rows) -> None`
  - Zapisuje bbox-y wycinków A/B w formacie YOLO obok zdjęcia źródłowego.

## `Otolits_identyfication_program/image_loader.py`

### class `ImageLoader`
_Ładuje obrazy z katalogu (sekwencyjnie), skaluje do podglądu z limitem_

- `def __init__(self, image_dir: str, max_preview_px: int=1920)`
- `def from_path(cls, image_path: str, max_preview_px: int=1920) -> 'ImageLoader'`
  - Tworzy ImageLoader skierowany na konkretny plik (web usecase).
- `def load_image(self) -> Optional[np.ndarray]`
  - Ładuje obraz z pełną walidacją i obsługą błędów.
- `def next_image(self) -> Optional[np.ndarray]`
  - Ładuje następny obraz w kolejności
- `def _resize_to_preview(self, image: np.ndarray) -> np.ndarray`
  - Skaluje obraz do `max_preview_px` z zachowaniem proporcji i korektą orientacji EXIF.
- `def _get_exif_orientation(self) -> Optional[int]`
  - Pobiera orientację EXIF z aktualnego pliku (optymalizacja).
- `def _apply_orientation(self, image: np.ndarray, orientation: int) -> np.ndarray`
  - Stosuje korektę orientacji do obrazu.
- `def get_original_image(self, copy: bool=True) -> Optional[np.ndarray]`
  - Zwraca oryginalny obraz (opcjonalnie kopię).
- `def scale_coords_to_original(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]`
  - Transformuje współrzędne z podglądu do oryginału.
- `def clear(self) -> None`
  - Zwalnia zasoby pamięci zajmowane przez obecny obraz.
- `def current_image_path(self) -> Optional[str]`
  - Ścieżka do aktualnego obrazu

## `Otolits_identyfication_program/image_window.py`

### class `ImageWindow`

- `def __init__(self, image_loader, bbox_manager, input_handler, auto_detector=None)`
- `def _prepare_display_image(self)`
- `def _release_resources(self)`
- `def update_display(self)`
- `def show_image(self)`
- `def mark_dirty(self)`
- `def _handle_mouse_event(self, event, x, y, flags, param)`
- `def _handle_next_image(self)`
- `def _handle_crop_boxes(self)`
- `def _cleanup(self)`

## `Otolits_identyfication_program/input_handler.py`

- `def _get_tk_root()`
  - Singleton Tk root używany przez dialogi desktop. Tworzy `Tk()` raz,

### class `WorkMode(Enum)`


### class `ManualMode(Enum)`


### class `SelectionContext`


### class `InputHandler`

- `def __init__(self, bbox_manager, row_detector)`
- `def get_mode_info(self) -> str`
- `def get_key_bindings_info(self) -> dict`
- `def keyboard_callback(self, key: int) -> bool`
- `def mouse_callback(self, event, x, y) -> bool`
- `def _handle_left_down(self, x: int, y: int) -> bool`
- `def _handle_mouse_move(self, x: int, y: int) -> bool`
- `def _handle_left_up(self, x: int, y: int) -> bool`
- `def _reset_selection(self) -> None`
- `def _set_work_mode(self, mode: WorkMode) -> None`
- `def _set_manual_mode(self, mode: ManualMode) -> None`
- `def reset_to_defaults(self) -> None`
- `def _update_row_boxes(self, row)`
  - Pomocnicza metoda do aktualizacji boxów w wierszu.
- `def _edit_row_label(self, line: RowLine) -> None`
  - Otwiera tkinter prompt 'A' / 'B' / 'auto' i ustawia override w Row.
- `def _prompt_compartment_choice(current: Optional[str]) -> Optional[str]`
  - Prompt tkinter z 'A' / 'B' / 'auto'. Zwraca wybór lub None (Cancel).

## `Otolits_identyfication_program/main.py`

**Has `__main__` entry point.**

_(no top-level classes/functions)_

## `Otolits_identyfication_program/row_detector.py`

### class `RowLine`

- `def move(self, dx: float, dy: float)`

### class `RowDetector`

- `def __init__(self, bbox_manager)`
- `def start_new_line(self, x: int, y: int) -> None`
- `def update_line_end(self, x: int, y: int) -> None`
  - Aktualizuje koniec linii tylko gdy jest aktywny proces rysowania
- `def finish_line(self) -> None`
  - Kończy rysowanie linii z walidacją i gwarancją spójności stanu
- `def _reset_drawing_state(self) -> None`
  - Prywatna metoda do resetowania stanu rysowania
- `def draw_rows(self, image: np.ndarray) -> None`
  - Rysuje wszystkie linie na obrazie
- `def clear_rows(self) -> None`
  - Czyści wszystkie linie
- `def remove_line(self, line: RowLine) -> bool`
  - Usuwa linię z listy
- `def get_line_at(self, x: int, y: int, tolerance: float=10.0) -> Optional[RowLine]`
  - Znajduje linię w pobliżu punktu (x,y)
- `def _assign_boxes_to_line(self) -> None`
  - Optymalne przypisywanie boxów do linii z zachowaniem dokładnej logiki.
- `def _distance_to_line(self, p1: Tuple[float, float], p2: Tuple[float, float], point: Tuple[float, float]) -> float`
  - Oblicza odległość punktu od linii

## `Otolits_identyfication_program/tests/__init__.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/tests/bounding_box_manager_test.py`

- `def test_add_box()`
- `def test_remove_box()`
- `def test_update_box()`
- `def test_get_box_at()`
- `def test_get_box_at_no_match()`

## `Otolits_identyfication_program/tests/image_loader_test.py`

**Has `__main__` entry point.**

### class `TestImageLoader(unittest.TestCase)`

- `def setUpClass(cls)`
  - Tworzy katalog testowy i zapisuje w nim przykładowy obraz
- `def tearDownClass(cls)`
  - Usuwa katalog testowy po zakończeniu testów
- `def setUp(self)`
  - Inicjalizuje ImageLoader dla każdego testu
- `def test_image_files_listed_correctly(self)`
  - Sprawdza, czy pliki są poprawnie wykrywane i sortowane
- `def test_load_image_returns_valid_image(self)`
  - Sprawdza, czy załadowany obraz nie jest pusty
- `def test_screen_scaling(self)`
  - Sprawdza, czy obraz po skalowaniu mieści się na ekranie
- `def test_next_image_behavior(self)`
  - Sprawdza, czy metoda next_image() przechodzi do następnego obrazu lub zwraca None na końcu
- `def test_empty_directory(self)`
  - Sprawdza, czy klasa obsługuje pusty katalog
- `def test_nonexistent_directory(self)`
  - Sprawdza, czy klasa obsłuży nieistniejący katalog i zgłosi wyjątek
- `def test_scale_factor_stored_correctly(self)`
  - Sprawdza, czy ImageLoader poprawnie zapisuje współczynnik skalowania

## `Otolits_identyfication_program/web/__init__.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/web/app.py`

_FastAPI app — entry point dla uvicorn._

- `async def lifespan(app: FastAPI)`
  - Startup: ładuje model YOLO do app.state.detector. Shutdown: nic — model
- `async def no_cache_for_frontend(request, call_next)`
  - Wyłącz cache przeglądarki dla statycznego frontendu (HTML/JS/CSS).

## `Otolits_identyfication_program/web/config.py`

_Konfiguracja aplikacji webowej — odczyt env vars z domyślnymi._

- `def describe() -> str`
  - Tekstowy dump konfiguracji do logów startupu — sanity check.

## `Otolits_identyfication_program/web/routers/__init__.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/web/routers/calibration.py`

_API endpointy kalibracji skali (μm/px) per katalog._

- `def _calibration_to_response(c: calibration_service.Calibration) -> CalibrationResponse`
- `def get_calibration(dir: str=Query('', description='Katalog relatywny do DATA_ROOT'))`
  - Zwraca kalibrację dla katalogu lub null (200 z body=null) jeśli brak.
- `def post_calibration(req: SaveRequest)`
  - Zapisuje kalibrację dla katalogu.

### class `CalibrationResponse(BaseModel)`


### class `SaveRequest(BaseModel)`


## `Otolits_identyfication_program/web/routers/crop.py`

_API endpoint do cropowania boxów + pseudo-labellingu._

- `def crop(req: CropRequest) -> CropResponse`
  - Cropuje boxy z wierszy do output_dir; opcjonalnie zapisuje pseudo-label.

### class `BoxIn(BaseModel)`


### class `LineIn(BaseModel)`


### class `RowIn(BaseModel)`


### class `CropRequest(BaseModel)`


### class `SavedFile(BaseModel)`


### class `CropResponse(BaseModel)`


## `Otolits_identyfication_program/web/routers/detect.py`

_API endpoint do automatycznej detekcji otolitów przez YOLO._

- `def detect(req: DetectRequest, request: Request) -> DetectResponse`
  - Wykrywa otolity YOLO na obrazie wskazanym ścieżką relatywną do DATA_ROOT.

### class `DetectRequest(BaseModel)`


### class `DetectedBox(BaseModel)`


### class `DetectResponse(BaseModel)`


## `Otolits_identyfication_program/web/routers/fs.py`

_API endpoints do nawigacji po katalogach + tworzenia nowych._

- `def list_directory(path: str=Query('', description='Relatywna do DATA_ROOT; pusty = root')) -> ListResponse`
  - Lista podkatalogów + plików obrazowych w danym katalogu.
- `def make_directory(req: MkdirRequest) -> MkdirResponse`
  - Tworzy nowy katalog pod DATA_ROOT (idempotentne: `exist_ok=True`).

### class `DirEntryOut(BaseModel)`


### class `FileEntryOut(BaseModel)`


### class `ListResponse(BaseModel)`


### class `MkdirRequest(BaseModel)`


### class `MkdirResponse(BaseModel)`


## `Otolits_identyfication_program/web/routers/image.py`

_API endpoint do pobrania obrazu (preview, przeskalowanego do MAX_PREVIEW_PX)._

- `def preview(path: str=Query(..., description='Ścieżka pliku relatywna do DATA_ROOT')) -> Response`
  - Zwraca obraz preview (PNG, BGR przez OpenCV) przeskalowany do `MAX_PREVIEW_PX`.

## `Otolits_identyfication_program/web/services/__init__.py`

_(no top-level classes/functions)_

## `Otolits_identyfication_program/web/services/calibration.py`

_Kalibracja skali (μm/px) per katalog — sidecar `calibration.json`._

- `def _calibration_path(rel_dir: str) -> Path`
  - Lokalizacja sidecar JSON. Rzuca PermissionError przy path traversal.
- `def load(rel_dir: str) -> Optional[Calibration]`
  - Wczytuje kalibrację z `<rel_dir>/calibration.json`. None jeśli brak.
- `def save(rel_dir: str, um_per_px: float, magnification: str, reference_image: str, p1: tuple[float, float], p2: tuple[float, float], length_um: float) -> Calibration`
  - Zapisuje kalibrację do `<rel_dir>/calibration.json`. Nadpisuje istniejącą.

### class `Calibration`
_Pełne dane kalibracji + metadane do wyświetlenia w UI / logu._

- `def to_dict(self) -> dict`
- `def from_dict(cls, data: dict) -> 'Calibration'`

## `Otolits_identyfication_program/web/services/fs_browser.py`

_Bezpieczna nawigacja po katalogach w obrębie DATA_ROOT._

- `def safe_resolve(rel_path: str) -> Path`
  - Resolve relatywnej ścieżki do absolutnej z guardem przeciw path traversal.
- `def to_rel(abs_path: Path) -> str`
  - Konwersja absolutnej ścieżki na string relatywny do DATA_ROOT.
- `def list_dir(rel_path: str) -> ListResult`
  - Lista podkatalogów + plików obrazowych w `rel_path` (relatywnym do DATA_ROOT).
- `def make_dir(rel_path: str) -> Path`
  - Tworzy katalog (z parentami) pod DATA_ROOT.

### class `DirEntry`


### class `FileEntry`


### class `ListResult`


## `Otolits_identyfication_program/web/services/image_service.py`

_Wrapper na ImageLoader dla web — ładowanie pojedynczego obrazu po ścieżce_

- `def load(rel_path: str) -> LoadedImage`
  - Ładuje obraz wskazany ścieżką relatywną do DATA_ROOT.
- `def encode_png(image: np.ndarray) -> bytes`
  - Koduje obraz BGR (numpy ndarray) do bajtów PNG.
- `def get_preview_meta(loaded: LoadedImage) -> dict`
  - Metadane preview do nagłówków HTTP (X-Original-Width itp.).

### class `LoadedImage`
_Wynik load_for_preview / load_for_detect._


## `Picks_modification_scripts/Resize.py`

**Has `__main__` entry point.**

- `def resize_image(input_path, output_path)`
- `def process_images()`
