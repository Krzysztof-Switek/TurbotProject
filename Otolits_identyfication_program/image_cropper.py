import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


# "Ładne" długości paska skali w μm — wybór z tej listy daje czytelne podpisy
# typu "500 μm", "1 mm" itp. Standardowe wartości w mikroskopii.
_NICE_UM_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

if TYPE_CHECKING:
    from image_loader import ImageLoader
    from row_detector import RowLine
    from bounding_box import BoundingBox
    from bounding_box_manager import BoundingBoxManager
    from input_handler import InputHandler


def compute_row_labels(rows, swap: bool = False) -> Tuple[Dict[int, str], Optional[str]]:
    """Wyznacza etykiety (np. "A_1", "B_3") dla wszystkich wierszy.

    Zwraca krotkę (labels_by_row_id, error_msg):
    - labels_by_row_id: {id(row): "A_n"} dla każdego wiersza posiadającego boxy.
    - error_msg: None jeśli OK, lub komunikat błędu walidacji (>6 wierszy total
      albo >3 wierszy w wycinku).

    Algorytm: sort top→bottom → walidacja total → split na A/B przez largest
    gap Y → walidacja per wycinek → numerowanie "od dołu" w obrębie wycinka.

    swap=True zamienia litery wycinków A↔B na końcu (przycisk "swap slices" w UI).
    Walidacja jest symetryczna, więc liczona przed zamianą.
    """
    sorted_rows = sorted(
        (r for r in rows if getattr(r, 'boxes', None)),
        key=lambda row: min(b.y1 for b in row.boxes),
    )

    total = len(sorted_rows)
    if total > 6:
        return {}, (
            f"BŁĄD: wykryto {total} wierszy łącznie (max 6: po 3 na wycinek). "
            f"Popraw wiersze ręcznie (usuń nadmiarowe linie)."
        )

    a_rows, b_rows = _split_compartments(sorted_rows)
    n_a = len(a_rows)
    n_b = len(b_rows)

    if n_a > 3 or n_b > 3:
        return {}, (
            f"BŁĄD: wycinek A ma {n_a} wierszy, wycinek B ma {n_b} wierszy "
            f"(max 3 na wycinek). Popraw wiersze ręcznie."
        )

    labels: Dict[int, str] = {}
    for label, rows_in_c in (('A', a_rows), ('B', b_rows)):
        eff_label = ('B' if label == 'A' else 'A') if swap else label
        n = len(rows_in_c)
        for offset, row in enumerate(rows_in_c):
            row_num = 3 - (n - 1 - offset)
            labels[id(row)] = f"{eff_label}_{row_num}"
    return labels, None


def compute_compartment_bboxes(
    rows,
    margin: int = 40,
    labels: Optional[Dict[int, str]] = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Wyznacza bbox każdego wycinka jako bounding-box wszystkich boxów w jego
    wierszach + margines (w pikselach przestrzeni podglądu).

    Zwraca {'A': (x1, y1, x2, y2), 'B': (x1, y1, x2, y2)}. Wycinki bez wierszy
    pomijane. Gdy compute_row_labels zwraca błąd walidacji, zwracamy {}.

    Parametr `labels` pozwala przekazać już obliczone etykiety i uniknąć
    powtórnego wywołania compute_row_labels (perf optimization w UI).
    """
    if labels is None:
        labels, err = compute_row_labels(rows)
        if err is not None:
            return {}

    by_compartment: Dict[str, list] = {'A': [], 'B': []}
    for row in rows:
        if not getattr(row, 'boxes', None):
            continue
        label = labels.get(id(row))
        if not label:
            continue
        by_compartment[label[0]].extend(row.boxes)

    result: Dict[str, Tuple[float, float, float, float]] = {}
    for label, boxes in by_compartment.items():
        if not boxes:
            continue
        x1 = min(b.x1 for b in boxes) - margin
        y1 = min(b.y1 for b in boxes) - margin
        x2 = max(b.x2 for b in boxes) + margin
        y2 = max(b.y2 for b in boxes) + margin
        result[label] = (x1, y1, x2, y2)
    return result


def _split_compartments(sorted_rows):
    """Podział wierszy na wycinek A (górny) i B (dolny).

    Wiersze z compartment_override == 'A' / 'B' trafiają do swojego wycinka
    niezależnie od geometrii. Pozostałe (override == None) dzielone przez
    największą lukę Y między sobą. Końcowe listy posortowane top→bottom.

    sorted_rows musi być już posortowane top→bottom.
    """
    def _row_top_y(row):
        return min(b.y1 for b in row.boxes)

    forced_a = [r for r in sorted_rows if getattr(r, 'compartment_override', None) == 'A']
    forced_b = [r for r in sorted_rows if getattr(r, 'compartment_override', None) == 'B']
    auto = [r for r in sorted_rows if getattr(r, 'compartment_override', None) is None]

    auto_a, auto_b = _split_auto_by_largest_gap(auto)

    a_rows = sorted(forced_a + auto_a, key=_row_top_y)
    b_rows = sorted(forced_b + auto_b, key=_row_top_y)
    return a_rows, b_rows


def _split_auto_by_largest_gap(sorted_rows):
    """Dzieli wiersze (bez override) przez największą lukę Y między centroidami.

    sorted_rows musi być już posortowane top→bottom.
    """
    n = len(sorted_rows)
    if n == 0:
        return [], []
    if n == 1:
        return list(sorted_rows), []

    centroids = [
        sum((b.y1 + b.y2) / 2 for b in row.boxes) / len(row.boxes)
        for row in sorted_rows
    ]
    gaps = [(centroids[i + 1] - centroids[i], i) for i in range(n - 1)]
    _, split_idx = max(gaps)

    a_rows = list(sorted_rows[:split_idx + 1])
    b_rows = list(sorted_rows[split_idx + 1:])
    return a_rows, b_rows


def _pick_nice_um(width_px: int, um_per_px: float) -> float:
    """Wybiera 'ładną' długość paska skali (μm) dla wycinka.

    Cel: pasek zajmuje ~25% szerokości, max 40% (żeby zostawić margines).
    Zwraca 0 gdy żadna wartość z `_NICE_UM_VALUES` się nie mieści (wycinek
    skrajnie mały) — wtedy pasek nie jest rysowany.
    """
    if um_per_px <= 0 or width_px <= 0:
        return 0.0
    width_um = width_px * um_per_px
    target_um = 0.25 * width_um
    max_um = 0.40 * width_um
    fitting = [v for v in _NICE_UM_VALUES if v <= max_um]
    if not fitting:
        return 0.0
    return min(fitting, key=lambda v: abs(v - target_um))


def draw_scale_bar(image: np.ndarray, um_per_px: float) -> np.ndarray:
    """Rysuje pasek skali w prawym dolnym rogu wycinka.

    Wygląd: biały prostokąt z czarnym outline + biały tekst z czarnym outline
    nad paskiem. PIL używany dla rendering Unicode (μm).
    Zwraca nowy obraz (bez modyfikacji wejścia).
    """
    if um_per_px is None or um_per_px <= 0:
        return image

    h, w = image.shape[:2]
    nice_um = _pick_nice_um(w, um_per_px)
    if nice_um <= 0:
        return image

    bar_len_px = max(1, int(nice_um / um_per_px))

    # Etykieta: ≥1000 μm → mm, inaczej μm.
    if nice_um >= 1000:
        mm_val = nice_um / 1000
        label = f"{mm_val:g} mm"
    else:
        label = f"{int(nice_um)} μm"

    # Konwersja BGR (cv2) → RGB → PIL.
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    margin = 15
    bar_height = 5
    x_right = w - margin
    x_left = x_right - bar_len_px
    y_bar_bottom = h - margin
    y_bar_top = y_bar_bottom - bar_height

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    text_x = x_right - tw
    text_y = y_bar_top - th - 6

    # Pasek z czarnym outline.
    draw.rectangle(
        [x_left - 1, y_bar_top - 1, x_right + 1, y_bar_bottom + 1],
        fill=(0, 0, 0),
    )
    draw.rectangle(
        [x_left, y_bar_top, x_right, y_bar_bottom],
        fill=(255, 255, 255),
    )

    # Tekst z czarnym outline (4 offsety + biały środek).
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((text_x + dx, text_y + dy), label, font=font, fill=(0, 0, 0))
    draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@dataclass
class CropResult:
    image: np.ndarray
    box_index: int
    row_index: int
    original_coords: Tuple[int, int, int, int]
    filename: str


class ImageCropper:
    def __init__(self, output_dir: str = "output_crops", image_loader: 'ImageLoader' = None):
        self.output_dir = output_dir
        self.image_loader = image_loader
        os.makedirs(output_dir, exist_ok=True)

    def crop_and_save(
        self,
        original_image: np.ndarray,
        rows: List['RowLine'],
        boxes: List['BoundingBox'],
        um_per_px: Optional[float] = None,
        swap: bool = False,
    ) -> List[CropResult]:
        if original_image is None:
            print("Brak obrazu do wycięcia")
            return []

        results = []

        original_path = self.image_loader.current_image_path if self.image_loader else None
        original_filename = os.path.splitext(os.path.basename(original_path))[0] if original_path else "image"

        # Wyznacz etykiety dla wszystkich wierszy (logika wspólna z UI).
        labels, err = compute_row_labels(rows, swap=swap)
        if err is not None:
            print(f"{err} Anuluję zapis.")
            return []

        # Iteracja w kolejności globalnej (top→bottom) zachowując semantykę
        # CropResult.row_index = 1..n.
        sorted_rows = sorted(
            (r for r in rows if getattr(r, 'boxes', None)),
            key=lambda row: min(b.y1 for b in row.boxes),
        )

        for absolute_row_idx, row in enumerate(sorted_rows, start=1):
            prefix = labels[id(row)]

            # Sort boxes left to right
            sorted_boxes = sorted(row.boxes, key=lambda b: (b.x1 + b.x2) / 2)

            for box_idx, box in enumerate(sorted_boxes, start=1):
                # Original cropping logic remains unchanged
                if self.image_loader:
                    x1, y1, x2, y2 = self.image_loader.scale_coords_to_original(box.x1, box.y1, box.x2, box.y2)
                else:
                    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

                h, w = original_image.shape[:2]
                x1, x2 = sorted([max(0, min(w, x1)), max(0, min(w, x2))])
                y1, y2 = sorted([max(0, min(h, y1)), max(0, min(h, y2))])

                if x1 >= x2 or y1 >= y2:
                    continue

                try:
                    cropped = original_image[y1:y2, x1:x2].copy()
                    if cropped.size == 0:
                        continue

                    # Pasek skali (opcjonalny — gdy backend dostał um_per_px).
                    if um_per_px is not None and um_per_px > 0:
                        cropped = draw_scale_bar(cropped, um_per_px)

                    # New filename format
                    filename = f"{original_filename}{prefix}_{box_idx}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, cropped)

                    results.append(CropResult(
                        image=cropped,
                        box_index=box_idx,
                        row_index=absolute_row_idx,  # Absolute position preserved
                        original_coords=(x1, y1, x2, y2),
                        filename=filename
                    ))
                except Exception as e:
                    print(f"Błąd podczas wycinania boxu: {e}")

        return results

    def process_cropping(self, bbox_manager: 'BoundingBoxManager', input_handler: 'InputHandler'):
        """ Handles cropping boxes and saving them to disk """
        try:
            if not self.image_loader:
                print("ImageLoader nie został poprawnie zainicjalizowany")
                return

            original_image = self.image_loader.get_original_image()
            if original_image is None:
                print("Nie można załadować oryginalnego obrazu")
                return

            rows = input_handler.row_detector.rows if hasattr(input_handler, 'row_detector') else []

            print("Rozpoczynanie procesu wycinania boxów...")
            results = self.crop_and_save(
                original_image,
                rows,
                bbox_manager.boxes
            )

            if results:
                print(f"\nPomyślnie wycięto i zapisano {len(results)} boxów:")
                for result in results:
                    print(f"- {result.filename}")
                print(f"Pliki zapisano w: {os.path.abspath(self.output_dir)}\n")

                # Zapis adnotacji wycinków w formacie YOLO obok zdjęcia
                # źródłowego (active learning loop dla przyszłego treningu).
                self._save_compartment_annotations(original_image, rows)
            else:
                print("Nie udało się wyciąć żadnych boxów")

        except Exception as e:
            print(f"Image_cropper - Błąd podczas wycinania boxów: {str(e)}")

    def _save_compartment_annotations(self, original_image: np.ndarray, rows) -> None:
        """Zapisuje bbox-y wycinków A/B w formacie YOLO obok zdjęcia źródłowego.

        Format: `0 cx cy w h` znormalizowane do wymiarów oryginalnego obrazu,
        jedna linia per wycinek. Klasa 0 = "compartment". Plik `.txt` lądą
        w katalogu zdjęcia źródłowego (np. `test_images/<img>.txt`).
        """
        bboxes_preview = compute_compartment_bboxes(rows)
        if not bboxes_preview:
            return

        image_path = self.image_loader.current_image_path if self.image_loader else None
        if not image_path:
            return

        h, w = original_image.shape[:2]
        annotation_path = os.path.splitext(image_path)[0] + ".txt"

        lines = []
        for label in ('A', 'B'):
            if label not in bboxes_preview:
                continue
            x1_p, y1_p, x2_p, y2_p = bboxes_preview[label]
            if self.image_loader:
                x1, y1, x2, y2 = self.image_loader.scale_coords_to_original(
                    x1_p, y1_p, x2_p, y2_p
                )
            else:
                x1, y1, x2, y2 = x1_p, y1_p, x2_p, y2_p

            # Clip do granic obrazu (compartment z marginesem może wyjść poza)
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            return

        try:
            with open(annotation_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
            print(f"Zapisano adnotacje YOLO: {annotation_path}")
        except OSError as e:
            print(f"Nie udało się zapisać adnotacji {annotation_path}: {e}")