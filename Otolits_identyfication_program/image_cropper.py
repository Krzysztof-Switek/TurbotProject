import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from image_loader import ImageLoader
    from row_detector import RowLine
    from bounding_box import BoundingBox
    from bounding_box_manager import BoundingBoxManager
    from input_handler import InputHandler


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

    def crop_and_save(self, original_image: np.ndarray, rows: List['RowLine'], boxes: List['BoundingBox']) -> List[
        CropResult]:
        if original_image is None:
            print("Brak obrazu do wycięcia")
            return []

        results = []

        original_path = self.image_loader.current_image_path if self.image_loader else None
        original_filename = os.path.splitext(os.path.basename(original_path))[0] if original_path else "image"

        # Sort rows from top to bottom
        sorted_rows = sorted(rows, key=lambda row: min(b.y1 for b in row.boxes))

        # Walidacja globalna: max 6 wierszy łącznie (po 3 na wycinek A i B).
        total = len(sorted_rows)
        if total > 6:
            print(
                f"BŁĄD: wykryto {total} wierszy łącznie (max 6: po 3 na wycinek). "
                f"Anuluję zapis. Popraw wiersze ręcznie (usuń nadmiarowe linie)."
            )
            return []

        # Podział wierszy na wycinek A (górny) i B (dolny) przez największą lukę Y.
        a_rows, b_rows = self._split_compartments(sorted_rows)
        n_a = len(a_rows)
        n_b = len(b_rows)

        # Walidacja per wycinek: max 3 wiersze w każdym.
        if n_a > 3 or n_b > 3:
            print(
                f"BŁĄD: wycinek A ma {n_a} wierszy, wycinek B ma {n_b} wierszy "
                f"(max 3 na wycinek). Anuluję zapis. Popraw wiersze ręcznie."
            )
            return []

        # Iteracja per wycinek; numerowanie "od dołu" w obrębie wycinka:
        # najniższy wiersz = _3, każdy w górę = numer o 1 mniejszy.
        absolute_row_idx = 0
        for label, rows_in_c in (('A', a_rows), ('B', b_rows)):
            n = len(rows_in_c)
            for offset, row in enumerate(rows_in_c):
                absolute_row_idx += 1
                row_num = 3 - (n - 1 - offset)
                prefix = f"{label}_{row_num}"
                display_row_num = row_num

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

    def _split_compartments(self, sorted_rows):
        """Podział wierszy na wycinek A (górny) i B (dolny) przez największą lukę Y.

        sorted_rows musi być już posortowane top→bottom. Dla 0 wierszy zwraca
        ([], []), dla 1 wiersza — wszystko do A. Dla 2+ wierszy znajduje
        największą pionową lukę między centroidami wierszy i tam dzieli.

        Brak heurystyki "is large enough" — zawsze split przez największą lukę.
        Manualny override przyjdzie z UI w późniejszych iteracjach.
        """
        n = len(sorted_rows)
        if n == 0:
            return [], []
        if n == 1:
            return list(sorted_rows), []

        # Centroid Y każdego wiersza = średnia Y środków boxów w wierszu.
        centroids = [
            sum((b.y1 + b.y2) / 2 for b in row.boxes) / len(row.boxes)
            for row in sorted_rows
        ]
        # Największa luka między kolejnymi centroidami → indeks po którym dzielimy.
        gaps = [(centroids[i + 1] - centroids[i], i) for i in range(n - 1)]
        _, split_idx = max(gaps)

        a_rows = list(sorted_rows[:split_idx + 1])
        b_rows = list(sorted_rows[split_idx + 1:])
        return a_rows, b_rows

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

            print("Rozpoczynanie procesu wycinania boxów...")
            results = self.crop_and_save(
                original_image,
                input_handler.row_detector.rows if hasattr(input_handler, 'row_detector') else [],
                bbox_manager.boxes
            )

            if results:
                print(f"\nPomyślnie wycięto i zapisano {len(results)} boxów:")
                for result in results:
                    print(f"- {result.filename}")
                print(f"Pliki zapisano w: {os.path.abspath(self.output_dir)}\n")
            else:
                print("Nie udało się wyciąć żadnych boxów")

        except Exception as e:
            print(f"Image_cropper - Błąd podczas wycinania boxów: {str(e)}")