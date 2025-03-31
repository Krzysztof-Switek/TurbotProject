import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from image_loader import ImageLoader
    from row_detector import RowLine
    from bounding_box_manager import BoundingBox
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

    def crop_and_save(self,
                      original_image: np.ndarray,
                      rows: List['RowLine'],
                      boxes: List['BoundingBox']) -> List[CropResult]:

        if original_image is None:
            print("Brak obrazu do wycięcia")
            return []

        results = []

        # Sortowanie wierszy od góry do dołu
        sorted_rows = sorted(rows,
                             key=lambda row: row.intercept if abs(row.slope) < 0.01 else min(b.y1 for b in row.boxes))

        for row_idx, row in enumerate(sorted_rows):
            # Sortowanie boxów w wierszu od lewej do prawej
            sorted_boxes = sorted(row.boxes, key=lambda b: (b.x1 + b.x2) / 2)

            for box_idx, box in enumerate(sorted_boxes):
                # Przelicz współrzędne boxa na oryginalną rozdzielczość
                if self.image_loader:
                    x1, y1, x2, y2 = self.image_loader.scale_coords_to_original(box.x1, box.y1, box.x2, box.y2)
                else:
                    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

                # Zabezpieczenie przed przekroczeniem wymiarów
                h, w = original_image.shape[:2]
                x1, x2 = sorted([max(0, min(w, x1)), max(0, min(w, x2))])
                y1, y2 = sorted([max(0, min(h, y1)), max(0, min(h, y2))])

                if x1 >= x2 or y1 >= y2:
                    continue

                try:
                    cropped = original_image[y1:y2, x1:x2].copy()
                    if cropped.size == 0:
                        continue

                    filename = f"row_{row_idx:02d}_box_{box_idx:02d}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, cropped)

                    results.append(CropResult(
                        image=cropped,
                        box_index=box_idx,
                        row_index=row_idx,
                        original_coords=(x1, y1, x2, y2),
                        filename=filename
                    ))
                except Exception as e:
                    print(f"Błąd podczas wycinania boxu: {e}")

        return results

    def process_cropping(self, bbox_manager: 'BoundingBoxManager', input_handler: 'InputHandler'):
        """ Obsługuje wycinanie boxów i zapisuje je na dysk """
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
