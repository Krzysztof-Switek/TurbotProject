import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CropResult:
    image: np.ndarray
    box_index: int
    row_index: int
    original_coords: Tuple[int, int, int, int]
    filename: str


class ImageCropper:
    def __init__(self, output_dir: str = "output_crops"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def crop_and_save(self,
                      original_image: np.ndarray,
                      rows: List['RowLine'],
                      boxes: List['BoundingBox']) -> List[CropResult]:
        """
        Wycina i zapisuje boxy z oryginalnego obrazu w kolejności wyznaczonej przez linie.

        Args:
            original_image: Oryginalny obraz (pełna rozdzielczość) jako numpy array
            rows: Lista obiektów RowLine z RowDetector
            boxes: Lista obiektów BoundingBox z BoundingBoxManager

        Returns:
            Lista obiektów CropResult z wynikami wycinania
        """
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
                # Znajdź oryginalny box w pełnej rozdzielczości
                original_box = next((b for b in boxes if b.id == box.id), None)
                if not original_box:
                    continue

                # Konwersja współrzędnych na integer i zabezpieczenie przed przekroczeniem wymiarów
                h, w = original_image.shape[:2]
                x1 = int(max(0, min(w, original_box.x1)))
                y1 = int(max(0, min(h, original_box.y1)))
                x2 = int(max(0, min(w, original_box.x2)))
                y2 = int(max(0, min(h, original_box.y2)))

                # Sprawdzenie poprawności współrzędnych
                if x1 >= x2 or y1 >= y2:
                    print(f"Nieprawidłowe współrzędne boxu: ({x1}, {y1}, {x2}, {y2}) - pominięto")
                    continue

                # Wycinanie obszaru
                try:
                    cropped = original_image[y1:y2, x1:x2].copy()  # .copy() aby uniknąć problemów z pamięcią
                    if cropped.size == 0:
                        print(f"Pusty wycinek dla boxu: ({x1}, {y1}, {x2}, {y2}) - pominięto")
                        continue

                    # Generuj nazwę pliku
                    filename = f"row_{row_idx:02d}_box_{box_idx:02d}.png"
                    filepath = os.path.join(self.output_dir, filename)

                    # Zapisz do pliku
                    cv2.imwrite(filepath, cropped)
                    results.append(CropResult(
                        image=cropped,
                        box_index=box_idx,
                        row_index=row_idx,
                        original_coords=(x1, y1, x2, y2),
                        filename=filename
                    ))
                except Exception as e:
                    print(f"Błąd podczas przetwarzania boxu: {str(e)}")

        return results