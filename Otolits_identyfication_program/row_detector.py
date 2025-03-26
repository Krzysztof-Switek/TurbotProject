import numpy as np
import cv2
from typing import List
from dataclasses import dataclass


@dataclass
class RowLine:
    slope: float
    intercept: float
    boxes: List['BoundingBox']


class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []
        self.used_boxes = set()  # Zbiór przechowujący ID już użytych boxów

    def detect_rows(self) -> List[RowLine]:
        """Iteracyjna metoda wykrywająca niezależne wiersze od góry do dołu"""
        self.rows = []
        self.used_boxes = set()

        remaining_boxes = self.bbox_manager.boxes.copy()

        while remaining_boxes:
            # Sortuj pozostałe boxy od góry do dołu
            remaining_boxes.sort(key=lambda b: b.y1)

            # Wybierz grupę kandydatów na pierwszy wiersz (najwyższe boxy)
            candidate_boxes = self._select_candidate_boxes(remaining_boxes)

            if not candidate_boxes:
                break

            # Dopasuj linię do kandydatów
            line = self._fit_line_to_boxes(candidate_boxes)

            # Znajdź wszystkie boxy należące do tego wiersza
            row_boxes = self._find_boxes_for_line(line, remaining_boxes)

            if row_boxes:
                self.rows.append(line)
                for box in row_boxes:
                    self.used_boxes.add(box.id)
                    remaining_boxes.remove(box)
            else:
                # Jeśli nie znaleziono boxów dla linii, dodaj najwyższy jako osobny wiersz
                single_box = remaining_boxes.pop(0)
                self.rows.append(RowLine(0.0, (single_box.y1 + single_box.y2) / 2, [single_box]))
                self.used_boxes.add(single_box.id)

        return self.rows

    def _select_candidate_boxes(self, boxes: List['BoundingBox'], threshold: float = 30.0) -> List['BoundingBox']:
        """Wybierz grupę kandydatów na wiersz (najwyższe boxy w podobnej pozycji pionowej)"""
        if not boxes:
            return []

        candidates = [boxes[0]]
        first_center = (boxes[0].y1 + boxes[0].y2) / 2

        for box in boxes[1:]:
            center = (box.y1 + box.y2) / 2
            if abs(center - first_center) < threshold:
                candidates.append(box)
            else:
                break

        return candidates

    def _fit_line_to_boxes(self, boxes: List['BoundingBox']) -> RowLine:
        """Dopasuj linię do grupy boxów metodą najmniejszych kwadratów"""
        if len(boxes) == 1:
            box = boxes[0]
            return RowLine(0.0, (box.y1 + box.y2) / 2, boxes)

        x_centers = [(b.x1 + b.x2) / 2 for b in boxes]
        y_centers = [(b.y1 + b.y2) / 2 for b in boxes]

        A = np.vstack([x_centers, np.ones(len(x_centers))]).T
        slope, intercept = np.linalg.lstsq(A, y_centers, rcond=None)[0]

        return RowLine(slope, intercept, boxes)

    def _find_boxes_for_line(self, line: RowLine, boxes: List['BoundingBox'], threshold: float = 15.0) -> List[
        'BoundingBox']:
        """Znajdź wszystkie boxy należące do danej linii"""
        row_boxes = []

        for box in boxes:
            if box.id in self.used_boxes:
                continue

            # Oblicz odległość środka boxa od linii
            x_center = (box.x1 + box.x2) / 2
            y_center = (box.y1 + box.y2) / 2
            line_y = line.slope * x_center + line.intercept
            distance = abs(y_center - line_y)

            if distance < threshold:
                row_boxes.append(box)

        return row_boxes

    def draw_rows(self, image):
        """Rysuje wykryte linie wierszy na obrazie"""
        for line in self.rows:
            if not line.boxes:
                continue

            x_coords = []
            for box in line.boxes:
                x_coords.append(box.x1)
                x_coords.append(box.x2)

            min_x = int(max(0, min(x_coords)))
            max_x = int(max(0, max(x_coords)))

            if len(line.boxes) == 1 or abs(line.slope) < 0.01:  # Dla pojedynczych lub prawie poziomych
                y = int(line.intercept)
                cv2.line(image, (min_x, y), (max_x, y), (0, 0, 255), 2)
            else:
                y1 = int(line.slope * min_x + line.intercept)
                y2 = int(line.slope * max_x + line.intercept)
                cv2.line(image, (min_x, y1), (max_x, y2), (0, 0, 255), 2)