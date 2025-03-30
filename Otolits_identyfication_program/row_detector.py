import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
import uuid
from bounding_box import BoundingBox


@dataclass
class RowLine:
    p1: Tuple[float, float]  # Punkt startowy linii
    p2: Tuple[float, float]  # Punkt końcowy linii
    boxes: List[BoundingBox]  # Lista przypisanych boxów
    id: str  # Unikalny identyfikator
    color: Tuple[int, int, int] = (0, 255, 255)  # Domyślnie żółty


class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []
        self.current_line: Optional[RowLine] = None

    def start_new_line(self, x: int, y: int) -> None:
        """Rozpoczyna nową linię wiersza"""
        self.current_line = RowLine(
            p1=(float(x), float(y)),
            p2=(float(x), float(y)),
            boxes=[],
            id=str(uuid.uuid4())
        )
        self.rows.append(self.current_line)

    def update_line_end(self, x: int, y: int) -> None:
        """Aktualizuje koniec linii"""
        if self.current_line:
            self.current_line.p2 = (float(x), float(y))

    def assign_boxes_to_current_line(self) -> None:
        """Przypisuje boxy do aktualnej linii"""
        if not self.current_line:
            return

        for box in self.bbox_manager.boxes:
            if self._is_box_near_line(box):
                self.current_line.boxes.append(box)

    def draw_rows(self, image: np.ndarray) -> None:
        """Rysuje linie wierszy na obrazie"""
        if image is None:
            return

        for row in self.rows:
            if row.p1 and row.p2:
                cv2.line(image,
                         (int(row.p1[0]), int(row.p1[1])),
                         (int(row.p2[0]), int(row.p2[1])),
                         row.color, 2)

    def clear_rows(self) -> None:
        """Czyści wszystkie wiersze"""
        self.rows = []
        self.current_line = None

    def _is_box_near_line(self, box: BoundingBox) -> bool:
        """Sprawdza czy box jest blisko aktualnej linii"""
        if not self.current_line:
            return False

        # Prosta metoda - sprawdzenie odległości środka boxa od linii
        box_center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        line_p1 = self.current_line.p1
        line_p2 = self.current_line.p2

        # Oblicz odległość punktu od linii
        distance = self._distance_to_line(line_p1, line_p2, box_center)

        # Próg odległości to 20% wysokości boxa
        threshold = (box.y2 - box.y1) * 0.2
        return distance < threshold

    def _distance_to_line(self, p1: Tuple[float, float],
                          p2: Tuple[float, float],
                          point: Tuple[float, float]) -> float:
        """Oblicza odległość punktu od linii"""
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = point

        if x1 == x2:  # Linia pionowa
            return abs(x0 - x1)

        # Równanie linii: Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        return abs(A * x0 + B * y0 + C) / np.sqrt(A ** 2 + B ** 2)