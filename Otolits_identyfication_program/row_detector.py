import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
import uuid
from math import hypot


@dataclass
class RowLine:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    id: str
    color: Tuple[int, int, int] = (0, 255, 255)  # Żółty
    thickness: int = 2

    def move(self, dx: float, dy: float):
        """Przesuwa całą linię"""
        self.p1 = (self.p1[0] + dx, self.p1[1] + dy)
        self.p2 = (self.p2[0] + dx, self.p2[1] + dy)

    def move_p1(self, dx: float, dy: float):
        self.p1 = (self.p1[0] + dx, self.p1[1] + dy)

    def move_p2(self, dx: float, dy: float):
        self.p2 = (self.p2[0] + dx, self.p2[1] + dy)

class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []
        self.current_line: Optional[RowLine] = None
        self.drawing_started = False

    def start_new_line(self, x: int, y: int) -> None:
        self.current_line = RowLine(
            p1=(float(x), float(y)),
            p2=(float(x), float(y)),  # Początkowo ten sam punkt
            id=str(uuid.uuid4())
        )
        self.drawing_started = True

    def update_line_end(self, x: int, y: int) -> None:
        if self.current_line and self.drawing_started:
            self.current_line.p2 = (float(x), float(y))

    def finish_line(self) -> None:
        if self.current_line and self.drawing_started:
            self.rows.append(self.current_line)
            self.current_line = None
            self.drawing_started = False

    def draw_rows(self, image: np.ndarray) -> None:
        """Rysuje wszystkie linie na obrazie"""
        if image is None:
            return

        # Rysowanie istniejących linii (grubsze, bardziej widoczne)
        for row in self.rows:
            cv2.line(image,
                     (int(row.p1[0]), int(row.p1[1])),
                     (int(row.p2[0]), int(row.p2[1])),
                     row.color, 2)  # Zwiększona grubość do 3px

        # Rysowanie aktualnie tworzonej linii (przerywana)
        if self.current_line and self.drawing_started:
            cv2.line(image,
                     (int(self.current_line.p1[0]), int(self.current_line.p1[1])),
                     (int(self.current_line.p2[0]), int(self.current_line.p2[1])),
                     self.current_line.color, 1, cv2.LINE_AA)  # Używamy current_line zamiast row


    def clear_rows(self) -> None:
        """Czyści wszystkie linie"""
        self.rows = []
        self.current_line = None
        self.drawing_started = False
        print("All lines cleared")  # Debug

    def remove_line(self, line: RowLine) -> bool:
        """Usuwa linię z listy"""
        try:
            self.rows.remove(line)
            return True
        except ValueError:
            return False

    def get_line_at(self, x: int, y: int, tolerance: float = 10.0) -> Optional[RowLine]:
        """Znajduje linię w pobliżu punktu (x,y)"""
        for line in self.rows:
            # Sprawdzamy odległość od końców linii i od samej linii
            dist_p1 = hypot(x - line.p1[0], y - line.p1[1])
            dist_p2 = hypot(x - line.p2[0], y - line.p2[1])
            dist_line = self._distance_to_line(line.p1, line.p2, (x, y))

            if min(dist_p1, dist_p2, dist_line) < tolerance:
                return line
        return None

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

        return abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5

    # def _assign_boxes_to_line(self) -> None:
    #     """Prywatna metoda do przypisywania boxów do linii"""
    #     if not self.current_line:
    #         return
    #
    #     for box in self.bbox_manager.boxes:
    #         if self._is_box_near_line(box):
    #             # Tutaj można dodać logikę przypisania boxa do linii
    #             pass
    #
    # def _is_box_near_line(self, box: BoundingBox) -> bool:
    #     """Sprawdza czy box jest blisko aktualnej linii"""
    #     if not self.current_line:
    #         return False
    #
    #     box_center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
    #     distance = self._distance_to_line(self.current_line.p1, self.current_line.p2, box_center)
    #     return distance < (box.y2 - box.y1) * 0.5  # Próg 50% wysokości boxa
    #
    # def _distance_to_line(self, p1: Tuple[float, float],
    #                       p2: Tuple[float, float],
    #                       point: Tuple[float, float]) -> float:
    #     """Oblicza odległość punktu od linii"""
    #     x1, y1 = p1
    #     x2, y2 = p2
    #     x0, y0 = point
    #
    #     if x1 == x2:  # Linia pionowa
    #         return abs(x0 - x1)
    #
    #     # Równanie linii: Ax + By + C = 0
    #     A = y2 - y1
    #     B = x1 - x2
    #     C = x2 * y1 - x1 * y2
    #
    #     return abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5


