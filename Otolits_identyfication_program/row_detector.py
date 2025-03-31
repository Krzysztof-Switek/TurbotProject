import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from math import hypot
from bounding_box import BoundingBox


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
    @dataclass
    class Row:
        id: int  # Numer wiersza
        line: RowLine  # Linia przypisana do tego wiersza
        boxes: List[BoundingBox] = field(default_factory=list)  # Typowana lista bboxów

        def add_box(self, box: BoundingBox, max_distance: float = 50.0) -> bool:
            """Dodaje box do wiersza jeśli jest blisko linii"""
            if not self._is_box_near_line(box, max_distance):
                return False

            if box not in self.boxes:
                self.boxes.append(box)
                # Automatyczne sortowanie od lewej do prawej
                self.boxes.sort(key=lambda b: b.x1 + b.width() / 2)  # Sort po środku boxa
            return True

        def remove_box(self, box: BoundingBox) -> bool:
            """Usuwa box z wiersza"""
            try:
                self.boxes.remove(box)
                return True
            except ValueError:
                return False

        def _is_box_near_line(self, box: BoundingBox, max_distance: float) -> bool:
            """Sprawdza czy box jest wystarczająco blisko linii wiersza"""
            box_center = (box.x1 + box.width() / 2, box.y1 + box.height() / 2)
            distance = self._distance_to_line(self.line.p1, self.line.p2, box_center)
            return distance <= max_distance

        def _distance_to_line(self, p1: Tuple[float, float],
                              p2: Tuple[float, float],
                              point: Tuple[float, float]) -> float:
            """Oblicza odległość punktu od linii"""
            x1, y1 = p1
            x2, y2 = p2
            x0, y0 = point

            if x1 == x2:  # Linia pionowa
                return abs(x0 - x1)

            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2

            return abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5

    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowDetector.Row] = []
        self.current_line: Optional[RowLine] = None
        self.drawing_started = False

    def start_new_line(self, x: int, y: int) -> None:
        self.current_line = RowLine(
            p1=(float(x), float(y)),
            p2=(float(x), float(y)),
            id=str(uuid.uuid4())
        )
        self.drawing_started = True

    def update_line_end(self, x: int, y: int) -> None:
        if self.current_line and self.drawing_started:
            self.current_line.p2 = (float(x), float(y))

    def finish_line(self) -> None:
        if self.current_line and self.drawing_started:
            print(f"[DEBUG] Przypisywanie boxów do linii {self.current_line.id}...")
            self._assign_boxes_to_line()
            self.current_line = None
            self.drawing_started = False

    def draw_rows(self, image: np.ndarray) -> None:
        """Rysuje wszystkie linie na obrazie"""
        if image is None:
            return

        for row in self.rows:
            cv2.line(image,
                     (int(row.line.p1[0]), int(row.line.p1[1])),
                     (int(row.line.p2[0]), int(row.line.p2[1])),
                     row.line.color, 2)

        if self.current_line and self.drawing_started:
            cv2.line(image,
                     (int(self.current_line.p1[0]), int(self.current_line.p1[1])),
                     (int(self.current_line.p2[0]), int(self.current_line.p2[1])),
                     self.current_line.color, 1, cv2.LINE_AA)

    def clear_rows(self) -> None:
        """Czyści wszystkie linie"""
        self.rows = []
        self.current_line = None
        self.drawing_started = False
        print("All lines cleared")

    def remove_line(self, line: RowLine) -> bool:
        """Usuwa linię z listy"""
        try:
            self.rows = [row for row in self.rows if row.line != line]
            return True
        except ValueError:
            return False

    def get_line_at(self, x: int, y: int, tolerance: float = 10.0) -> Optional[RowLine]:
        """Znajduje linię w pobliżu punktu (x,y)"""
        for row in self.rows:
            dist_p1 = hypot(x - row.line.p1[0], y - row.line.p1[1])
            dist_p2 = hypot(x - row.line.p2[0], y - row.line.p2[1])
            dist_line = self._distance_to_line(row.line.p1, row.line.p2, (x, y))

            if min(dist_p1, dist_p2, dist_line) < tolerance:
                return row.line
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

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        return abs(A * x0 + B * y0 + C) / (A ** 2 + B ** 2) ** 0.5

    def _assign_boxes_to_line(self) -> None:
        """Przypisuje boxy do linii, tworząc nowy wiersz."""
        if not self.current_line:
            print("[DEBUG] Brak aktualnej linii, wychodzę.")
            return

        new_row = self.Row(
            id=len(self.rows) + 1,
            line=self.current_line
        )

        for box in self.bbox_manager.boxes:
            print(f"[DEBUG] Sprawdzam box {box}")
            new_row.add_box(box)  # Automatyczna walidacja i sortowanie

        if new_row.boxes:
            self.rows.append(new_row)
            print(f"Utworzono wiersz {new_row.id} z {len(new_row.boxes)} boxami")
        else:
            print("Żaden box nie został przypisany do linii.")