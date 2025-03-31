import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import uuid
import math
from bounding_box import BoundingBox
from math import hypot

@dataclass
class RowLine:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    id: str
    color: Tuple[int, int, int] = (0, 255, 255)  # Żółty
    thickness: int = 2

    def move(self, dx: float, dy: float):
        self.p1 = (self.p1[0] + dx, self.p1[1] + dy)
        self.p2 = (self.p2[0] + dx, self.p2[1] + dy)


class RowDetector:
    @dataclass
    class Row:
        id: int
        line: RowLine
        boxes: List[BoundingBox] = field(default_factory=list)

        def add_box(self, box: BoundingBox) -> bool:
            """Dodaje box jeśli przecina się z linią wiersza"""
            if not self._does_line_intersect_box(self.line, box):
                return False

            if box not in self.boxes:
                self.boxes.append(box)
                self.boxes.sort(key=lambda b: b.x1 + b.width() / 2)
            return True

        def remove_box(self, box: BoundingBox) -> bool:
            try:
                self.boxes.remove(box)
                return True
            except ValueError:
                return False

        def _does_line_intersect_box(self, line: RowLine, box: BoundingBox) -> bool:
            """Sprawdza czy linia przecina box (dokładne sprawdzenie geometrii)"""
            # Konwersja punktów do numpy array
            line_p1 = np.array(line.p1)
            line_p2 = np.array(line.p2)

            # Wierzchołki boxa w kolejności zgodnej z ruchem wskazówek zegara
            box_corners = [
                np.array([box.x1, box.y1]),  # Lewy górny
                np.array([box.x2, box.y1]),  # Prawy górny
                np.array([box.x2, box.y2]),  # Prawy dolny
                np.array([box.x1, box.y2])  # Lewy dolny
            ]

            # Sprawdź przecięcie z każdą krawędzią boxa
            for i in range(4):
                a = box_corners[i]
                b = box_corners[(i + 1) % 4]

                if self._line_segments_intersect(line_p1, line_p2, a, b):
                    return True

            # Sprawdź czy linia jest całkowicie wewnątrz boxa
            return (self._point_in_box(line_p1, box) or
                    self._point_in_box(line_p2, box))

        def _line_segments_intersect(self, p1: np.ndarray, p2: np.ndarray,
                                     q1: np.ndarray, q2: np.ndarray) -> bool:
            """Sprawdza czy dwa odcinki się przecinają"""
            r = p2 - p1
            s = q2 - q1
            qp = q1 - p1

            cross_rs = np.cross(r, s)
            if abs(cross_rs) < 1e-12:  # Linie równoległe
                return False

            t = np.cross(qp, s) / cross_rs
            u = np.cross(qp, r) / cross_rs

            return (0 <= t <= 1) and (0 <= u <= 1)

        def _point_in_box(self, point: np.ndarray, box: BoundingBox) -> bool:
            """Sprawdza czy punkt jest wewnątrz boxa"""
            return (box.x1 <= point[0] <= box.x2 and
                    box.y1 <= point[1] <= box.y2)

    # Reszta klasy RowDetector pozostaje bez zmian (poza usunięciem nieużywanych metod)
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
        """Aktualizuje koniec linii tylko gdy jest aktywny proces rysowania"""
        if self.current_line is not None:  # Wystarczy sprawdzenie current_line
            self.current_line.p2 = (float(x), float(y))

    def finish_line(self) -> None:
        """Kończy rysowanie linii z walidacją i gwarancją spójności stanu"""
        # Sprawdzenie spójności stanu
        if self.current_line is None:
            self.drawing_started = False  # Naprawa ewentualnej niespójności
            return

        # Walidacja długości linii
        line_length = math.hypot(
            self.current_line.p2[0] - self.current_line.p1[0],
            self.current_line.p2[1] - self.current_line.p1[1]
        )

        if line_length < 10.0:  # Minimalna długość 10px
            print("Odrzucono linię: zbyt krótka (minimalna długość: 10px)")
            self._reset_drawing_state()
            return

        # Przypisanie bboxów do linii
        self._assign_boxes_to_line()
        self._reset_drawing_state()

    def _reset_drawing_state(self) -> None:
        """Prywatna metoda do resetowania stanu rysowania"""
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

        if self.current_line is not None:
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


    def _assign_boxes_to_line(self) -> None:
        """Przypisuje boxy do linii, tworząc nowy wiersz."""
        if not self.current_line:
            return

        new_row = self.Row(
            id=len(self.rows) + 1,
            line=self.current_line
        )

        for box in self.bbox_manager.boxes:
            new_row.add_box(box)


        if new_row.boxes:
            self.rows.append(new_row)
            print(f"W wierszu {new_row.id} przypisano {len(new_row.boxes)} boksów.")