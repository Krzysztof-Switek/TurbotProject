import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
import uuid
from enum import Enum, auto
from bounding_box import BoundingBox

class RowEditMode(Enum):
    NONE = auto()
    EDIT = auto()
    ADD = auto()


@dataclass
class RowLine:
    slope: float
    intercept: float
    boxes: List[BoundingBox]
    id: str
    p1: Optional[Tuple[float, float]] = None
    p2: Optional[Tuple[float, float]] = None
    locked: bool = False
    color: Tuple[int, int, int] = (0, 0, 255)  # Domyślnie czerwony


class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []
        self.edit_mode = RowEditMode.NONE
        self.selected_row = None
        self.drag_start = None
        self.drag_type = None

    def set_edit_mode(self, mode: RowEditMode) -> bool:
        """Ustawia tryb edycji linii"""
        if isinstance(mode, RowEditMode):  # Dodane sprawdzenie typu
            self.edit_mode = mode
            self._reset_selection()
            return True
        return False

    def handle_mouse_event(self, event, x, y) -> bool:
        """Obsługa zdarzeń myszy w trybie edycji"""
        if not isinstance(self.edit_mode, RowEditMode):
            return False

        handlers = {
            cv2.EVENT_LBUTTONDOWN: self._handle_left_click,
            cv2.EVENT_MOUSEMOVE: self._handle_mouse_move,
            cv2.EVENT_LBUTTONUP: self._handle_left_release
        }

        if event in handlers:
            return handlers[event](x, y)
        return False

    def draw_rows(self, image: np.ndarray) -> None:
        """Rysowanie linii wierszy na obrazie"""
        if image is None or len(image.shape) < 2:
            return

        for row in self.rows:
            if not row.p1 or not row.p2:
                continue

            color = row.color
            thickness = 3 if row == self.selected_row else 2

            h, w = image.shape[:2]
            p1 = (int(np.clip(row.p1[0], 0, w - 1)), int(np.clip(row.p1[1], 0, h - 1)))
            p2 = (int(np.clip(row.p2[0], 0, w - 1)), int(np.clip(row.p2[1], 0, h - 1)))

            cv2.line(image, p1, p2, color, thickness)

            if row == self.selected_row:
                cv2.circle(image, p1, 8, (255, 0, 0), -1)
                cv2.circle(image, p2, 8, (255, 0, 0), -1)

    def assign_boxes_to_selected_row(self) -> None:
        """Przypisuje zaznaczone boxy do aktualnie wybranego wiersza"""
        if not self.selected_row:
            return

        # Pobierz wszystkie boxy z manager'a
        all_boxes = self.bbox_manager.get_boxes()

        # Przypisz tylko te boxy, które nie są jeszcze przypisane do innych wierszy
        used_boxes = set()
        for row in self.rows:
            used_boxes.update(row.boxes)

        for box in all_boxes:
            if box not in used_boxes and self._is_box_near_line(box, self.selected_row):
                self.selected_row.boxes.append(box)

    def get_cropped_boxes(self) -> List[BoundingBox]:
        """Zwraca boxy przypisane do wierszy"""
        cropped_boxes = []
        for row in self.rows:
            cropped_boxes.extend(row.boxes)
        return cropped_boxes

    def clear_rows(self) -> None:
        """Czyści wszystkie wiersze i przypisane boxy"""
        self.rows = []
        self._reset_selection()

    # Metody prywatne
    def _handle_left_click(self, x: int, y: int) -> bool:
        if self.edit_mode == RowEditMode.ADD:
            return self._add_new_line(x, y)
        elif self.edit_mode == RowEditMode.EDIT:
            return self._select_line_for_edit(x, y)
        return False

    def _handle_mouse_move(self, x: int, y: int) -> bool:
        if not self.selected_row or not self.drag_start or not self.drag_type:
            return False

        if self.drag_type == 'move':
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]

            self.selected_row.p1 = (self.selected_row.p1[0] + dx, self.selected_row.p1[1] + dy)
            self.selected_row.p2 = (self.selected_row.p2[0] + dx, self.selected_row.p2[1] + dy)
            self._update_line_from_points()
            self.drag_start = (x, y)
            return True

        elif self.drag_type == 'p1':
            self.selected_row.p1 = (x, y)
            self._update_line_from_points()
            self.drag_start = (x, y)
            return True

        elif self.drag_type == 'p2':
            self.selected_row.p2 = (x, y)
            self._update_line_from_points()
            self.drag_start = (x, y)
            return True

        return False

    def _handle_left_release(self, x: int, y: int) -> bool:
        if self.selected_row:
            self.selected_row.locked = True
            self._update_line_from_points()
            self._reset_selection()
            return True
        return False

    def _add_new_line(self, x: int, y: int) -> bool:
        new_row = RowLine(
            slope=0.0,
            intercept=float(y),
            boxes=[],
            id=str(uuid.uuid4()),
            p1=(x - 100, y),
            p2=(x + 100, y),
            color=(0, 255, 255))
        self.rows.append(new_row)
        self.selected_row = new_row
        self.drag_type = 'p2'
        self.drag_start = (x, y)
        return True

    def _select_line_for_edit(self, x: int, y: int) -> bool:
        closest_row = None
        min_dist = float('inf')

        for row in self.rows:
            if row.locked or not row.p1 or not row.p2:
                continue

            dist_p1 = np.hypot(x - row.p1[0], y - row.p1[1])
            dist_p2 = np.hypot(x - row.p2[0], y - row.p2[1])
            line_dist = self._distance_to_line(row.p1, row.p2, (x, y))

            if dist_p1 < 30 and dist_p1 < min_dist:
                min_dist = dist_p1
                closest_row = row
                self.drag_type = 'p1'
            elif dist_p2 < 30 and dist_p2 < min_dist:
                min_dist = dist_p2
                closest_row = row
                self.drag_type = 'p2'
            elif line_dist < 20 and line_dist < min_dist:
                min_dist = line_dist
                closest_row = row
                self.drag_type = 'move'

        if closest_row:
            self.selected_row = closest_row
            self.drag_start = (x, y)
            return True
        return False

    def _update_line_from_points(self) -> None:
        if not self.selected_row or not self.selected_row.p1 or not self.selected_row.p2:
            return

        x1, y1 = self.selected_row.p1
        x2, y2 = self.selected_row.p2

        if x1 == x2:
            self.selected_row.slope = float('inf')
            self.selected_row.intercept = x1
        else:
            self.selected_row.slope = (y2 - y1) / (x2 - x1)
            self.selected_row.intercept = y1 - self.selected_row.slope * x1

    def _reset_selection(self) -> None:
        self.selected_row = None
        self.drag_start = None
        self.drag_type = None

    def _distance_to_line(self, p1: Tuple[float, float], p2: Tuple[float, float],
                          point: Tuple[float, float]) -> float:
        """Oblicza odległość punktu od linii zdefiniowanej przez p1 i p2"""
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = point

        if x1 == x2:  # Linia pionowa
            return abs(x0 - x1)

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        return abs(A * x0 + B * y0 + C) / np.sqrt(A ** 2 + B ** 2)

    def _is_box_near_line(self, box: BoundingBox, row: RowLine) -> bool:
        """Sprawdza czy box jest blisko linii wiersza"""
        if not row.p1 or not row.p2:
            return False

        box_center_y = (box.y1 + box.y2) / 2
        line_y_at_box_center = row.slope * ((box.x1 + box.x2) / 2) + row.intercept

        # Dopuszczalna odległość to 20% wysokości boxa
        threshold = (box.y2 - box.y1) * 0.2
        return abs(box_center_y - line_y_at_box_center) < threshold

