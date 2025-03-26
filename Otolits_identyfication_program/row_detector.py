import numpy as np
import cv2
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import uuid


class RowEditMode(Enum):
    NONE = 0
    MOVE = 1
    ROTATE = 2
    ADD = 3
    DELETE = 4


@dataclass
class RowLine:
    slope: float
    intercept: float
    boxes: List['BoundingBox']
    id: str  # Unikalny identyfikator


class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []
        self.edit_mode = RowEditMode.NONE
        self.selected_row: Optional[RowLine] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.used_boxes: Set[str] = set()

    def detect_rows(self) -> List[RowLine]:
        """Iteracyjna metoda wykrywająca niezależne wiersze od góry do dołu"""
        self.rows = []
        self.used_boxes = set()
        remaining_boxes = [b for b in self.bbox_manager.boxes if b.id not in self.used_boxes]

        while remaining_boxes:
            remaining_boxes.sort(key=lambda b: b.y1)
            candidate_boxes = self._select_candidate_boxes(remaining_boxes)

            if not candidate_boxes:
                break

            line = self._fit_line_to_boxes(candidate_boxes)
            row_boxes = self._find_boxes_for_line(line, remaining_boxes)

            if row_boxes:
                line.id = str(uuid.uuid4())
                line.boxes = row_boxes
                self.rows.append(line)
                self.used_boxes.update(box.id for box in row_boxes)
                remaining_boxes = [b for b in remaining_boxes if b.id not in self.used_boxes]
            else:
                single_box = remaining_boxes.pop(0)
                self.rows.append(RowLine(
                    slope=0.0,
                    intercept=(single_box.y1 + single_box.y2) / 2,
                    boxes=[single_box],
                    id=str(uuid.uuid4())
                ))
                self.used_boxes.add(single_box.id)

        return self.rows

    def set_edit_mode(self, mode: RowEditMode):
        """Ustaw tryb edycji wierszy"""
        self.edit_mode = mode
        self.selected_row = None
        self.drag_start = None

    def handle_mouse_event(self, event, x, y) -> bool:
        """Obsługa zdarzeń myszy w trybie edycji"""
        if self.edit_mode == RowEditMode.NONE:
            return False

        if event == cv2.EVENT_LBUTTONDOWN:
            return self._handle_left_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            return self._handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            return self._handle_left_release(x, y)
        return False

    def _handle_left_click(self, x: int, y: int) -> bool:
        if self.edit_mode == RowEditMode.ADD:
            new_row = RowLine(
                slope=0.0,
                intercept=float(y),
                boxes=[],
                id=str(uuid.uuid4())
            )
            self._assign_boxes_to_row(new_row)
            self.rows.append(new_row)
            return True

        # Znajdź najbliższy wiersz
        closest_row = None
        min_dist = float('inf')

        for row in self.rows:
            dist = self._distance_to_row(row, x, y)
            if dist < min_dist and dist < 20:  # Próg 20 pikseli
                min_dist = dist
                closest_row = row

        if closest_row:
            self.selected_row = closest_row
            self.drag_start = (x, y)
            return True
        return False

    def _handle_mouse_move(self, x: int, y: int) -> bool:
        if not self.selected_row or not self.drag_start:
            return False

        dx = x - self.drag_start[0]
        dy = y - self.drag_start[1]

        if self.edit_mode == RowEditMode.MOVE:
            self.selected_row.intercept += dy
        elif self.edit_mode == RowEditMode.ROTATE:
            self.selected_row.slope += dx * 0.01  # Bardziej intuicyjna zmiana nachylenia

        self._assign_boxes_to_row(self.selected_row)
        self.drag_start = (x, y)
        return True

    def _handle_left_release(self, x: int, y: int) -> bool:
        if self.edit_mode == RowEditMode.DELETE and self.selected_row:
            self.rows.remove(self.selected_row)
            self.selected_row = None
            return True

        self.selected_row = None
        self.drag_start = None
        return False

    def _select_candidate_boxes(self, boxes: List['BoundingBox'], threshold: float = 30.0) -> List['BoundingBox']:
        """Wybierz grupę kandydatów na wiersz"""
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
        """Dopasuj linię do grupy boxów"""
        if len(boxes) == 1:
            box = boxes[0]
            return RowLine(0.0, (box.y1 + box.y2) / 2, boxes, str(uuid.uuid4()))

        x_centers = [(b.x1 + b.x2) / 2 for b in boxes]
        y_centers = [(b.y1 + b.y2) / 2 for b in boxes]

        A = np.vstack([x_centers, np.ones(len(x_centers))]).T
        slope, intercept = np.linalg.lstsq(A, y_centers, rcond=None)[0]

        return RowLine(slope, intercept, boxes, str(uuid.uuid4()))

    def _find_boxes_for_line(self, line: RowLine, boxes: List['BoundingBox'], threshold: float = 15.0) -> List[
        'BoundingBox']:
        """Znajdź boxy należące do linii"""
        row_boxes = []
        for box in boxes:
            if box.id in self.used_boxes:
                continue

            x_center = (box.x1 + box.x2) / 2
            y_center = (box.y1 + box.y2) / 2
            line_y = line.slope * x_center + line.intercept
            if abs(y_center - line_y) < threshold:
                row_boxes.append(box)
        return row_boxes

    def _distance_to_row(self, row: RowLine, x: int, y: int) -> float:
        """Oblicz odległość punktu od linii wiersza"""
        if not row.boxes:
            return abs(y - row.intercept)

        x_centers = [(b.x1 + b.x2) / 2 for b in row.boxes]
        min_x, max_x = min(x_centers), max(x_centers)

        if x < min_x:
            row_y = row.slope * min_x + row.intercept
        elif x > max_x:
            row_y = row.slope * max_x + row.intercept
        else:
            row_y = row.slope * x + row.intercept

        return abs(y - row_y)

    def _assign_boxes_to_row(self, row: RowLine, threshold: float = 15.0):
        """Przypisz boxy do wiersza"""
        # Usuń boxy z innych wierszy
        for other_row in self.rows:
            if other_row != row:
                other_row.boxes = [b for b in other_row.boxes
                                   if abs((b.y1 + b.y2) / 2 - (
                                other_row.slope * (b.x1 + b.x2) / 2 + other_row.intercept)) < threshold]

        # Przypisz nowe boxy
        row.boxes = []
        for box in self.bbox_manager.boxes:
            x_center = (box.x1 + box.x2) / 2
            y_center = (box.y1 + box.y2) / 2
            line_y = row.slope * x_center + row.intercept
            if abs(y_center - line_y) < threshold:
                # Sprawdź czy nie należy już do innego wiersza
                in_other_row = False
                for other_row in self.rows:
                    if other_row != row and box in other_row.boxes:
                        other_dist = abs(y_center - (other_row.slope * x_center + other_row.intercept))
                        if other_dist < threshold and other_dist < abs(y_center - line_y):
                            in_other_row = True
                            break
                if not in_other_row:
                    row.boxes.append(box)

    def draw_rows(self, image):
        """Rysuj linie wierszy i informacje"""
        if image is None or not hasattr(image, 'shape'):
            return

        # Rysuj linie wierszy
        for row in self.rows:
            if not row.boxes:  # Pomijaj puste wiersze
                continue

            color = (0, 255, 255) if row == self.selected_row else (0, 0, 255)
            thickness = 3 if row == self.selected_row else 2

            try:
                x_coords = [b.x1 for b in row.boxes] + [b.x2 for b in row.boxes]
                min_x = max(0, min(x_coords))  # Ogranicz do zakresu obrazu
                max_x = min(image.shape[1], max(x_coords))

                if abs(row.slope) < 0.01:  # Linia pozioma
                    y = int(row.intercept)
                    y = max(0, min(y, image.shape[0]))  # Ogranicz Y do wysokości obrazu
                    cv2.line(image, (int(min_x), y), (int(max_x), y), color, thickness)
                else:
                    y1 = int(row.slope * min_x + row.intercept)
                    y2 = int(row.slope * max_x + row.intercept)
                    y1 = max(0, min(y1, image.shape[0]))
                    y2 = max(0, min(y2, image.shape[0]))
                    cv2.line(image, (int(min_x), y1), (int(max_x), y2), color, thickness)
            except (ValueError, TypeError) as e:
                print(f"Błąd rysowania wiersza {row.id}: {e}")
                continue

        # Informacje o liczbie wierszy i trybie
        info_text = f"Wiersze: {len(self.rows)} | Tryb: {self.edit_mode.name}"
        cv2.putText(image, info_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)