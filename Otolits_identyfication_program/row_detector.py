import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
import uuid
from enum import Enum, auto

class RowEditMode(Enum):
    NONE = auto()
    EDIT = auto()
    ADD = auto()

@dataclass
class RowLine:
    slope: float
    intercept: float
    boxes: List['BoundingBox']
    id: str
    p1: Optional[Tuple[float, float]] = None
    p2: Optional[Tuple[float, float]] = None

class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []
        self.edit_mode = RowEditMode.NONE
        self.selected_row: Optional[RowLine] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_type: Optional[str] = None
        self.line_extension = 0.3

    # Public methods
    def set_edit_mode(self, mode: RowEditMode):
        self.edit_mode = mode
        self._reset_selection()

    def detect_rows(self) -> List[RowLine]:
        self.rows = []
        if not self.bbox_manager.boxes:
            return self.rows

        boxes_sorted = sorted(self.bbox_manager.boxes, key=lambda b: (b.y1 + b.y2) / 2)
        current_group = []

        for box in boxes_sorted:
            if not current_group:
                current_group.append(box)
            else:
                last_box = current_group[-1]
                current_center = (box.y1 + box.y2) / 2
                last_center = (last_box.y1 + last_box.y2) / 2
                if abs(current_center - last_center) < 30:
                    current_group.append(box)
                else:
                    self._create_row_from_boxes(current_group)
                    current_group = [box]

        if current_group:
            self._create_row_from_boxes(current_group)

        return self.rows

    def handle_mouse_event(self, event, x, y) -> bool:
        if self.edit_mode == RowEditMode.NONE:
            return False

        handlers = {
            cv2.EVENT_LBUTTONDOWN: self._handle_left_click,
            cv2.EVENT_MOUSEMOVE: self._handle_mouse_move,
            cv2.EVENT_LBUTTONUP: self._handle_left_release
        }

        if event in handlers:
            return handlers[event](x, y)
        return False

    def draw_rows(self, image):
        if image is None:
            return

        for row in self.rows:
            if not row.p1 or not row.p2:
                continue

            color = (0, 255, 255) if row == self.selected_row else (0, 0, 255)
            thickness = 2

            p1 = (int(max(0, min(row.p1[0], image.shape[1]))), int(max(0, min(row.p1[1], image.shape[0]))))
            p2 = (int(max(0, min(row.p2[0], image.shape[1]))), int(max(0, min(row.p2[1], image.shape[0]))))

            cv2.line(image, p1, p2, color, thickness)

            if row == self.selected_row:
                cv2.circle(image, p1, 8, (255, 0, 0), -1)
                cv2.circle(image, p2, 8, (255, 0, 0), -1)

    # Private methods
    def _reset_selection(self):
        self.selected_row = None
        self.drag_start = None
        self.drag_type = None

    def _create_row_from_boxes(self, boxes: List['BoundingBox']):
        if not boxes:
            return

        if len(boxes) == 1:
            box = boxes[0]
            new_row = RowLine(
                slope=0.0,
                intercept=(box.y1 + box.y2) / 2,
                boxes=boxes.copy(),
                id=str(uuid.uuid4())
            )
        else:
            x_centers = [(b.x1 + b.x2) / 2 for b in boxes]
            y_centers = [(b.y1 + b.y2) / 2 for b in boxes]
            A = np.vstack([x_centers, np.ones(len(x_centers))]).T
            slope, intercept = np.linalg.lstsq(A, y_centers, rcond=None)[0]
            new_row = RowLine(slope, intercept, boxes.copy(), str(uuid.uuid4()))

        self._update_line_endpoints(new_row)
        self.rows.append(new_row)

    def _handle_left_click(self, x: int, y: int) -> bool:
        if self.edit_mode == RowEditMode.ADD:
            return self._add_new_line(x, y)
        elif self.edit_mode == RowEditMode.EDIT:
            return self._select_line_for_edit(x, y)
        return False

    def _handle_mouse_move(self, x: int, y: int) -> bool:
        if not self.selected_row or not self.drag_start or not self.drag_type:
            return False

        dx, dy = x - self.drag_start[0], y - self.drag_start[1]

        if self.drag_type == 'move':
            self.selected_row.intercept += dy
            self._update_line_endpoints(self.selected_row)
        elif self.drag_type == 'p1':
            self.selected_row.p1 = (x, y)
        elif self.drag_type == 'p2':
            self.selected_row.p2 = (x, y)

        if self.drag_type in ['p1', 'p2']:
            self._update_line_from_points()

        self.drag_start = (x, y)
        return True

    def _handle_left_release(self, x: int, y: int) -> bool:
        if self.selected_row:
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
            p2=(x + 100, y)
        )
        self.rows.append(new_row)
        self.selected_row = new_row
        self.drag_type = 'p2'
        self.drag_start = (x, y)
        return True

    def _select_line_for_edit(self, x: int, y: int) -> bool:
        closest_row = None
        min_dist = float('inf')

        for row in self.rows:
            if not row.p1 or not row.p2:
                continue

            dist_p1 = np.sqrt((x - row.p1[0]) ** 2 + (y - row.p1[1]) ** 2)
            dist_p2 = np.sqrt((x - row.p2[0]) ** 2 + (y - row.p2[1]) ** 2)
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

    def _update_line_endpoints(self, row: RowLine):
        if not row.boxes:
            if not row.p1 or not row.p2:
                row.p1 = (0, row.intercept)
                row.p2 = (1000, row.intercept)
            return

        x_coords = [b.x1 for b in row.boxes] + [b.x2 for b in row.boxes]
        min_x, max_x = min(x_coords), max(x_coords)
        width = max_x - min_x

        extended_min = min_x - self.line_extension * width
        extended_max = max_x + self.line_extension * width

        if abs(row.slope) < 0.01:
            row.p1 = (extended_min, row.intercept)
            row.p2 = (extended_max, row.intercept)
        else:
            row.p1 = (extended_min, row.slope * extended_min + row.intercept)
            row.p2 = (extended_max, row.slope * extended_max + row.intercept)

    def _update_line_from_points(self):
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

        self._update_line_endpoints(self.selected_row)

    def _distance_to_line(self, p1, p2, point):
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = point

        if x1 == x2:
            return abs(x0 - x1)

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        return abs(A * x0 + B * y0 + C) / np.sqrt(A ** 2 + B ** 2)