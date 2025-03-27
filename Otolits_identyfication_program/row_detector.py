import numpy as np
import cv2
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
import uuid
from enum import Enum, auto


class RowEditMode(Enum):
    NONE = auto()
    EDIT = auto()
    ADD = auto()


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    text: str = ""
    confidence: float = 0.0
    id: str = ""


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
        self.selected_row: Optional[RowLine] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_type: Optional[str] = None

        # Parametry konfiguracyjne
        self.line_extension_factor = 0.2  # 20% rozszerzenia linii
        self.min_line_length = 50  # Minimalna długość linii
        self.max_slope = 0.1  # Maksymalne dopuszczalne nachylenie
        self.y_grouping_threshold = 0.4  # 40% wysokości boxu
        self.x_grouping_threshold = 1.5  # 1.5 szerokości boxu
        self.debug_mode = False  # Tryb debugowania

    def set_edit_mode(self, mode: RowEditMode) -> bool:
        """Ustawia tryb edycji linii. Zwraca True jeśli zmiana się powiodła."""
        if isinstance(mode, RowEditMode):
            self.edit_mode = mode
            self._reset_selection()
            return True
        return False

    def detect_rows(self) -> List[RowLine]:
        """
        Główna metoda wykrywająca wiersze. Algorytm:
        1. Sortuje boxy od góry do dołu obrazu
        2. Grupuje boxy w wiersze na podstawie odległości
        3. Dla każdej grupy oblicza linię metodą najmniejszych kwadratów
        4. Gwarantuje, że każdy box należy do dokładnie jednego wiersza
        5. Zapobiega przecięciom linii
        """
        self.rows = [row for row in self.rows if row.locked]  # Zachowaj zamrożone linie
        used_boxes: Set[BoundingBox] = set()

        # Zbierz wszystkie boxy nieprzypisane do zamrożonych linii
        all_boxes = set(self.bbox_manager.boxes)
        for row in self.rows:
            used_boxes.update(row.boxes)

        remaining_boxes = sorted(
            [b for b in all_boxes if b not in used_boxes],
            key=lambda b: ((b.y1 + b.y2) / 2, b.x1)  # Sortuj Y-potem-X
        )

        if not remaining_boxes and not self.rows:
            return self.rows

        # Oblicz średnie rozmiary boxów dla dynamicznych progów
        avg_height = np.mean([b.y2 - b.y1 for b in remaining_boxes]) if remaining_boxes else 30
        avg_width = np.mean([b.x2 - b.x1 for b in remaining_boxes]) if remaining_boxes else 30

        y_threshold = avg_height * self.y_grouping_threshold
        x_threshold = avg_width * self.x_grouping_threshold

        # Faza 1: Grupowanie boxów w wiersze
        while remaining_boxes:
            current_box = remaining_boxes.pop(0)
            current_group = [current_box]

            # Szukaj sąsiadów w poziomie i pionie
            i = 0
            while i < len(remaining_boxes):
                box = remaining_boxes[i]
                last_in_group = current_group[-1]

                # Warunki grupowania
                y_condition = abs((box.y1 + box.y2) / 2 - (last_in_group.y1 + last_in_group.y2) / 2) < y_threshold
                x_condition = (box.x1 - last_in_group.x2) < x_threshold

                if y_condition and x_condition:
                    current_group.append(remaining_boxes.pop(i))
                else:
                    i += 1

            # Utwórz linię dla grupy
            if current_group:
                self._create_row_from_boxes(current_group)
                used_boxes.update(current_group)

        # Faza 2: Przypisz pozostałe boxy do najbliższych linii
        self._assign_remaining_boxes(used_boxes)

        if self.debug_mode:
            print(f"Debug: Stworzono {len(self.rows)} wierszy")
            for i, row in enumerate(self.rows):
                print(f"Wiersz {i}: boxy={len(row.boxes)}, slope={row.slope:.2f}")

        return self.rows

    def handle_mouse_event(self, event, x, y) -> bool:
        """Obsługa zdarzeń myszy w trybie edycji"""
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

    def _assign_remaining_boxes(self, used_boxes: Set[BoundingBox]) -> None:
        """Przypisuje pozostałe boxy do najbliższych istniejących linii."""
        remaining_boxes = [b for b in self.bbox_manager.boxes if b not in used_boxes]

        for box in remaining_boxes:
            if not self.rows:
                # Jeśli nie ma żadnych linii, utwórz nową dla tego boxa
                self._create_row_from_boxes([box])
                continue

            # Znajdź najbliższą nie-zamrożoną linię
            box_center_y = (box.y1 + box.y2) / 2
            closest_row = min(
                (r for r in self.rows if not r.locked),
                key=lambda r: abs((r.p1[1] + r.p2[1]) / 2 - box_center_y),
                default=None
            )

            if closest_row:
                closest_row.boxes.append(box)
                self._update_line_endpoints(closest_row)
                if self.debug_mode:
                    print(f"Debug: Przypisano box {box.id} do wiersza {closest_row.id}")
            else:
                # Jeśli wszystkie linie są zamrożone, utwórz nową
                self._create_row_from_boxes([box])


    def _handle_left_click(self, x: int, y: int) -> bool:
        """Obsługa pojedynczego kliknięcia myszą"""
        if self.edit_mode == RowEditMode.ADD:
            return self._add_new_line(x, y)
        elif self.edit_mode == RowEditMode.EDIT:
            return self._select_line_for_edit(x, y)
        return False

    def _handle_mouse_move(self, x: int, y: int) -> bool:
        """Obsługa ruchu myszą z wciśniętym przyciskiem"""
        if not self.selected_row or not self.drag_start or not self.drag_type:
            return False

        if self.drag_type == 'move':
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]

            temp_row = RowLine(
                slope=self.selected_row.slope,
                intercept=self.selected_row.intercept,
                boxes=self.selected_row.boxes.copy(),
                id=self.selected_row.id,
                p1=(self.selected_row.p1[0] + dx, self.selected_row.p1[1] + dy),
                p2=(self.selected_row.p2[0] + dx, self.selected_row.p2[1] + dy)
            )

            other_rows = [row for row in self.rows if row.id != self.selected_row.id]
            if not self._check_line_intersections(temp_row, other_rows):
                self.selected_row.p1 = (self.selected_row.p1[0] + dx, self.selected_row.p1[1] + dy)
                self.selected_row.p2 = (self.selected_row.p2[0] + dx, self.selected_row.p2[1] + dy)
                self._update_line_from_points()
                self.drag_start = (x, y)
                return True
            return False

        elif self.drag_type == 'p1':
            temp_row = RowLine(
                slope=self.selected_row.slope,
                intercept=self.selected_row.intercept,
                boxes=self.selected_row.boxes.copy(),
                id=self.selected_row.id,
                p1=(x, y),
                p2=self.selected_row.p2
            )

            other_rows = [row for row in self.rows if row.id != self.selected_row.id]
            if not self._check_line_intersections(temp_row, other_rows):
                self.selected_row.p1 = (x, y)
                self._update_line_from_points()
                self.drag_start = (x, y)
                return True
            return False

        elif self.drag_type == 'p2':
            temp_row = RowLine(
                slope=self.selected_row.slope,
                intercept=self.selected_row.intercept,
                boxes=self.selected_row.boxes.copy(),
                id=self.selected_row.id,
                p1=self.selected_row.p1,
                p2=(x, y)
            )

            other_rows = [row for row in self.rows if row.id != self.selected_row.id]
            if not self._check_line_intersections(temp_row, other_rows):
                self.selected_row.p2 = (x, y)
                self._update_line_from_points()
                self.drag_start = (x, y)
                return True
            return False

        return False

    def _handle_left_release(self, x: int, y: int) -> bool:
        """Obsługa zwolnienia przycisku myszy"""
        if self.selected_row:
            self.selected_row.locked = True
            self._update_line_from_points()
            self._reset_selection()
            return True
        return False

    def _add_new_line(self, x: int, y: int) -> bool:
        """Dodawanie nowej linii"""
        new_row = RowLine(
            slope=0.0,
            intercept=float(y),
            boxes=[],
            id=str(uuid.uuid4()),
            p1=(x - 100, y),
            p2=(x + 100, y),
            color=(0, 255, 255)  # Żółty dla nowo dodanych linii
        )
        self.rows.append(new_row)
        self.selected_row = new_row
        self.drag_type = 'p2'
        self.drag_start = (x, y)
        return True

    def _select_line_for_edit(self, x: int, y: int) -> bool:
        """Wybór linii do edycji z uwzględnieniem statusu locked"""
        closest_row = None
        min_dist = float('inf')

        for row in self.rows:
            if row.locked or not row.p1 or not row.p2:
                continue

            dist_p1 = np.hypot(x - row.p1[0], y - row.p1[1])  # Bezpieczniejsze niż sqrt
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
        """Aktualizacja parametrów linii na podstawie punktów końcowych"""
        if not self.selected_row or not self.selected_row.p1 or not self.selected_row.p2:
            return

        x1, y1 = self.selected_row.p1
        x2, y2 = self.selected_row.p2

        if x1 == x2:  # Linia pionowa
            self.selected_row.slope = float('inf')
            self.selected_row.intercept = x1
        else:
            self.selected_row.slope = (y2 - y1) / (x2 - x1)
            self.selected_row.intercept = y1 - self.selected_row.slope * x1

    def draw_rows(self, image: np.ndarray) -> None:
        """Rysowanie linii wierszy na obrazie"""
        if image is None or len(image.shape) < 2:
            return

        for row in self.rows:
            if not row.p1 or not row.p2:
                continue

            color = row.color
            thickness = 3 if row == self.selected_row else 2

            # Zabezpieczenie przed rysowaniem poza obrazem
            h, w = image.shape[:2]
            p1 = (int(np.clip(row.p1[0], 0, w - 1)), int(np.clip(row.p1[1], 0, h - 1)))
            p2 = (int(np.clip(row.p2[0], 0, w - 1)), int(np.clip(row.p2[1], 0, h - 1)))

            cv2.line(image, p1, p2, color, thickness)

            if row == self.selected_row:
                cv2.circle(image, p1, 8, (255, 0, 0), -1)
                cv2.circle(image, p2, 8, (255, 0, 0), -1)

    def _reset_selection(self) -> None:
        """Resetowanie stanu selekcji"""
        self.selected_row = None
        self.drag_start = None
        self.drag_type = None

    def _create_row_from_boxes(self, boxes: List[BoundingBox]) -> None:
        """
        Tworzy nową linię na podstawie grupy boxów.
        Wymusza przyjęcie linii poziomej jeśli nachylenie jest zbyt duże.
        """
        if not boxes:
            return

        # Oblicz parametry linii
        x_centers = [(b.x1 + b.x2) / 2 for b in boxes]
        y_centers = [(b.y1 + b.y2) / 2 for b in boxes]

        # Metoda najmniejszych kwadratów
        A = np.vstack([x_centers, np.ones(len(x_centers))]).T
        slope, intercept = np.linalg.lstsq(A, y_centers, rcond=None)[0]

        # Wymuś linię poziomą jeśli nachylenie zbyt duże
        if abs(slope) > self.max_slope:
            slope = 0.0
            intercept = np.median(y_centers)  # Median jest bardziej odporny na outliers

        new_row = RowLine(
            slope=slope,
            intercept=intercept,
            boxes=boxes.copy(),
            id=str(uuid.uuid4()),
            color=(0, 255, 0)  # Zielony dla automatycznych linii
        )

        self._update_line_endpoints(new_row)

        # Sprawdź przecięcia z istniejącymi liniami
        if not self._check_line_intersections(new_row, [r for r in self.rows if r.locked]):
            self.rows.append(new_row)
        elif self.debug_mode:
            print(f"Debug: Odrzucono linię z powodu przecięcia (ID: {new_row.id})")

    def _update_line_endpoints(self, row: RowLine) -> None:
        """Aktualizuje punkty końcowe linii na podstawie przypisanych boxów."""
        if not row.boxes:
            return

        x_coords = [b.x1 for b in row.boxes] + [b.x2 for b in row.boxes]
        min_x, max_x = min(x_coords), max(x_coords)
        extension = (max_x - min_x) * self.line_extension_factor

        if abs(row.slope) < 1e-6:  # Linia pozioma
            y = row.intercept
            row.p1 = (min_x - extension, y)
            row.p2 = (max_x + extension, y)
        else:  # Linia ukośna
            row.p1 = (min_x - extension, row.slope * (min_x - extension) + row.intercept)
            row.p2 = (max_x + extension, row.slope * (max_x + extension) + row.intercept)

    def _distance_to_line(self, p1: Tuple[float, float], p2: Tuple[float, float],
                          point: Tuple[float, float]) -> float:
        """Oblicza odległość punktu od linii zdefiniowanej przez p1 i p2"""
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

    def _check_line_intersections(self, line: RowLine, other_lines: List[RowLine]) -> bool:
        """Sprawdza czy linia przecina którąś z podanych linii"""
        if not line.p1 or not line.p2:
            return False

        for other_line in other_lines:
            if not other_line.p1 or not other_line.p2:
                continue

            if self._do_lines_intersect(line.p1, line.p2, other_line.p1, other_line.p2):
                return True
        return False

    def _do_lines_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                            p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Sprawdza czy dwa odcinki się przecinają"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = p1, p2
        C, D = p3, p4

        # Sprawdzenie czy odcinki się przecinają
        intersect = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # Ignorujemy przypadki gdy końce się stykają
        if (A == C or A == D or B == C or B == D):
            return False

        return intersect
