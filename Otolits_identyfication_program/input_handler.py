from enum import Enum, auto
import cv2
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from row_detector import RowLine
from bounding_box import BoundingBox
from math import hypot

class WorkMode(Enum):
    AUTO = auto()  # Tryb automatyczny (domyślny)
    MANUAL = auto()  # Tryb manualny


class ManualMode(Enum):
    ADD_BOX = auto()  # Dodawanie boxów
    ADD_LINE = auto()  # Dodawanie linii
    DELETE = auto()  # Usuwanie elementów
    MOVE = auto()  # Przesuwanie elementów
    RESIZE = auto()  # Zmiana rozmiaru


@dataclass
class SelectionContext:
    element: Optional[object] = None
    drag_start: Optional[Tuple[int, int]] = None
    is_drawing: bool = False
    corner_idx: Optional[Union[int, str]] = None  # Dla boxów: 0-3, dla linii: 'p1'/'p2'


class InputHandler:
    def __init__(self, bbox_manager, row_detector):
        self.bbox_manager = bbox_manager
        self.row_detector = row_detector
        self.work_mode = WorkMode.AUTO
        self.manual_mode = ManualMode.ADD_BOX
        self.selection = SelectionContext()

        # Uproszczona mapa klawiszy
        self.key_bindings = {
            ord('a'): lambda: self._set_work_mode(WorkMode.AUTO),
            ord('m'): lambda: self._set_work_mode(WorkMode.MANUAL),
            ord('b'): lambda: self._set_manual_mode(ManualMode.ADD_BOX),
            ord('l'): lambda: self._set_manual_mode(ManualMode.ADD_LINE),
            ord('d'): lambda: self._set_manual_mode(ManualMode.DELETE),
            ord('v'): lambda: self._set_manual_mode(ManualMode.MOVE),
            ord('r'): lambda: self._set_manual_mode(ManualMode.RESIZE),
            27: self._reset_selection  # ESC
        }

    def get_mode_info(self) -> str:
        """Zwraca tekst informujący o aktualnym trybie"""
        mode_info = f"Tryb: {'AUTO' if self.work_mode == WorkMode.AUTO else 'MANUAL'}"

        if self.work_mode == WorkMode.MANUAL:
            mode_names = {
                ManualMode.ADD_BOX: "Dodawanie boxów",
                ManualMode.ADD_LINE: "Dodawanie linii",
                ManualMode.DELETE: "Usuwanie",
                ManualMode.MOVE: "Przesuwanie",
                ManualMode.RESIZE: "Zmiana rozmiaru"  # Dodajemy nowy tryb
            }
            mode_info += f" | {mode_names[self.manual_mode]}"

        return mode_info

    def keyboard_callback(self, key: int) -> bool:
        """Obsługa zdarzeń klawiatury"""
        if key in self.key_bindings:
            self.key_bindings[key]()
            print(f"\n{self.get_mode_info()}")
            return True
        return False

    def mouse_callback(self, event, x, y) -> bool:
        """Obsługa zdarzeń myszy"""
        if self.work_mode == WorkMode.AUTO:
            return False

        handlers = {
            cv2.EVENT_LBUTTONDOWN: self._handle_left_down,
            cv2.EVENT_MOUSEMOVE: self._handle_mouse_move,
            cv2.EVENT_LBUTTONUP: self._handle_left_up
        }
        return handlers.get(event, lambda *_: False)(x, y)

    def _handle_left_down(self, x: int, y: int) -> bool:
        self.selection.is_drawing = True

        if self.manual_mode == ManualMode.ADD_BOX:
            self.selection.element = BoundingBox(x, y, x, y, is_temp=True)
        elif self.manual_mode == ManualMode.ADD_LINE:
            self.row_detector.start_new_line(x, y)
        elif self.manual_mode == ManualMode.DELETE:
            if box := self.bbox_manager.get_box_at(x, y):
                self.bbox_manager.remove_box(box)
                return True
            elif line := self.row_detector.get_line_at(x, y):
                self.row_detector.remove_line(line)
                return True
        elif self.manual_mode == ManualMode.MOVE:
            if box := self.bbox_manager.get_box_at(x, y):
                self.selection.element = box
            elif line := self.row_detector.get_line_at(x, y):
                if isinstance(line, RowLine):
                    self.selection.element = line
        elif self.manual_mode == ManualMode.RESIZE:  # Nowy tryb
            if box := self.bbox_manager.get_box_at(x, y):
                self.selection.element = box
                self.selection.corner_idx = box.get_nearest_corner(x, y)
            elif line := self.row_detector.get_line_at(x, y):
                # Sprawdź który koniec linii jest bliżej
                dist_p1 = hypot(x - line.p1[0], y - line.p1[1])
                dist_p2 = hypot(x - line.p2[0], y - line.p2[1])
                self.selection.element = line
                self.selection.corner_idx = 'p1' if dist_p1 < dist_p2 else 'p2'

        self.selection.drag_start = (x, y)
        return True

    def _handle_mouse_move(self, x: int, y: int) -> bool:

        if not self.selection.is_drawing:
            return False

        if self.manual_mode == ManualMode.ADD_LINE:
            self.row_detector.update_line_end(x, y)
            return True

        if not self.selection.element:
            return False

        dx, dy = x - self.selection.drag_start[0], y - self.selection.drag_start[1]
        self.selection.drag_start = (x, y)

        if self.manual_mode == ManualMode.MOVE:
            if isinstance(self.selection.element, RowLine):
                self.selection.element.move(dx, dy)

            elif isinstance(self.selection.element, BoundingBox):
                self.selection.element.move(dx, dy)

        elif self.manual_mode == ManualMode.RESIZE:
            if isinstance(self.selection.element, RowLine):
                if self.selection.corner_idx == 'p1':
                    self.selection.element.p1 = (float(x), float(y))
                elif self.selection.corner_idx == 'p2':
                    self.selection.element.p2 = (float(x), float(y))
            elif isinstance(self.selection.element, BoundingBox):
                self.selection.element.resize_corner(self.selection.corner_idx, x, y)

        return True

    def _handle_left_up(self, x: int, y: int) -> bool:
        """Obsługa zwolnienia LPM"""
        if not self.selection.is_drawing:
            return False

        self.selection.is_drawing = False

        if self.manual_mode == ManualMode.ADD_BOX and self.selection.element:
            x1, y1 = self.selection.element.x1, self.selection.element.y1
            x2, y2 = x, y

            # Tylko jeśli box ma dodatnią powierzchnię
            if x1 != x2 and y1 != y2:
                final_box = BoundingBox(
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                    is_temp=False
                )
                self.bbox_manager.add_box(final_box)

        elif self.manual_mode == ManualMode.ADD_LINE:
            self.row_detector.finish_line()

        self._reset_selection()
        return True

    def _reset_selection(self) -> None:
        """Resetuje stan zaznaczenia"""
        self.selection = SelectionContext()

    def _set_work_mode(self, mode: WorkMode) -> None:
        """Ustawia tryb pracy (AUTO/MANUAL)"""
        self.work_mode = mode
        self._reset_selection()
        print(f"Aktywny tryb: {'AUTO' if mode == WorkMode.AUTO else 'MANUAL'}")

    def _set_manual_mode(self, mode: ManualMode) -> None:
        """Ustawia tryb manualny"""
        if self.work_mode == WorkMode.MANUAL:
            self.manual_mode = mode
            self._reset_selection()
            print(f"Tryb manualny: {mode.name}")