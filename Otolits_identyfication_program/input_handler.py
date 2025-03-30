from bounding_box import BoundingBox
from enum import Enum, auto
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


class WorkMode(Enum):
    AUTO = auto()  # Tryb automatyczny (domyślny)
    MANUAL = auto()  # Tryb manualny (aktywowany klawiszem 'm')


class ManualMode(Enum):
    BOX = auto()  # Edycja bounding boxów
    LINE = auto()  # Edycja linii wierszy


@dataclass
class SelectionContext:
    element: Optional['BoundingBox'] = None
    drag_start: Optional[Tuple[float, float]] = None
    is_drawing: bool = False


class InputHandler:
    def __init__(self, bbox_manager, row_detector):
        self.bbox_manager = bbox_manager
        self.row_detector = row_detector
        self.work_mode = WorkMode.AUTO  # Domyślnie tryb automatyczny
        self.manual_mode = ManualMode.BOX  # Domyślny tryb manualny
        self.selection = SelectionContext()

        # Mapa przycisków klawiatury
        self.key_bindings = {
            ord('m'): self._toggle_work_mode,  # Przełączanie trybu pracy
            ord('b'): lambda: self._set_manual_mode(ManualMode.BOX),
            ord('l'): lambda: self._set_manual_mode(ManualMode.LINE),
            8: self._delete_selected,  # Backspace - usuń zaznaczenie
            27: self._reset_selection  # ESC - anuluj
        }

    def keyboard_callback(self, key):
        if key in self.key_bindings:
            self.key_bindings[key]()
            print(f"Tryb: {'AUTO' if self.work_mode == WorkMode.AUTO else 'MANUAL'}")
            if self.work_mode == WorkMode.MANUAL:
                print(f"Tryb manualny: {'BOX' if self.manual_mode == ManualMode.BOX else 'LINE'}")
            return True
        return False

    def mouse_callback(self, event, x, y):
        if self.work_mode == WorkMode.AUTO:
            return False

        handlers = {
            cv2.EVENT_LBUTTONDOWN: self._handle_click,
            cv2.EVENT_MOUSEMOVE: self._handle_drag,
            cv2.EVENT_LBUTTONUP: self._handle_release
        }
        return handlers.get(event, lambda *_: False)(x, y)

    def _toggle_work_mode(self):
        """Przełączanie między trybem AUTO i MANUAL"""
        self.work_mode = WorkMode.MANUAL if self.work_mode == WorkMode.AUTO else WorkMode.AUTO
        self._reset_selection()

    def _set_manual_mode(self, mode):
        """Ustawia tryb manualny (BOX/LINE)"""
        if self.work_mode == WorkMode.MANUAL:
            self.manual_mode = mode
            self._reset_selection()

    def _handle_click(self, x, y):
        if self.work_mode != WorkMode.MANUAL:
            return False

        self.selection.is_drawing = True

        if self.manual_mode == ManualMode.BOX:
            self.selection.element = BoundingBox(x, y, x, y, is_temp=True)
        else:
            self.row_detector.start_new_line(x, y)

        self.selection.drag_start = (x, y)
        return True

    def _handle_drag(self, x, y):
        """Obsługa przeciągania myszą w trybie manualnym"""
        if not self.selection.is_drawing:
            return False

        if self.manual_mode == ManualMode.BOX and self.selection.element:
            # Aktualizuj współrzędne boxa
            self.selection.element.x2 = x
            self.selection.element.y2 = y
        elif self.manual_mode == ManualMode.LINE:
            self.row_detector.update_line_end(x, y)

        return True

    def _handle_release(self, x, y):
        if not self.selection.is_drawing:
            return False

        self.selection.is_drawing = False

        if self.manual_mode == ManualMode.BOX and self.selection.element:
            # Tworzymy finalny box po zwolnieniu przycisku
            temp_box = self.selection.element
            final_box = BoundingBox(
                min(temp_box.x1, x),
                min(temp_box.y1, y),
                max(temp_box.x1, x),
                max(temp_box.y1, y),
                is_temp=False
            )
            self.bbox_manager.add_box(final_box)

        elif self.manual_mode == ManualMode.LINE:
            self.row_detector.assign_boxes_to_current_line()

        self._reset_selection()
        return True

    def _delete_selected(self):
        """Usuwa zaznaczony element (box/linię)"""
        if self.work_mode == WorkMode.MANUAL:
            # W tej uproszczonej wersji usuwamy ostatni element
            if self.manual_mode == ManualMode.BOX and self.bbox_manager.boxes:
                self.bbox_manager.boxes.pop()
            elif self.manual_mode == ManualMode.LINE and self.row_detector.rows:
                self.row_detector.rows.pop()
        return True

    def _reset_selection(self):
        """Resetuje stan zaznaczenia"""
        self.selection = SelectionContext()