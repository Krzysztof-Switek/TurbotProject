from enum import Enum, auto
import cv2
import sys
from row_detector import RowEditMode  # Zmieniony import

class Mode(Enum):
    AUTO = auto()
    MANUAL = auto()
    MOVE = auto()
    RESIZE = auto()
    DELETE = auto()

class InputHandler:
    def __init__(self, bounding_box_manager, row_detector):
        self.bbox_manager = bounding_box_manager
        self.row_detector = row_detector
        self.mode = Mode.AUTO
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        self._update_status()

    # Public methods
    def set_mode(self, mode):
        if isinstance(mode, Mode):
            old_mode = self.mode
            self.mode = mode

            if mode != Mode.DELETE:
                self.row_detector.set_edit_mode(RowEditMode.NONE)

            self._update_status()
            return old_mode != mode
        return False

    def keyboard_callback(self, key):
        key_mappings = {
            ord('m'): Mode.MANUAL,
            ord('v'): Mode.MOVE,
            ord('r'): Mode.RESIZE,
            ord('d'): Mode.DELETE,
            27: 'exit',
            ord('1'): RowEditMode.EDIT,
            ord('2'): RowEditMode.ADD,
            ord('0'): RowEditMode.NONE
        }

        if key in key_mappings:
            action = key_mappings[key]

            if action == 'exit':
                cv2.destroyAllWindows()
                sys.exit()
            elif isinstance(action, RowEditMode):
                self.row_detector.set_edit_mode(action)
                self._update_status()
                return True
            else:
                return self.set_mode(action)

        return False

    def mouse_callback(self, event, x, y, flags, param):
        mode_handlers = {
            Mode.MANUAL: self._handle_manual_mode,
            Mode.MOVE: self._handle_move_mode,
            Mode.RESIZE: self._handle_resize_mode,
            Mode.DELETE: self._handle_delete_mode
        }

        if self.mode in mode_handlers:
            return mode_handlers[self.mode](event, x, y)
        return False

    def reset(self):
        self.mode = Mode.AUTO
        self.row_detector.set_edit_mode(RowEditMode.NONE)
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        self._update_status()

    # Private methods
    def _update_status(self):
        status_lines = [
            "=" * 50,
            f"GŁÓWNY TRYB: {self.mode.name}",
            "",
            "EDYCJA LINII:",
            f"[1] {'Tryb edycji linii (AKTYWNY)' if self.row_detector.edit_mode == RowEditMode.EDIT else 'Edytuj istniejące linie'}",
            f"[2] {'Tryb dodawania linii (AKTYWNY)' if self.row_detector.edit_mode == RowEditMode.ADD else 'Dodaj nową linię'}",
            "[0] Wyłącz edycję linii",
            "",
            "INNE TRYBY:",
            "[m] Ręczne dodawanie boxów",
            "[v] Przesuwanie boxów",
            "[r] Zmiana rozmiaru boxów",
            "[d] Usuwanie boxów/linii",
            "[ESC] Wyjście",
            "=" * 50
        ]

        print("\n" * 2 + "\n".join(status_lines) + "\n" * 2)

    def _handle_manual_mode(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pos = (x, y)
            self.current_pos = (x, y)
            self.drawing = True
            return True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_pos = (x, y)
            return True

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            x1, y1 = self.start_pos
            x2, y2 = x, y
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.bbox_manager.add_box(x1, y1, x2, y2)

            self.drawing = False
            self.start_pos = None
            self.current_pos = None
            return True

        return False

    def _handle_move_mode(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            box = self.bbox_manager.get_box_at(x, y, tolerance=5)
            if box:
                self.selected_box = box
                self.drag_offset = (x - box.x1, y - box.y1)
                return True

        elif event == cv2.EVENT_MOUSEMOVE and self.selected_box:
            dx = x - self.selected_box.x1 - self.drag_offset[0]
            dy = y - self.selected_box.y1 - self.drag_offset[1]
            self.selected_box.move(dx, dy)
            return True

        elif event == cv2.EVENT_LBUTTONUP and self.selected_box:
            self.selected_box = None
            return True

        return False

    def _handle_resize_mode(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            box = self.bbox_manager.get_box_at(x, y, tolerance=10)
            if box:
                self.selected_box = box
                self.drag_corner = box.get_nearest_corner(x, y)
                return True

        elif event == cv2.EVENT_MOUSEMOVE and self.selected_box:
            self.selected_box.resize(self.drag_corner, x, y)
            return True

        elif event == cv2.EVENT_LBUTTONUP and self.selected_box:
            self.selected_box = None
            return True

        return False

    def _handle_delete_mode(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            box = self.bbox_manager.get_box_at(x, y, tolerance=5)
            if box:
                self.bbox_manager.remove_box(box)
                return True
        return False