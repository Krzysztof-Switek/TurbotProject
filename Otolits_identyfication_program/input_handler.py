from enum import Enum, auto
import cv2
import sys
from row_detector import RowEditMode


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
        self._show_initial_status()  # Pokaz status tylko raz przy starcie

    def _show_initial_status(self):
        """Pokazuje status tylko raz przy inicjalizacji"""
        status_lines = [
            "=" * 50,
            "Dostępne tryby pracy:",
            "[m] - Tryb manualny (dodawanie boxów)",
            "[v] - Tryb przesuwania boxów",
            "[r] - Tryb zmiany rozmiaru",
            "[d] - Tryb usuwania boxów",
            "[1] - Edycja istniejących linii",
            "[2] - Dodawanie nowej linii",
            "[0] - Wyłącz edycję linii",
            "[ESC] - Wyjście",
            "=" * 50
        ]
        print("\n".join(status_lines))

    def get_current_mode_text(self):
        """Generuje tekst do wyświetlenia na obrazie"""
        mode_text = f"Tryb: {self.mode.name}"
        if self.row_detector.edit_mode != RowEditMode.NONE:
            mode_text += f" | Edycja linii: {self.row_detector.edit_mode.name}"
        return mode_text

    def set_mode(self, mode):
        """Zmiana trybu pracy"""
        if isinstance(mode, Mode):
            old_mode = self.mode
            self.mode = mode

            if mode != Mode.DELETE:
                self.row_detector.set_edit_mode(RowEditMode.NONE)

            return old_mode != mode
        return False

    def keyboard_callback(self, key):
        """Obsługa zdarzeń klawiatury"""
        key_actions = {
            ord('m'): lambda: self.set_mode(Mode.MANUAL),
            ord('v'): lambda: self.set_mode(Mode.MOVE),
            ord('r'): lambda: self.set_mode(Mode.RESIZE),
            ord('d'): lambda: self.set_mode(Mode.DELETE),
            27: lambda: [cv2.destroyAllWindows(), sys.exit()],
            ord('1'): lambda: self._set_row_edit_mode(RowEditMode.EDIT),
            ord('2'): lambda: self._set_row_edit_mode(RowEditMode.ADD),
            ord('0'): lambda: self._set_row_edit_mode(RowEditMode.NONE)
        }

        if key in key_actions:
            return key_actions[key]()
        return False

    def _set_row_edit_mode(self, mode):
        """Ustawia tryb edycji linii"""
        self.row_detector.set_edit_mode(mode)
        return True

    def _set_edit_mode(self, mode):
        """Ustawia tryb edycji linii i wymusza aktualizację statusu"""
        self.row_detector.set_edit_mode(mode)
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy"""
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
        """Resetowanie stanu do domyślnego"""
        self.mode = Mode.AUTO
        self.row_detector.set_edit_mode(RowEditMode.NONE)
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        self._update_status()

    def _update_status(self):
        """Aktualizacja statusu z wyraźnym wskazaniem aktywnych trybów"""
        # Definicje stylów
        ACTIVE_STYLE = "\033[1;32m"  # Pogrubiony zielony
        MODE_STYLE = "\033[1;34m"  # Pogrubiony niebieski
        RESET_STYLE = "\033[0m"  # Resetowanie stylów

        # Przygotowanie linii statusu
        status_lines = [
            "=" * 50,
            f"GŁÓWNY TRYB: {MODE_STYLE}{self.mode.name}{RESET_STYLE}",
            "",
            "EDYCJA LINII:",
            self._format_option("[1]", "Edytuj istniejące linie", RowEditMode.EDIT),
            self._format_option("[2]", "Dodaj nową linię", RowEditMode.ADD),
            self._format_option("[0]", "Wyłącz edycję linii", RowEditMode.NONE),
            "",
            "INNE TRYBY:",
            self._format_option("[m]", "Ręczne dodawanie boxów", Mode.MANUAL),
            self._format_option("[v]", "Przesuwanie boxów", Mode.MOVE),
            self._format_option("[r]", "Zmiana rozmiaru boxów", Mode.RESIZE),
            self._format_option("[d]", "Usuwanie boxów/linii", Mode.DELETE),
            "[ESC] Wyjście",
            "=" * 50
        ]

        # Czyszczenie konsoli i wyświetlenie statusu
        print("\033c", end="")  # Czyści konsolę
        print("\n".join(status_lines))

    def _format_option(self, prefix, text, mode_type):
        """Formatuje opcję menu z uwzględnieniem aktywności"""
        ACTIVE_STYLE = "\033[1;32m"
        RESET_STYLE = "\033[0m"

        if isinstance(mode_type, RowEditMode):
            is_active = self.row_detector.edit_mode == mode_type
        else:
            is_active = self.mode == mode_type

        if is_active:
            return f"{prefix} {ACTIVE_STYLE}{text.upper()} (AKTYWNY){RESET_STYLE}"
        return f"{prefix} {text}"

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