from enum import Enum, auto
import cv2
import sys
from row_detector import RowEditMode


class Mode(Enum):
    AUTO = auto()  # Tryb automatyczny (domyślny)
    MANUAL = auto()  # Ręczne dodawanie boxów
    MOVE = auto()  # Przesuwanie boxów
    RESIZE = auto()  # Zmiana rozmiaru boxów
    DELETE = auto()  # Usuwanie boxów


class InputHandler:
    def __init__(self, bounding_box_manager, row_manager):
        self.bbox_manager = bounding_box_manager
        self.row_detector = row_manager
        self.mode = Mode.AUTO
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        print("Aktywny tryb: AUTO")

    def set_mode(self, mode):
        """Zmiana trybu pracy"""
        if isinstance(mode, Mode):
            old_mode = self.mode
            self.mode = mode
            print(f"Aktywny tryb: {mode.name}")
            return old_mode != mode  # Zwraca True tylko jeśli tryb się zmienił
        return False

    def keyboard_callback(self, key):
        """Obsługa zdarzeń klawiatury"""
        key_to_mode = {
            ord('m'): Mode.MANUAL,
            ord('v'): Mode.MOVE,
            ord('r'): Mode.RESIZE,
            ord('d'): Mode.DELETE,
            27: 'exit'  # ESC
        }

        if key in key_to_mode:
            if key_to_mode[key] == 'exit':
                cv2.destroyAllWindows()
                sys.exit()
            return self.set_mode(key_to_mode[key])

        # Obsługa trybów edycji wierszy
        if not hasattr(self.row_detector, 'edit_mode'):
            return False

        try:
            if key == ord('1'):  # Tryb przesuwania wierszy
                self.row_detector.edit_mode = RowEditMode.MOVE
                return True
            elif key == ord('2'):  # Tryb obracania wierszy
                self.row_detector.edit_mode = RowEditMode.ROTATE
                return True
            elif key == ord('3'):  # Tryb dodawania wierszy
                self.row_detector.edit_mode = RowEditMode.ADD
                return True
            elif key == ord('4'):  # Tryb usuwania wierszy
                self.row_detector.edit_mode = RowEditMode.DELETE
                return True
            elif key == ord('0'):  # Wyłącz edycję wierszy
                self.row_detector.edit_mode = RowEditMode.NONE
                return True
        except Exception as e:
            print(f"Błąd zmiany trybu wierszy: {e}")
            return False

        return False

    def reset(self):
        """Resetuje stan handlera do domyślnego"""
        self.mode = Mode.AUTO
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        print("Zresetowano do trybu AUTO")