from enum import Enum, auto
import cv2
import sys

class Mode(Enum):
    AUTO = auto()
    MANUAL = auto()
    MOVE = auto()
    RESIZE = auto()
    DELETE = auto()

class InputHandler:
    def __init__(self, bounding_box_manager, row_manager):
        self.bbox_manager = bounding_box_manager
        self.row_manager = row_manager
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
            self.mode = mode
            print(f"Aktywny tryb: {mode.name}")

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
            self.set_mode(key_to_mode[key])