import cv2
from enum import Enum, auto
import numpy as np


class Mode(Enum):
    MANUAL = auto()
    DELETE = auto()
    MOVE = auto()
    RESIZE = auto()


class InputHandler:
    def __init__(self, bounding_box_manager, row_manager):
        self.bbox_manager = bounding_box_manager
        self.row_manager = row_manager
        self.mode = Mode.MANUAL
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None

    def set_mode(self, mode):
        """Zmiana trybu pracy z walidacją"""
        if isinstance(mode, Mode):
            self.mode = mode
            print(f"Aktywny tryb: {mode.name.lower()}")
        else:
            raise ValueError("Nieprawidłowy tryb pracy")

    def mouse_callback(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy z płynnym podglądem"""
        image = param['image'] if param and 'image' in param else None

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_pos = (x, y)
            self.drawing = True

            if self.mode == Mode.DELETE:
                box = self.bbox_manager.get_box_at(x, y, tolerance=5)
                if box:
                    self.bbox_manager.remove_box(box)
                    self._update_display(image)
            elif self.mode == Mode.MANUAL:
                self.temp_image = image.copy()  # Zachowaj oryginalny obraz

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.mode == Mode.MANUAL:
                # Przygotuj obraz z istniejącymi boxami + podglądem nowego
                display_image = self.temp_image.copy()

                # 1. Narysuj wszystkie istniejące boxy
                for box in self.bbox_manager.boxes:
                    cv2.rectangle(display_image,
                                  (box.x1, box.y1),
                                  (box.x2, box.y2),
                                  (0, 255, 0), 2)

                # 2. Dodaj podgląd aktualnie rysowanego boxa (czerwony)
                cv2.rectangle(display_image,
                              self.start_pos,
                              (x, y),
                              (0, 0, 255), 2)

                cv2.imshow("Otolith Annotation Tool", display_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.mode == Mode.MANUAL:
                x1, x2 = sorted([self.start_pos[0], x])
                y1, y2 = sorted([self.start_pos[1], y])
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.bbox_manager.add_box(x1, y1, x2, y2)
            self.drawing = False
            self._update_display(image)  # Odśwież końcowy obraz

    def _handle_left_click(self, x, y):
        self.start_pos = (x, y)
        if self.mode == Mode.MANUAL:
            self.drawing = True
            self.selected_box = None

    def _handle_left_release(self, x, y, image):
        if self.drawing and self.mode == Mode.MANUAL:
            x1, x2 = sorted([self.start_pos[0], x])
            y1, y2 = sorted([self.start_pos[1], y])
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Minimalny rozmiar
                self.bbox_manager.add_box(x1, y1, x2, y2)
                self._update_display(image)
        self._reset_state()

    def _handle_mouse_move(self, x, y, image):
        """Obsługa ruchu myszy"""
        if self.drawing and self.mode == Mode.MANUAL:
            temp_image = image.copy()
            cv2.rectangle(temp_image, self.start_pos, (x, y), (0, 255, 0), 2)
            self._update_display(temp_image)  # Używamy _update_display do pokazywania obrazu

        elif self.selected_box and self.mode == Mode.MOVE:
            dx = x - self.start_pos[0]
            dy = y - self.start_pos[1]
            temp_box = self.selected_box.get_coordinates()
            cv2.rectangle(image, (temp_box[0] + dx, temp_box[1] + dy),
                          (temp_box[2] + dx, temp_box[3] + dy), (255, 0, 0), 2)
            self._update_display(image)

        elif self.selected_box and self.mode == Mode.RESIZE:
            temp_image = image.copy()
            new_coords = self._get_resized_coords(x, y)
            cv2.rectangle(temp_image, (new_coords[0], new_coords[1]),
                          (new_coords[2], new_coords[3]), (0, 0, 255), 2)
            self._update_display(temp_image)

    def _handle_right_click(self, x, y):
        """Obsługa kliknięcia PPM - szybkie przełączanie trybów"""
        box = self.bbox_manager.get_box_at(x, y, tolerance=5)
        if box:
            self.selected_box = box
            self.set_mode(Mode.MOVE if self.mode != Mode.MOVE else Mode.MANUAL)

    def _get_nearest_corner(self, x, y):
        """Określa najbliższy róg do zmiany rozmiaru"""
        corners = [
            (self.selected_box.x1, self.selected_box.y1),
            (self.selected_box.x2, self.selected_box.y1),
            (self.selected_box.x1, self.selected_box.y2),
            (self.selected_box.x2, self.selected_box.y2)
        ]
        distances = [((cx - x) ** 2 + (cy - y) ** 2) for cx, cy in corners]
        return corners[distances.index(min(distances))]

    def _get_resized_coords(self, x, y):
        """Oblicza nowe współrzędne podczas zmiany rozmiaru"""
        if not hasattr(self, 'drag_corner'):
            return self.selected_box.get_coordinates()

        if self.drag_corner == (self.selected_box.x1, self.selected_box.y1):
            return x, y, self.selected_box.x2, self.selected_box.y2
        elif self.drag_corner == (self.selected_box.x2, self.selected_box.y1):
            return self.selected_box.x1, y, x, self.selected_box.y2
        elif self.drag_corner == (self.selected_box.x1, self.selected_box.y2):
            return x, self.selected_box.y1, self.selected_box.x2, y
        else:
            return self.selected_box.x1, self.selected_box.y1, x, y

    def _show_temp_image(self, image):
        """Wyświetla tymczasowy obraz z boxami"""
        box_layer = self.bbox_manager.get_box_layer()

        # Dopasowanie rozmiaru box_layer
        if box_layer.shape[:2] != image.shape[:2]:
            box_layer = cv2.resize(box_layer, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Dopasowanie liczby kanałów box_layer
        if len(box_layer.shape) == 2:  # Jeśli tylko jeden kanał (szarość), konwertuj do BGR
            box_layer = cv2.cvtColor(box_layer, cv2.COLOR_GRAY2BGR)
        elif box_layer.shape[2] == 4:  # Jeśli ma kanał alfa, konwertuj do BGR
            box_layer = cv2.cvtColor(box_layer, cv2.COLOR_BGRA2BGR)

        # Łączenie obrazów
        combined = cv2.addWeighted(image, 0.7, box_layer, 0.3, 0)

        # Zamiast wyświetlania w nowym oknie, aktualizujemy istniejące
        cv2.imshow("Otolith Annotation Tool", combined)

    def _update_display(self, image):
        """Aktualizuje wyświetlanie - deleguje do głównej klasy"""
        # Możesz pozostawić tę metodę pustą lub użyć do podglądu
        pass

    def _reset_state(self):
        """Resetuje stan interakcji"""
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        if hasattr(self, 'drag_corner'):
            del self.drag_corner

    def keyboard_callback(self, key):
        """Rozszerzona obsługa klawiatury"""
        key_to_mode = {
            ord('m'): Mode.MANUAL,
            ord('d'): Mode.DELETE,
            ord('v'): Mode.MOVE,
            ord('r'): Mode.RESIZE,
            27: 'exit'  # ESC
        }

        if key in key_to_mode:
            if key_to_mode[key] == 'exit':
                cv2.destroyAllWindows()
                return
            self.set_mode(key_to_mode[key])

        elif key == ord('c') and self.selected_box:  # Kopiuj box
            x1, y1, x2, y2 = self.selected_box.get_coordinates()
            self.bbox_manager.add_box(x1 + 10, y1 + 10, x2 + 10, y2 + 10)

        elif key == ord('s'):  # Sortuj boxy
            self.bbox_manager.boxes.sort(key=lambda b: b.x1)

