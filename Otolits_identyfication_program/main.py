from Otolits_identyfication_program.image_loader import ImageLoader
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager
from Otolits_identyfication_program.row_manager import RowManager
from enum import Enum, auto
import cv2
import sys
import numpy as np


class Mode(Enum):
    AUTO = auto()  # Tryb automatyczny (domyślny)
    MANUAL = auto()  # Ręczne dodawanie boxów
    MOVE = auto()  # Przesuwanie boxów
    RESIZE = auto()  # Zmiana rozmiaru boxów
    DELETE = auto()  # Usuwanie boxów


class InputHandler:
    def __init__(self, bounding_box_manager, row_manager):
        self.bbox_manager = bounding_box_manager
        self.row_manager = row_manager
        self.mode = Mode.AUTO  # Domyślny tryb
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
        self.selected_box = None
        self.drag_offset = None
        print("Aktywny tryb: AUTO")  # Domyślny tryb przy inicjalizacji

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


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler
        self.current_image = None
        self.temp_image = None

    def _prepare_display_image(self):
        """Przygotowanie obrazu do wyświetlenia"""
        if self.current_image is None:
            return None

        if len(self.current_image.shape) == 2:  # Grayscale
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGRA2BGR)
        return self.current_image.copy()

    def update_display(self, temp_box_coords=None):
        """Aktualizacja wyświetlanego obrazu z boxami"""
        display_image = self._prepare_display_image()
        if display_image is None:
            return

        # Narysuj wszystkie istniejące boxy (zielone)
        for box in self.bbox_manager.boxes:
            try:
                pt1 = (int(box.x1), int(box.y1))
                pt2 = (int(box.x2), int(box.y2))
                cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 2)
            except Exception as e:
                print(f"Błąd rysowania boxa {box}: {e}")
                continue

        # Podgląd nowego boxa (czerwony) tylko w trybie MANUAL
        if temp_box_coords and self.input_handler.mode == Mode.MANUAL:
            try:
                x1, y1, x2, y2 = map(int, temp_box_coords)
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            except Exception as e:
                print(f"Błąd rysowania tymczasowego boxa: {e}")

        cv2.imshow("Otolith Annotation Tool", display_image)

    def show_image(self):
        """Główna pętla wyświetlania obrazu"""
        self.current_image = self.image_loader.load_image()
        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        print(f"\nZaładowany obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")

        # Inicjalizacja okna
        cv2.namedWindow("Otolith Annotation Tool", cv2.WINDOW_NORMAL)
        self.update_display()

        # Ustawienie callbacka myszy
        cv2.setMouseCallback("Otolith Annotation Tool",
                             self._handle_mouse_event,
                             {'image': self.current_image})

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or cv2.getWindowProperty("Otolith Annotation Tool", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('n'):
                self._handle_next_image()
            else:
                self.input_handler.keyboard_callback(key)

        cv2.destroyAllWindows()
        sys.exit()

    def _handle_mouse_event(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy"""
        try:
            x, y = int(x), int(y)  # Upewnij się, że współrzędne są integerami
        except (ValueError, TypeError):
            print(f"Nieprawidłowe współrzędne myszy: x={x}, y={y}")
            return

        # Tryb DELETE
        if self.input_handler.mode == Mode.DELETE and event == cv2.EVENT_LBUTTONDOWN:
            box = self.bbox_manager.get_box_at(x, y, tolerance=5)
            if box:
                self.bbox_manager.remove_box(box)
                self.update_display()
            return

        # Tryb MOVE
        if self.input_handler.mode == Mode.MOVE:
            if event == cv2.EVENT_LBUTTONDOWN:
                box = self.bbox_manager.get_box_at(x, y, tolerance=5)
                if box:
                    self.input_handler.selected_box = box
                    self.input_handler.drag_offset = (x - box.x1, y - box.y1)
                    self.input_handler.start_pos = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE and self.input_handler.selected_box:
                dx = x - self.input_handler.start_pos[0]
                dy = y - self.input_handler.start_pos[1]
                temp_box = self.input_handler.selected_box

                # Utwórz kopię obrazu do podglądu
                display_image = self._prepare_display_image()
                if display_image is None:
                    return

                # Narysuj wszystkie boxy
                for box in self.bbox_manager.boxes:
                    if box != temp_box:
                        pt1 = (int(box.x1), int(box.y1))
                        pt2 = (int(box.x2), int(box.y2))
                        cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 2)

                # Narysuj podgląd przesuwanego boxa (niebieski)
                pt1 = (int(temp_box.x1 + dx), int(temp_box.y1 + dy))
                pt2 = (int(temp_box.x2 + dx), int(temp_box.y2 + dy))
                cv2.rectangle(display_image, pt1, pt2, (255, 0, 0), 2)
                cv2.imshow("Otolith Annotation Tool", display_image)

            elif event == cv2.EVENT_LBUTTONUP and self.input_handler.selected_box:
                dx = x - self.input_handler.start_pos[0]
                dy = y - self.input_handler.start_pos[1]
                box = self.input_handler.selected_box
                box.move(dx, dy)
                self.input_handler.selected_box = None
                self.update_display()
            return

        # Tryb RESIZE
        if self.input_handler.mode == Mode.RESIZE:
            if event == cv2.EVENT_LBUTTONDOWN:
                box = self.bbox_manager.get_box_at(x, y, tolerance=10)
                if box:
                    self.input_handler.selected_box = box
                    self.input_handler.start_pos = (x, y)
                    self.input_handler.drag_corner = box.get_nearest_corner(x, y)

            elif event == cv2.EVENT_MOUSEMOVE and self.input_handler.selected_box:
                box = self.input_handler.selected_box
                corner = self.input_handler.drag_corner

                # Utwórz kopię obrazu do podglądu
                display_image = self._prepare_display_image()
                if display_image is None:
                    return

                # Narysuj wszystkie boxy
                for b in self.bbox_manager.boxes:
                    if b != box:
                        pt1 = (int(b.x1), int(b.y1))
                        pt2 = (int(b.x2), int(b.y2))
                        cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 2)

                # Narysuj podgląd zmienianego boxa (żółty)
                temp_coords = list(box.get_coordinates())
                if corner == 1:  # lewy górny
                    temp_coords[0], temp_coords[1] = x, y
                elif corner == 2:  # prawy górny
                    temp_coords[2], temp_coords[1] = x, y
                elif corner == 3:  # lewy dolny
                    temp_coords[0], temp_coords[3] = x, y
                elif corner == 4:  # prawy dolny
                    temp_coords[2], temp_coords[3] = x, y

                pt1 = (int(temp_coords[0]), int(temp_coords[1]))
                pt2 = (int(temp_coords[2]), int(temp_coords[3]))
                cv2.rectangle(display_image, pt1, pt2, (0, 255, 255), 2)
                cv2.imshow("Otolith Annotation Tool", display_image)

            elif event == cv2.EVENT_LBUTTONUP and self.input_handler.selected_box:
                box = self.input_handler.selected_box
                corner = self.input_handler.drag_corner
                box.resize(corner, x, y)
                self.input_handler.selected_box = None
                self.update_display()
            return

        # Tryb MANUAL
        if self.input_handler.mode == Mode.MANUAL:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.input_handler.start_pos = (x, y)
                self.input_handler.drawing = True
                self.temp_image = self._prepare_display_image()

            elif event == cv2.EVENT_MOUSEMOVE and self.input_handler.drawing:
                temp_box = (self.input_handler.start_pos[0],
                            self.input_handler.start_pos[1], x, y)
                self.update_display(temp_box_coords=temp_box)

            elif event == cv2.EVENT_LBUTTONUP and self.input_handler.drawing:
                x1, x2 = sorted([self.input_handler.start_pos[0], x])
                y1, y2 = sorted([self.input_handler.start_pos[1], y])
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Minimalny rozmiar
                    self.bbox_manager.add_box(x1, y1, x2, y2)
                self.input_handler.drawing = False
                self.update_display()

    def _handle_next_image(self):
        """Obsługa przejścia do następnego obrazu"""
        next_image = self.image_loader.next_image()
        if next_image is not None:
            self.current_image = next_image
            self.bbox_manager = BoundingBoxManager(self.current_image.shape)
            self.input_handler.bbox_manager = self.bbox_manager
            print(f"Nowy obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")
            self.update_display()
        else:
            print("To już ostatnie zdjęcie.")


if __name__ == "__main__":
    try:
        image_dir = "test_images"
        print(f"\nŁadowanie obrazów z: {image_dir}")

        image_loader = ImageLoader(image_dir)
        first_image = image_loader.load_image()

        if first_image is None:
            raise FileNotFoundError(f"Nie znaleziono obrazów w {image_dir}")

        print("\nInformacje o pierwszym obrazie:")
        print(f"Kształt: {first_image.shape}")
        print(f"Typ danych: {first_image.dtype}")

        bbox_manager = BoundingBoxManager(first_image.shape)
        row_manager = RowManager()
        input_handler = InputHandler(bbox_manager, row_manager)

        print("\nSterowanie:")
        print("m - tryb manualny (dodawanie boxów)")
        print("v - tryb przesuwania boxów")
        print("r - tryb zmiany rozmiaru")
        print("d - tryb usuwania boxów")
        print("n - następny obraz")
        print("q - wyjście")

        ImageWindow(image_loader, bbox_manager, input_handler).show_image()

    except Exception as e:
        print(f"\nBłąd: {str(e)}")
        sys.exit(1)

