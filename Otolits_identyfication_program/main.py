from Otolits_identyfication_program.image_loader import ImageLoader
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager
from Otolits_identyfication_program.row_manager import RowManager
from Otolits_identyfication_program.input_handler import InputHandler, Mode
import cv2
import sys
import numpy as np


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler
        self.current_image = None
        self.temp_image = None
        self.window_initialized = False

    def _prepare_display_image(self):
        """Przygotowanie obrazu do wyświetlenia z konwersją kolorów"""
        if self.current_image is None:
            return None

        if len(self.current_image.shape) == 2:  # Grayscale
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGRA2BGR)
        return self.current_image.copy()

    def update_display(self, temp_box_coords=None):
        """
        Aktualizuje wyświetlany obraz
        temp_box_coords: (x1,y1,x2,y2) dla podglądu boxa podczas rysowania
        """
        display_image = self._prepare_display_image()
        if display_image is None:
            return

        # 1. Narysuj wszystkie istniejące boxy (zielone)
        for box in self.bbox_manager.boxes:
            cv2.rectangle(display_image,
                          (box.x1, box.y1),
                          (box.x2, box.y2),
                          (0, 255, 0), 2)

        # 2. Jeśli jest aktywny podgląd nowego boxa (czerwony)
        if temp_box_coords and self.input_handler.mode == Mode.MANUAL:
            x1, y1, x2, y2 = temp_box_coords
            cv2.rectangle(display_image,
                          (x1, y1),
                          (x2, y2),
                          (0, 0, 255), 2)

        cv2.imshow("Otolith Annotation Tool", display_image)

    def show_image(self):
        self.current_image = self.image_loader.load_image()
        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        print(f"\nZaładowany obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")

        # Inicjalizacja managerów
        self.bbox_manager = BoundingBoxManager(self.current_image.shape)
        self.input_handler.bbox_manager = self.bbox_manager
        self.input_handler.set_mode(Mode.MOVE)  # Domyślny tryb - tylko przeglądanie

        # Inicjalizacja okna i wyświetlenie obrazu
        cv2.namedWindow("Otolith Annotation Tool", cv2.WINDOW_NORMAL)
        self.update_display()  # Pierwsze wyświetlenie obrazu
        self.window_initialized = True

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
            elif key in [ord('m'), ord('v'), ord('r'), ord('d')]:
                self.input_handler.keyboard_callback(key)
                print(f"Aktywny tryb: {self.input_handler.mode.name}")

        cv2.destroyAllWindows()
        sys.exit()

    def _handle_mouse_event(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy z kontrolą trybu"""
        if not self.window_initialized:
            return

        # Obsługa tylko w trybie manualnym
        if self.input_handler.mode == Mode.MANUAL:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.input_handler._handle_left_click(x, y)
                self.temp_image = self._prepare_display_image()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.input_handler.drawing:
                    temp_box = (self.input_handler.start_pos[0],
                                self.input_handler.start_pos[1], x, y)
                    self.update_display(temp_box_coords=temp_box)

            elif event == cv2.EVENT_LBUTTONUP:
                if self.input_handler.drawing:
                    self.input_handler._handle_left_release(x, y, self.current_image)
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
        print(f"\nKrytyczny błąd: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)