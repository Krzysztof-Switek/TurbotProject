from Otolits_identyfication_program.image_loader import ImageLoader
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager
from Otolits_identyfication_program.row_manager import RowManager
from Otolits_identyfication_program.input_handler import InputHandler, Mode
from Otolits_identyfication_program.model_yolo import YOLOModel
from Otolits_identyfication_program.gui import GUI
import cv2
import sys
import numpy as np


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler
        self.current_image = None

    def prepare_images(self):
        """Przygotowuje obrazy do wyświetlenia, zapewniając spójność formatów"""
        if self.current_image is None:
            return None, None

        # Konwersja do BGR jeśli potrzeba
        if len(self.current_image.shape) == 2:
            display_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:
            display_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        else:
            display_image = self.current_image.copy()

        # Pobranie warstwy boxów
        box_layer = self.bbox_manager.get_box_layer()

        # Jeśli box_layer nie ma trzech kanałów, konwertujemy go
        if box_layer.shape[:2] != display_image.shape[:2]:
            box_layer = cv2.resize(box_layer, (display_image.shape[1], display_image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        if len(box_layer.shape) == 2:
            box_layer = cv2.cvtColor(box_layer, cv2.COLOR_GRAY2BGR)
        elif box_layer.shape[2] == 4:
            box_layer = cv2.cvtColor(box_layer, cv2.COLOR_BGRA2BGR)

        return display_image, box_layer

    def update_display(self):
        """Aktualizuje wyświetlany obraz z boxami"""
        display_image, box_layer = self.prepare_images()
        if display_image is None or box_layer is None:
            return

        # Konwersja typów i nałożenie warstw
        combined = cv2.addWeighted(
            display_image.astype('float32'), 1,
            box_layer.astype('float32'), 0.7,
            0
        ).astype('uint8')

        cv2.imshow("Otolith Annotation Tool", combined)

    def show_image(self):
        self.current_image = self.image_loader.load_image()

        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        print(f"\nZaładowany obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")

        # Inicjalizacja managerów
        self.bbox_manager = BoundingBoxManager(self.current_image.shape)
        self.input_handler.bbox_manager = self.bbox_manager

        # Okno jest tworzone tylko raz
        if cv2.getWindowProperty("Otolith Annotation Tool", cv2.WND_PROP_VISIBLE) == -1:
            cv2.namedWindow("Otolith Annotation Tool", cv2.WINDOW_NORMAL)

        # Ustawienie callbacka tylko raz
        cv2.setMouseCallback("Otolith Annotation Tool", self.input_handler.mouse_callback, self.current_image)

        while True:
            self.update_display()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or cv2.getWindowProperty("Otolith Annotation Tool", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('n'):
                next_image = self.image_loader.next_image()
                if next_image is not None:
                    self.current_image = next_image
                    self.bbox_manager = BoundingBoxManager(self.current_image.shape)
                    self.input_handler.bbox_manager = self.bbox_manager
                    print(f"Nowy obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")
                else:
                    print("To już ostatnie zdjęcie.")
            elif key != 255:
                self.input_handler.keyboard_callback(key)

        cv2.destroyAllWindows()
        sys.exit()


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
        print(f"Liczba kanałów: {first_image.shape[2] if len(first_image.shape) > 2 else 1}")

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

        # Tworzymy okno tylko raz
        cv2.namedWindow("Otolith Annotation Tool", cv2.WINDOW_NORMAL)

        ImageWindow(image_loader, bbox_manager, input_handler).show_image()

    except Exception as e:
        print(f"\nKrytyczny błąd: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

