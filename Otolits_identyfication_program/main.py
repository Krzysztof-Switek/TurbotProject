from Otolits_identyfication_program.image_loader import ImageLoader
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager
from Otolits_identyfication_program.row_manager import RowManager
from Otolits_identyfication_program.input_handler import InputHandler
from Otolits_identyfication_program.model_yolo import YOLOModel
from Otolits_identyfication_program.gui import GUI
import cv2

class ImageWindow:
    def __init__(self, image_loader):
        self.image_loader = image_loader
        self.current_image = None

    def show_image(self):
        self.current_image = self.image_loader.load_image()

        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        while True:
            # Wyświetl obraz za pomocą OpenCV
            cv2.imshow("Image Viewer", self.current_image)

            key = cv2.waitKey(0) & 0xFF  # Oczekuj na naciśnięcie klawisza

            if key == ord('q'):
                # Naciśnij 'q' aby zamknąć
                break
            elif key == ord('n'):
                # Naciśnij 'n' aby załadować następne zdjęcie
                next_image = self.image_loader.next_image()
                if next_image is not None:
                    self.current_image = next_image
                else:
                    print("To już ostatnie zdjęcie.")
                    break

        # Zakończenie i zamknięcie okna
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_loader = ImageLoader("test_images")  # Ścieżka do katalogu ze zdjęciami
    bbox_manager = BoundingBoxManager()  # W tej wersji nie jest używany, ale dodajemy go
    row_manager = RowManager()  # W tej wersji nie jest używany, ale dodajemy go
    input_handler = InputHandler(bbox_manager, row_manager)  # W tej wersji nie jest używany, ale dodajemy go
    yolo_model = YOLOModel("path/to/yolo_model")  # Ścieżka do modelu YOLO

    # Inicjalizujemy GUI
    gui = GUI(image_loader, bbox_manager, row_manager, input_handler, yolo_model)  # W tej wersji GUI jest tylko zdefiniowane

    # Teraz uruchamiamy obraz
    image_window = ImageWindow(image_loader)
    image_window.show_image()

    # Uruchomienie GUI - w tej wersji jeszcze nie używamy pełnego GUI, ale klasa GUI jest przygotowana
    gui.run()
