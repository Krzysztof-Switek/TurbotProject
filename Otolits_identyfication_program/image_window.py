import cv2
import os
import sys
from input_handler import Mode
from bounding_box_manager import BoundingBoxManager
from row_detector import RowDetector, RowEditMode
from image_cropper import ImageCropper


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler
        row_detector = RowDetector(bbox_manager)
        self.input_handler.row_detector = row_detector
        self.current_image = None
        self.temp_image = None
        self.window_name = "Otolith Annotation Tool"
        self.image_cropper = ImageCropper(image_loader=image_loader)


    def _prepare_display_image(self):
        """Przygotowanie obrazu do wyświetlenia"""
        if self.current_image is None:
            return None

        if len(self.current_image.shape) == 2:  # Grayscale
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGRA2BGR)
        return self.current_image.copy()

    def _draw_mode_info(self, image):
        """Rysuje informację o trybie na obrazie"""
        if image is None:
            return

        mode_text = f"Tryb: {self.input_handler.mode.name}"
        if hasattr(self.input_handler, 'row_detector'):
            mode_text += f" | Linie: {self.input_handler.row_detector.edit_mode.name}"

        (text_width, text_height), _ = cv2.getTextSize(
            mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(image,
                      (10, 10),
                      (20 + text_width, 20 + text_height),
                      (0, 0, 0), -1)

        cv2.putText(image, mode_text,
                    (20, 20 + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    def update_display(self, temp_box_coords=None):
        """Aktualizacja wyświetlanego obrazu"""
        display_image = self._prepare_display_image()
        if display_image is None:
            return

        # Narysuj wszystkie boxy (zielone)
        for box in self.bbox_manager.boxes:
            pt1 = (int(box.x1), int(box.y1))
            pt2 = (int(box.x2), int(box.y2))
            cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 1)

        # Podświetl boxy przypisane do wierszy (niebieskie)
        if hasattr(self.input_handler, 'row_detector'):
            for row in self.input_handler.row_detector.rows:
                for box in row.boxes:
                    pt1 = (int(box.x1), int(box.y1))
                    pt2 = (int(box.x2), int(box.y2))
                    cv2.rectangle(display_image, pt1, pt2, (255, 0, 0), 2)

            # Narysuj linie wierszy
            self.input_handler.row_detector.draw_rows(display_image)

        # Podgląd nowego boxa w trybie MANUAL
        if temp_box_coords and self.input_handler.mode == Mode.MANUAL:
            x1, y1, x2, y2 = map(int, temp_box_coords)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        self._draw_mode_info(display_image)
        cv2.imshow(self.window_name, display_image)

    def show_image(self):
        """Główna pętla wyświetlania obrazu"""
        self.current_image = self.image_loader.load_image()
        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        print(f"Zdjęcie: {self.current_image.shape}, {self.current_image.dtype}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Warunki wyjścia
            if (key == ord('q') or
                    cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1):
                break

            # Obsługa klawisza Enter - wycinanie boxów
            if key == 13:  # 13 to kod klawisza Enter
                self._handle_crop_boxes()
                continue

            # Obsługa klawiszy
            if key == ord('n'):
                self._handle_next_image()
                continue

            # Obsługa pozostałych klawiszy
            if self.input_handler.keyboard_callback(key):
                self.update_display()
                cv2.waitKey(1)  # Dodatkowe odświeżenie

        cv2.destroyAllWindows()
        sys.exit()

    def _handle_mouse_event(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy"""
        try:
            x, y = int(x), int(y)
        except (ValueError, TypeError):
            print(f"Nieprawidłowe współrzędne myszy: x={x}, y={y}")
            return

        # Przekaż zdarzenie do input_handler
        if self.input_handler.mouse_callback(event, x, y):
            self.update_display()


    def _handle_next_image(self):
        """Obsługa przejścia do następnego obrazu z resetem do trybu AUTO"""
        next_image = self.image_loader.next_image()
        if next_image is not None:
            self.current_image = next_image
            self.bbox_manager = BoundingBoxManager(self.current_image.shape)
            self.input_handler.bbox_manager = self.bbox_manager
            self.input_handler.set_mode(Mode.AUTO)  # Reset do trybu AUTO

            print(f"Nowy obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")

            # Tutaj dodaj automatyczne wykrywanie obiektów w trybie AUTO
            self._auto_detect_objects()

            self.update_display()
        else:
            print("To już ostatnie zdjęcie.")

    def _auto_detect_objects(self):
        """Automatyczne wykrywanie obiektów w trybie AUTO"""
        if self.input_handler.mode == Mode.AUTO:
            # Tutaj implementacja automatycznego wykrywania obiektów
            # Przykładowe wykrywanie - w rzeczywistości użyj swojego algorytmu
            height, width = self.current_image.shape[:2]

            # Przykładowe wykrycie 2 obiektów (do zastąpienia rzeczywistym algorytmem)
            sample_boxes = [
                (width // 4, height // 4, width // 2, height // 2),
                (3 * width // 5, height // 3, 4 * width // 5, 2 * height // 3)
            ]

            for x1, y1, x2, y2 in sample_boxes:
                self.bbox_manager.add_box(x1, y1, x2, y2)

            print(f"Automatycznie wykryto {len(sample_boxes)} obiektów")

    def _handle_crop_boxes(self):
        """Obsługa wycinania boxów po naciśnięciu Enter"""
        try:
            if not hasattr(self, 'image_cropper') or not self.image_cropper:
                print("ImageCropper nie został poprawnie zainicjalizowany")
                return


            original_image = self.image_loader.get_current_original_image()
            if original_image is None:
                print("Nie można załadować oryginalnego obrazu")
                return

            print("Rozpoczynanie procesu wycinania boxów...")
            results = self.image_cropper.crop_and_save(
                original_image,
                self.input_handler.row_detector.rows,
                self.bbox_manager.boxes
            )

            if results:
                print(f"\nPomyślnie wycięto i zapisano {len(results)} boxów:")
                for result in results:
                    print(f"- {result.filename} (wiersz {result.row_index}, box {result.box_index})")
                print(f"Pliki zapisano w: {os.path.abspath(self.image_cropper.output_dir)}\n")
            else:
                print("Nie udało się wyciąć żadnych boxów")

        except Exception as e:
            print(f"Błąd podczas wycinania boxów: {str(e)}")

