import cv2
import os
import sys
from Otolits_identyfication_program.input_handler import WorkMode, ManualMode
from row_detector import RowDetector
from image_cropper import ImageCropper


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler

        # Inicjalizacja RowDetector i przypisanie do input_handler
        row_detector = RowDetector(bbox_manager)
        self.input_handler.row_detector = row_detector  # Teraz row_detector jest zdefiniowany

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

        mode_text = f"Tryb: {self.input_handler.work_mode.name}"
        if self.input_handler.work_mode.name == "MANUAL":
            mode_text += f" | Manual: {self.input_handler.manual_mode.name}"

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

    def update_display(self):
        display_image = self._prepare_display_image()
        if display_image is None:
            return

        # Narysuj wszystkie boxy (zielone)
        for box in self.bbox_manager.boxes:
            box.draw(display_image)

        # Narysuj tymczasowy box podczas tworzenia
        if (self.input_handler.work_mode == WorkMode.MANUAL and
                self.input_handler.manual_mode == ManualMode.BOX and
                self.input_handler.selection.is_drawing and
                self.input_handler.selection.element):
            box = self.input_handler.selection.element
            cv2.rectangle(display_image,
                          (int(box.x1), int(box.y1)),
                          (int(box.x2), int(box.y2)),
                          (0, 0, 255), 2)

        # Narysuj linie wierszy
        if hasattr(self.input_handler, 'row_detector'):
            self.input_handler.row_detector.draw_rows(display_image)

        self._draw_mode_info(display_image)
        cv2.imshow(self.window_name, display_image)

    def show_image(self):
        """Główna pętla wyświetlania obrazu"""
        self.current_image = self.image_loader.load_image()
        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if (key == ord('q') or
                    cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1):
                break

            if key == 13:  # Enter
                self._handle_crop_boxes()
                continue

            if key == ord('n'):
                self._handle_next_image()
                continue

            if self.input_handler.keyboard_callback(key):
                self.update_display()
                cv2.waitKey(1)

        cv2.destroyAllWindows()
        sys.exit()

    def _handle_mouse_event(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy"""
        try:
            x, y = int(x), int(y)
        except (ValueError, TypeError):
            print(f"Nieprawidłowe współrzędne myszy: x={x}, y={y}")
            return

        if self.input_handler.mouse_callback(event, x, y):
            self.update_display()  # Ważne - odświeżanie po każdym zdarzeniu
            cv2.waitKey(1)  # Dodatkowe odświeżenie

    def _handle_next_image(self):
        """Obsługa przejścia do następnego obrazu"""
        next_image = self.image_loader.next_image()
        if next_image is not None:
            self.current_image = next_image
            self.bbox_manager.clear_all()
            if hasattr(self.input_handler, 'row_detector'):
                self.input_handler.row_detector.clear_rows()
            self.update_display()
        else:
            print("To już ostatnie zdjęcie.")

    def _handle_crop_boxes(self):
        """Przekazuje żądanie wycięcia boxów do ImageCropper"""
        self.image_cropper.process_cropping(self.bbox_manager, self.input_handler)


