import cv2
import os
import sys
import time
import gc
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
        self.input_handler.row_detector = row_detector

        self.current_image = None
        self.window_name = "Otolith Annotation Tool"
        self.image_cropper = ImageCropper(image_loader=image_loader)

        # Optymalizacja - cache renderowania
        self.last_rendered_image = None
        self.dirty = True  # Flaga wskazująca potrzebę ponownego renderowania

        # Inicjalizacja struktur do zarządzania zasobami
        self._cached_images = []

    def _release_resources(self):
        """Bezpieczne zwalnianie zasobów graficznych"""
        # Zwolnienie cache'owanych obrazów
        for img in self._cached_images:
            if img is not None:
                img = None
        self._cached_images.clear()

        # Zwolnienie ostatnio renderowanego obrazu
        if self.last_rendered_image is not None:
            self.last_rendered_image = None

        # Wymuszenie garbage collection
        gc.collect()

    def _prepare_display_image(self):
        """Przygotowanie obrazu do wyświetlenia z zarządzaniem pamięcią"""
        if self.current_image is None:
            return None

        # Zwolnienie starych zasobów przed tworzeniem nowych
        if len(self._cached_images) > 5:  # Utrzymujemy rozsądny limit cache
            self._release_resources()

        if len(self.current_image.shape) == 2:  # Grayscale
            converted = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:  # RGBA
            converted = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGRA2BGR)
        else:
            converted = self.current_image.copy()

        self._cached_images.append(converted)
        return converted

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

    def mark_dirty(self):
        """Oznacza, że obraz wymaga ponownego renderowania"""
        self.dirty = True

    def update_display(self):
        """Renderuje tylko gdy jest to konieczne"""
        if not self.dirty and self.last_rendered_image is not None:
            cv2.imshow(self.window_name, self.last_rendered_image)
            return

        display_image = self._prepare_display_image()
        if display_image is None:
            return

        # 1. Najpierw rysujemy tymczasowy box (jeśli istnieje)
        if hasattr(self.input_handler, 'temp_box') and self.input_handler.temp_box:
            self.input_handler.temp_box.draw(display_image, thickness=1)

        # 2. Potem rysujemy wszystkie stałe boxy
        for box in self.bbox_manager.boxes:
            box.draw(display_image)

        # 3. Rysujemy linie wierszy
        if hasattr(self.input_handler, 'row_detector'):
            self.input_handler.row_detector.draw_rows(display_image)

        # 4. Rysujemy informację o trybie
        self._draw_mode_info(display_image)

        # Cache'owanie wyrenderowanego obrazu
        if self.last_rendered_image is not None:
            self.last_rendered_image = None
        self.last_rendered_image = display_image.copy()
        cv2.imshow(self.window_name, display_image)
        self.dirty = False

    def show_image(self):
        """Główna pętla wyświetlania obrazu"""
        try:
            self.current_image = self.image_loader.load_image()
            if self.current_image is None:
                print("Brak zdjęć do wyświetlenia.")
                return

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
            self.mark_dirty()
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
                    self.mark_dirty()
                    self.update_display()
        finally:
            self._cleanup()

    def _cleanup(self):
        """Bezpieczne zwolnienie wszystkich zasobów"""
        self._release_resources()
        cv2.destroyAllWindows()
        if hasattr(self, 'input_handler') and hasattr(self.input_handler, 'row_detector'):
            self.input_handler.row_detector.clear_rows()
        self.bbox_manager.clear_all()

    def _handle_mouse_event(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy"""
        try:
            x, y = int(x), int(y)
        except (ValueError, TypeError):
            print(f"Nieprawidłowe współrzędne myszy: x={x}, y={y}")
            return

        if self.input_handler.mouse_callback(event, x, y):
            self.mark_dirty()
            self.update_display()

    def _handle_next_image(self):
        """Obsługa przejścia do następnego obrazu z czyszczeniem zasobów"""
        self._release_resources()  # Czyszczenie przed zmianą obrazu

        next_image = self.image_loader.next_image()
        if next_image is not None:
            self.current_image = next_image
            self.bbox_manager.clear_all()
            if hasattr(self.input_handler, 'row_detector'):
                self.input_handler.row_detector.clear_rows()
            self.mark_dirty()
            self.update_display()
        else:
            print("To już ostatnie zdjęcie.")

    def _handle_crop_boxes(self):
        """Przekazuje żądanie wycięcia boxów do ImageCropper"""
        self.image_cropper.process_cropping(self.bbox_manager, self.input_handler)