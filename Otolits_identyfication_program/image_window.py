import cv2
import sys
from input_handler import Mode
from bounding_box_manager import BoundingBoxManager
from row_detector import RowDetector, RowEditMode


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler
        self.current_image = None
        self.temp_image = None
        self.row_detector = RowDetector(bbox_manager)
        self.input_handler.row_detector = self.row_detector
        self.window_name = "Otolith Annotation Tool"

    def _prepare_display_image(self):
        """Przygotowanie obrazu do wyświetlenia z konwersją kolorów"""
        if self.current_image is None:
            return None

        # Konwersja do BGR jeśli potrzebne
        if len(self.current_image.shape) == 2:  # Grayscale
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGRA2BGR)
        return self.current_image.copy()

    def _draw_mode_info(self, image):
        """Rysuje informację o trybie na obrazie"""
        if image is None:
            return

        # Tekst głównego trybu
        mode_text = f"Tryb: {self.input_handler.mode.name}"

        # Dodaj informację o trybie edycji linii jeśli aktywny
        if self.row_detector.edit_mode != RowEditMode.NONE:
            mode_text += f" | Edycja linii: {self.row_detector.edit_mode.name}"

        # Tło dla tekstu dla lepszej czytelności
        (text_width, text_height), _ = cv2.getTextSize(
            mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(image,
                      (10, 10),
                      (20 + text_width, 20 + text_height),
                      (0, 0, 0), -1)

        # Tekst trybu
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

        # Narysuj wszystkie istniejące boxy (zielone)
        for box in self.bbox_manager.boxes:
            pt1 = (int(box.x1), int(box.y1))
            pt2 = (int(box.x2), int(box.y2))
            cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 2)

        # Rysuj linie wierszy (czerwone)
        self.row_detector.draw_rows(display_image)

        # Podgląd nowego boxa w trybie MANUAL
        if temp_box_coords and self.input_handler.mode == Mode.MANUAL:
            x1, y1, x2, y2 = map(int, temp_box_coords)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Wyświetl informację o trybie
        self._draw_mode_info(display_image)

        cv2.imshow(self.window_name, display_image)

    def show_image(self):
        """Główna pętla wyświetlania obrazu"""
        self.current_image = self.image_loader.load_image()
        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        print(f"\nZaładowany obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Obsługa klawiszy
            if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('n'):
                self._handle_next_image()
            else:
                if self.input_handler.keyboard_callback(key):
                    self.update_display()

        cv2.destroyAllWindows()
        sys.exit()

    def _handle_mouse_event(self, event, x, y, flags, param):
        """Obsługa zdarzeń myszy"""
        try:
            x, y = int(x), int(y)  # Upewnij się, że współrzędne są integerami
        except (ValueError, TypeError):
            print(f"Nieprawidłowe współrzędne myszy: x={x}, y={y}")
            return

        # Najpierw sprawdź tryb edycji wierszy
        if self.row_detector.edit_mode != RowEditMode.NONE:
            if self.row_detector.handle_mouse_event(event, x, y):
                self.update_display()
            return

        # Następnie sprawdź tryby związane z bounding boxami
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

        # Obsługa prawego kliknięcia myszki
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.input_handler.mode == Mode.MANUAL:
                try:
                    self.row_detector.detect_rows()
                    self.update_display()
                except Exception as e:
                    print(f"Błąd wykrywania wierszy: {e}")

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

