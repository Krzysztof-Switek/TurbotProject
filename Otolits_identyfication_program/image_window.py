import cv2
import sys
from input_handler import Mode
from bounding_box_manager import BoundingBoxManager

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

        # Narysuj wszystkie istniejące boxy
        for box in self.bbox_manager.boxes:
            pt1 = (int(box.x1), int(box.y1))
            pt2 = (int(box.x2), int(box.y2))
            cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 2)

        # Podgląd nowego boxa w trybie MANUAL
        if temp_box_coords and self.input_handler.mode == Mode.MANUAL:
            x1, y1, x2, y2 = map(int, temp_box_coords)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Wyświetl tylko informację o trybie (bez liczby boxów)
        mode_text = f"Tryb: {self.input_handler.mode.name}"
        cv2.putText(display_image, mode_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

        cv2.imshow("Otolith Annotation Tool", display_image)

    def show_image(self):
        """Główna pętla wyświetlania obrazu"""
        self.current_image = self.image_loader.load_image()
        if self.current_image is None:
            print("Brak zdjęć do wyświetlenia.")
            return

        print(f"\nZaładowany obraz - kształt: {self.current_image.shape}, typ: {self.current_image.dtype}")

        cv2.namedWindow("Otolith Annotation Tool", cv2.WINDOW_NORMAL)
        self.update_display()
        cv2.setMouseCallback("Otolith Annotation Tool", self._handle_mouse_event)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or cv2.getWindowProperty("Otolith Annotation Tool", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('n'):
                self._handle_next_image()
            else:
                if self.input_handler.keyboard_callback(key):
                    self.update_display()  # Aktualizuj tylko jeśli zmienił się tryb

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

