import cv2
import gc
import traceback
from row_detector import RowDetector
from image_cropper import ImageCropper, compute_row_labels, compute_compartment_bboxes
from input_handler import WorkMode, ManualMode
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class ImageWindow:
    def __init__(self, image_loader, bbox_manager, input_handler, auto_detector=None):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.input_handler = input_handler
        self.auto_detector = auto_detector

        self.current_image = None
        self.window_name = "Otolith Annotation Tool"
        self.image_cropper = ImageCropper(image_loader=image_loader)
        self.input_handler.row_detector = RowDetector(bbox_manager)
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()

        self.dirty = True
        self._cached_images = []

    def _prepare_display_image(self):
        if self.current_image is None:
            return None

        if len(self._cached_images) > 5:
            self._release_resources()

        if len(self.current_image.shape) == 2:
            converted = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:
            converted = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGRA2BGR)
        else:
            converted = self.current_image.copy()

        self._cached_images.append(converted)
        return converted

    def _release_resources(self):
        self._cached_images.clear()
        gc.collect()

    def update_display(self):
        if not self.dirty:
            return

        display_image = self._prepare_display_image()
        if display_image is None:
            return

        for box in self.bbox_manager.boxes:
            in_row = False
            if hasattr(self.input_handler, 'row_detector'):
                in_row = any(box in row.boxes for row in self.input_handler.row_detector.rows)
            color = (0, 255, 0) if in_row else (0, 0, 255)
            box.draw(display_image, color=color)

        if hasattr(self.input_handler, 'temp_box') and self.input_handler.temp_box:
            self.input_handler.temp_box.draw(display_image)

        if hasattr(self.input_handler, 'row_detector'):
            self.input_handler.row_detector.draw_rows(display_image)

        # Etykiety wierszy + bbox-y wycinków — jedno wywołanie compute_row_labels.
        row_labels: dict = {}
        label_err = None
        compartment_bboxes: dict = {}
        if hasattr(self.input_handler, 'row_detector'):
            rows = self.input_handler.row_detector.rows
            row_labels, label_err = compute_row_labels(rows)
            if label_err is None:
                compartment_bboxes = compute_compartment_bboxes(rows, labels=row_labels)

        # Ramki wycinków A (niebieski) i B (fioletowy).
        compartment_colors = {'A': (255, 0, 0), 'B': (200, 0, 200)}  # BGR
        for label, (x1, y1, x2, y2) in compartment_bboxes.items():
            cv2.rectangle(
                display_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                compartment_colors[label],
                2,
            )

        pil_image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        mode_info = self.input_handler.get_mode_info()
        draw.text((10, 10), mode_info, font=self.font, fill=(255, 255, 255))

        # Statusbar: liczniki wierszy w wycinku A i B.
        counts = {'A': 0, 'B': 0}
        for label_str in row_labels.values():
            counts[label_str[0]] += 1
        status_text = f"Wycinek A: {counts['A']} wierszy | Wycinek B: {counts['B']} wierszy"
        draw.text((10, 32), status_text, font=self.font, fill=(255, 255, 255))

        # Etykieta wycinka ("A"/"B") w lewym górnym rogu jego ramki.
        compartment_text_colors = {'A': (0, 0, 255), 'B': (200, 0, 200)}
        for label, (x1, y1, _, _) in compartment_bboxes.items():
            draw.text(
                (int(x1) + 4, int(y1) + 2),
                label,
                font=self.font,
                fill=compartment_text_colors[label],
                stroke_width=2, stroke_fill=(255, 255, 255),
            )

        # Etykiety wierszy (A_1, B_3, ...) przy lewym końcu linii.
        if label_err:
            draw.text(
                (10, pil_image.height - 50),
                label_err,
                font=self.font,
                fill=(255, 80, 80),
                stroke_width=2, stroke_fill=(0, 0, 0),
            )
        elif hasattr(self.input_handler, 'row_detector'):
            for row in self.input_handler.row_detector.rows:
                label = row_labels.get(id(row))
                if not label:
                    continue
                x = int(row.line.p1[0]) + 8
                y = int(row.line.p1[1]) - 20
                draw.text(
                    (x, y),
                    label,
                    font=self.font,
                    fill=(255, 255, 255),
                    stroke_width=2, stroke_fill=(0, 0, 0),
                )

        y_pos = pil_image.height - 30
        font_small = self.font.font_variant(size=12)
        for key, desc in self.input_handler.get_key_bindings_info().items():
            text = f"{key}: {desc}"
            draw.text((10, y_pos), text, font=font_small, fill=(255, 255, 255))
            y_pos -= 20

        display_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        cv2.imshow(self.window_name, display_image)
        self.dirty = False

    def show_image(self):
        try:
            self.current_image = self.image_loader.load_image()
            if self.current_image is None:
                print("Brak obrazów do wyświetlenia")
                return

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
            self.mark_dirty()

            while True:
                # Render co iterację: prosty, niezawodny model. Aplikacja idle
                # generuje stały frame rate, ale dzięki temu żadna ścieżka stanu
                # nie potrzebuje pamiętać o wywołaniu mark_dirty.
                self.mark_dirty()
                self.update_display()

                key = cv2.waitKey(1)
                if key != -1:
                    key = key & 0xFF

                    if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    if key == 13:  # Enter
                        self._handle_crop_boxes()

                    if key == ord('n'):
                        self._handle_next_image()

                    # Konwersja na małą literę
                    key_char = chr(key).lower()
                    self.input_handler.keyboard_callback(ord(key_char))

        finally:
            self._cleanup()

    def mark_dirty(self):
        self.dirty = True

    def _handle_mouse_event(self, event, x, y, flags, param):
        try:
            if self.input_handler.mouse_callback(event, int(x), int(y)):
                self.mark_dirty()
        except (ValueError, TypeError):
            print(f"Błędne współrzędne myszy: {x}, {y}")
        except Exception as e:
            # Każdy inny błąd MUSI być widoczny w konsoli — cv2 mouse callback
            # silently swallowuje wyjątki, a my potrzebujemy diagnostyki.
            print(f"BŁĄD mouse callback: {type(e).__name__}: {e}")
            traceback.print_exc()

    def _handle_next_image(self):
        self._release_resources()

        try:
            next_image = self.image_loader.next_image()
            if next_image is None:  # Poprawione sprawdzanie
                print("To już ostatnie zdjęcie.")
                return

            self.current_image = next_image
            self.bbox_manager.clear_all()
            if hasattr(self.input_handler, 'row_detector'):
                self.input_handler.row_detector.clear_rows()

            # Automatyczne wykrywanie na nowym obrazie
            if self.auto_detector and self.auto_detector.model:
                auto_boxes = self.auto_detector.detect(next_image)
                for (x1, y1, x2, y2) in auto_boxes:
                    self.bbox_manager.add_box(x1, y1, x2, y2, label="auto")
                self.input_handler._set_work_mode(WorkMode.MANUAL)
                self.input_handler._set_manual_mode(ManualMode.ADD_LINE)

            self.input_handler.row_detector = RowDetector(self.bbox_manager)
            self.input_handler.reset_to_defaults()
            self.mark_dirty()

        except Exception as e:
            print(f"Błąd podczas ładowania obrazu: {str(e)}")
            traceback.print_exc()

    def _handle_crop_boxes(self):
        self.image_cropper.process_cropping(self.bbox_manager, self.input_handler)

    def _cleanup(self):
        self._release_resources()
        cv2.destroyAllWindows()
        if hasattr(self.input_handler, 'row_detector'):
            self.input_handler.row_detector.clear_rows()
        self.bbox_manager.clear_all()