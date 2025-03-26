import numpy as np
import cv2
from bounding_box import BoundingBox

class BoundingBoxManager:
    def __init__(self, image_shape):
        self.boxes = []

    def add_box(self, x1, y1, x2, y2, label=None):
        new_box = BoundingBox(x1, y1, x2, y2, label)
        self.boxes.append(new_box)
        print(f"Dodano box: ({x1},{y1})-({x2},{y2})")
        print(f"Aktualna liczba boxów: {len(self.boxes)}")
        return new_box

    def remove_box(self, box):
        if box in self.boxes:
            self.boxes.remove(box)
            self.update_box_layer()
            print(f"Usunięto box: {box}")
        else:
            print("Błąd: Box nie istnieje")

    def update_box(self, box, x1, y1, x2, y2):
        if box not in self.boxes:
            raise ValueError("Box nie istnieje w managerze")
        box.update(x1, y1, x2, y2)
        self.update_box_layer()

    def get_boxes(self):
        return self.boxes.copy()  # Zwracamy kopię dla bezpieczeństwa

    def get_box_at(self, x, y, tolerance=5):
        for box in reversed(self.boxes):
            if box.contains(x, y, tolerance):
                return box
        return None

    def update_box_layer(self):
        # Metoda może być pusta, ponieważ boxy są rysowane bezpośrednio
        pass

    def get_box_layer(self):
        # Zwraca pustą warstwę, ponieważ nie używamy już nakładania warstw
        return np.zeros_like(self.box_layer)

    def clear_all(self):
        self.boxes = []
        self.update_box_layer()

    def get_boxes_sorted(self, by='area', reverse=False):
        return sorted(self.boxes,
                     key=lambda b: getattr(b, by)(),
                     reverse=reverse)

    def to_list(self):
        return [box.to_dict() for box in self.boxes]

    def from_list(self, boxes_data):
        self.clear_all()
        for data in boxes_data:
            self.boxes.append(BoundingBox.from_dict(data))
        self.update_box_layer()

