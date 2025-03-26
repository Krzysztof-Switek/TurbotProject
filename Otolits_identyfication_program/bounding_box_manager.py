import numpy as np
import cv2
from bounding_box import BoundingBox

class BoundingBoxManager:
    def __init__(self, image_shape):
        self.boxes = []
        self.box_layer = np.zeros((*image_shape[:2], 4), dtype=np.uint8)  # Warstwa RGBA

    def add_box(self, x1, y1, x2, y2, label=None):
        new_box = BoundingBox(x1, y1, x2, y2, label)
        self.boxes.append(new_box)
        self.update_box_layer()
        return new_box

    def remove_box(self, box):
        if box in self.boxes:
            self.boxes.remove(box)
            self.update_box_layer()

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

    def update_box_layer(self, opacity=0.5):
        self.box_layer.fill(0)
        for box in self.boxes:
            color = (0, 255, 0, int(255*opacity))
            cv2.rectangle(self.box_layer,
                         (box.x1, box.y1),
                         (box.x2, box.y2),
                         color, 2)

    def get_box_layer(self):
        return self.box_layer

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