import numpy as np
from bounding_box import BoundingBox
from typing import List, Optional


class BoundingBoxManager:
    def __init__(self):
        self.boxes: List[BoundingBox] = []

    # Podstawowe operacje CRUD
    def add_box(self, x1_or_box, y1=None, x2=None, y2=None, label=None) -> BoundingBox:
        """Dodaje nowy bounding box - akceptuje współrzędne lub obiekt BoundingBox"""
        if isinstance(x1_or_box, BoundingBox):
            # Jeśli przekazano już obiekt BoundingBox
            new_box = x1_or_box
        else:
            # Jeśli przekazano współrzędne
            if None in (y1, x2, y2):
                raise ValueError("Należy podać wszystkie 4 współrzędne (x1, y1, x2, y2)")
            new_box = BoundingBox(x1_or_box, y1, x2, y2, label)

        self.boxes.append(new_box)
        return new_box

    def remove_box(self, box: BoundingBox) -> bool:
        """Usuwa box i zwraca status operacji"""
        try:
            self.boxes.remove(box)
            return True
        except ValueError:
            return False

    def update_box(self, box: BoundingBox, x1: float, y1: float, x2: float, y2: float) -> None:
        """Aktualizuje współrzędne istniejącego boxa"""
        if box not in self.boxes:
            raise ValueError("Box nie jest zarządzany przez ten manager")
        box.update(x1, y1, x2, y2)

    # Operacje zapytaniowe
    def get_boxes(self) -> List[BoundingBox]:
        """Zwraca kopię listy boxów (bezpieczne użycie zewnętrzne)"""
        return self.boxes.copy()

    def get_box_at(self, x: float, y: float, tolerance: float = 5.0) -> Optional[BoundingBox]:
        """Znajduje box zawierający punkt (x,y) z tolerancją"""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        for box in reversed(self.boxes):  # Sprawdzamy od najnowszych
            if box.contains(x, y, tolerance):
                return box
        return None

    # Operacje masowe
    def clear_all(self) -> None:
        """Usuwa wszystkie boxy"""
        self.boxes.clear()

    # Serializacja
    def to_list(self) -> List[dict]:
        """Eksport boxów do listy słowników"""
        return [box.to_dict() for box in self.boxes]

    def from_list(self, boxes_data: List[dict]) -> None:
        """Import boxów z listy słowników"""
        self.clear_all()
        for data in boxes_data:
            self.boxes.append(BoundingBox.from_dict(data))

