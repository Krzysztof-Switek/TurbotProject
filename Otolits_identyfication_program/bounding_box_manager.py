import numpy as np
from bounding_box import BoundingBox
from typing import List, Optional


class BoundingBoxManager:
    def __init__(self):
        self.boxes: List[BoundingBox] = []

    # Podstawowe operacje CRUD
    def add_box(self, x1: float, y1: float, x2: float, y2: float, label: Optional[str] = None) -> BoundingBox:
        """Dodaje nowy bounding box z automatyczną walidacją współrzędnych"""
        new_box = BoundingBox(x1, y1, x2, y2, label)
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
        for box in reversed(self.boxes):  # Sprawdzamy od najnowszych
            if box.contains(x, y, tolerance):
                return box
        return None

    # Operacje masowe
    def clear_all(self) -> None:
        """Usuwa wszystkie boxy"""
        self.boxes.clear()

    def get_boxes_sorted(self, by: str = 'area', reverse: bool = False) -> List[BoundingBox]:
        """Sortuje boxy według wybranej właściwości"""
        valid_attributes = {'area', 'width', 'height', 'aspect_ratio'}
        if by not in valid_attributes:
            raise ValueError(f"Nieprawidłowy atrybut sortowania. Dozwolone: {valid_attributes}")

        return sorted(self.boxes, key=lambda b: getattr(b, by)(), reverse=reverse)

    # Serializacja
    def to_list(self) -> List[dict]:
        """Eksport boxów do listy słowników"""
        return [box.to_dict() for box in self.boxes]

    def from_list(self, boxes_data: List[dict]) -> None:
        """Import boxów z listy słowników"""
        self.clear_all()
        for data in boxes_data:
            self.boxes.append(BoundingBox.from_dict(data))

