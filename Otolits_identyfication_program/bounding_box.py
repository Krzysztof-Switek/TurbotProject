import uuid
import math
import cv2
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class BoundingBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, label: Optional[str] = None, is_temp: bool = False):
        """Inicjalizacja boxa
        Args:
            is_temp: czy box jest tymczasowy podczas tworzenia
        """
        self.x1, self.x2 = float(x1), float(x2)
        self.y1, self.y2 = float(y1), float(y2)
        self.is_temp = is_temp
        if not is_temp:  # Walidacja tylko dla gotowych boxów
            self._validate_coordinates()

        self.label = label
        self.id = str(uuid.uuid4())
        self.color = (0, 255, 0)
        self.selected = False

    def _validate_coordinates(self) -> None:
        """Walidacja współrzędnych boxa"""
        if self.x1 == self.x2 or self.y1 == self.y2:
            raise ValueError("Bounding box nie może mieć zerowej szerokości/wysokości")
        if any(coord < 0 for coord in [self.x1, self.y1, self.x2, self.y2]):
            raise ValueError("Współrzędne nie mogą być ujemne")

    # ----- Podstawowe operacje -----
    def move(self, dx: float, dy: float) -> None:
        """Przesuwa box o wektor (dx, dy)"""
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy

    def resize_corner(self, corner_idx: int, x: float, y: float) -> None:
        """
        Zmienia rozmiar poprzez przeciąganie wybranego narożnika
        Args:
            corner_idx: 0-lewy górny, 1-prawy górny, 2-lewy dolny, 3-prawy dolny
        """
        corners = [
            (0, lambda x, y: setattr(self, 'x1', x) or setattr(self, 'y1', y)),
            (1, lambda x, y: setattr(self, 'x2', x) or setattr(self, 'y1', y)),
            (2, lambda x, y: setattr(self, 'x1', x) or setattr(self, 'y2', y)),
            (3, lambda x, y: setattr(self, 'x2', x) or setattr(self, 'y2', y))
        ]

        if 0 <= corner_idx < 4:
            corners[corner_idx][1](x, y)
            self._normalize_coords()

    def _normalize_coords(self) -> None:
        """Upewnia się że x1 < x2 i y1 < y2"""
        self.x1, self.x2 = sorted([self.x1, self.x2])
        self.y1, self.y2 = sorted([self.y1, self.y2])
        self._validate_coordinates()

    # ----- Operacje geometrzyczne -----
    def contains(self, x: float, y: float, tolerance: float = 0) -> bool:
        """Sprawdza czy punkt (x,y) jest w boxie z tolerancją"""
        return (self.x1 - tolerance <= x <= self.x2 + tolerance and
                self.y1 - tolerance <= y <= self.y2 + tolerance)

    def intersects(self, other: 'BoundingBox') -> bool:
        """Sprawdza czy boxy się nakładają"""
        return not (self.x2 < other.x1 or self.x1 > other.x2 or
                    self.y2 < other.y1 or self.y1 > other.y2)

    def distance_to(self, x: float, y: float) -> float:
        """Oblicza minimalną odległość punktu od boxa"""
        cx, cy = self.get_center()
        dx = max(self.x1 - x, 0, x - self.x2)
        dy = max(self.y1 - y, 0, y - self.y2)
        return math.sqrt(dx * dx + dy * dy)

    # ----- Właściwości geometryczne -----
    def get_center(self) -> Tuple[float, float]:
        """Zwraca środek boxa (x, y)"""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def get_corners(self) -> List[Tuple[float, float]]:
        """Zwraca współrzędne wszystkich rogów w kolejności:
        0-lewy górny, 1-prawy górny, 2-lewy dolny, 3-prawy dolny"""
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x1, self.y2),
            (self.x2, self.y2)
        ]

    def get_nearest_corner(self, x: float, y: float) -> int:
        """Zwraca indeks najbliższego narożnika (0-3)"""
        corners = self.get_corners()
        distances = [math.hypot(cx - x, cy - y) for cx, cy in corners]
        return distances.index(min(distances))

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self) -> float:
        return self.x2 - self.x1

    def height(self) -> float:
        return self.y2 - self.y1

    def aspect_ratio(self) -> float:
        return self.width() / self.height()

    # ----- Wizualizacja i serializacja -----
    def draw(self, image: np.ndarray, color: Optional[Tuple[int, int, int]] = None,
             thickness: int = 2) -> None:
        """Rysuje box na obrazie"""
        color = color if color is not None else self.color
        if self.selected:
            color = (0, 0, 255)  # Czerwony dla zaznaczonego
            thickness = max(3, thickness)

        cv2.rectangle(image,
                      (int(self.x1), int(self.y1)),
                      (int(self.x2), int(self.y2)),
                      color, thickness)

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika (do zapisu)"""
        return {
            'id': self.id,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'label': self.label,
            'color': self.color
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        """Tworzy box ze słownika"""
        box = cls(data['x1'], data['y1'], data['x2'], data['y2'], data.get('label'))
        if 'color' in data:
            box.color = tuple(data['color'])
        return box

    # ----- Metody specjalne -----
    def __str__(self) -> str:
        return f"BBox({self.x1:.1f},{self.y1:.1f})-({self.x2:.1f},{self.y2:.1f})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> 'BoundingBox':
        """Tworzy dokładną kopię boxa"""
        return BoundingBox(self.x1, self.y1, self.x2, self.y2, self.label)