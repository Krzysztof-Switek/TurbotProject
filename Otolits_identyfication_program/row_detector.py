import numpy as np
import cv2
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class RowLine:
    slope: float
    intercept: float
    boxes: List['BoundingBox']  # Lista bboxów przypisanych do tej linii


class RowDetector:
    def __init__(self, bbox_manager):
        self.bbox_manager = bbox_manager
        self.rows: List[RowLine] = []

    def detect_rows(self) -> List[RowLine]:
        """Główna metoda wykrywająca wiersze"""
        if not self.bbox_manager.boxes:
            return []

        # 1. Grupowanie bboxów w potencjalne wiersze
        boxes_sorted = sorted(self.bbox_manager.boxes, key=lambda b: b.y1)
        clusters = self._cluster_boxes(boxes_sorted)

        # 2. Dopasowanie linii do każdej grupy
        self.rows = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue  # Pomijamy pojedyncze bboxy

            x_centers = [(b.x1 + b.x2) / 2 for b in cluster]
            y_centers = [(b.y1 + b.y2) / 2 for b in cluster]

            # Dopasowanie linii metodą najmniejszych kwadratów
            A = np.vstack([x_centers, np.ones(len(x_centers))]).T
            slope, intercept = np.linalg.lstsq(A, y_centers, rcond=None)[0]

            self.rows.append(RowLine(slope, intercept, cluster))

        # 3. Upewniamy się, że linie nie przecinają się
        self._ensure_non_intersecting()

        return self.rows

    def _cluster_boxes(self, boxes: List['BoundingBox'], threshold: float = 30.0) -> List[List['BoundingBox']]:
        """Grupuje bboxy w wiersze na podstawie pozycji pionowej"""
        clusters = []
        current_cluster = [boxes[0]]

        for box in boxes[1:]:
            # Sprawdzamy czy bbox pasuje do obecnego klastra (na podstawie środka)
            last_box = current_cluster[-1]
            last_center = (last_box.y1 + last_box.y2) / 2
            current_center = (box.y1 + box.y2) / 2

            if abs(current_center - last_center) < threshold:
                current_cluster.append(box)
            else:
                clusters.append(current_cluster)
                current_cluster = [box]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _ensure_non_intersecting(self):
        """Upewnia się, że linie nie przecinają się"""
        if len(self.rows) < 2:
            return

        # Sortujemy linie po intercept
        self.rows.sort(key=lambda line: line.intercept)

        # Sprawdzamy czy nachylenia są rosnące
        for i in range(1, len(self.rows)):
            if self.rows[i].slope <= self.rows[i - 1].slope:
                # Jeśli nie, uśredniamy nachylenia
                avg_slope = (self.rows[i].slope + self.rows[i - 1].slope) / 2
                self.rows[i].slope = avg_slope
                self.rows[i - 1].slope = avg_slope

    def draw_rows(self, image):
        """Draw detected row lines"""
        for line in self.rows:
            if len(line.boxes) < 2:
                continue

            x_coords = [b.x1 for b in line.boxes] + [b.x2 for b in line.boxes]
            min_x, max_x = min(x_coords), max(x_coords)

            # Konwersja współrzędnych na integer
            min_x = int(min_x)
            max_x = int(max_x)
            y1 = int(line.slope * min_x + line.intercept)
            y2 = int(line.slope * max_x + line.intercept)

            # Upewniamy się, że współrzędne są w zakresie obrazu
            height, width = image.shape[:2]
            min_x = max(0, min(min_x, width - 1))
            max_x = max(0, min(max_x, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            cv2.line(image, (min_x, y1), (max_x, y2), (0, 0, 255), 2)