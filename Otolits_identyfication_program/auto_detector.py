import os
from ultralytics import YOLO
from typing import List, Tuple


class AutoDetector:
    def __init__(self, model_path='yolo/weights/best.pt'):
        """
        Inicjalizacja detektora YOLO
        Args:
            model_path: Ścieżka do wytrenowanego modelu .pt
        """
        self.model = YOLO(model_path) if os.path.exists(model_path) else None
        self.conf_threshold = 0.2  # Minimalna pewność detekcji
        self.iou_threshold = 0.6  # Próg nakładania się bboxów (60%)

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Główna metoda wykrywająca otolity na obrazie
        Zwraca:
            Listę bounding boxów w formacie (x1, y1, x2, y2)
        """
        if not self.model or image is None:
            return []

        # Krok 1: Wykonaj detekcję
        results = self.model(image)[0]

        # Krok 2: Filtruj wyniki
        boxes = self._filter_detections(results.boxes.data.tolist())

        # Krok 3: Usuń duplikaty
        return self._remove_overlapping_boxes(boxes)

    def _filter_detections(self, boxes_data: list) -> List[Tuple[float, float, float, float]]:
        """Filtruje detekcje według progu pewności i klasy"""
        return [
            box[:4] for box in boxes_data
            if box[4] > self.conf_threshold and box[5] == 1  # Tylko otolity (klasa 1)
        ]

    def _remove_overlapping_boxes(self, boxes: list) -> List[Tuple[int, int, int, int]]:
        """Usuwa nakładające się bboxy, zachowując większe"""
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if self._calculate_iou(boxes[i], boxes[j]) > self.iou_threshold:
                    if self._box_area(boxes[i]) > self._box_area(boxes[j]):
                        boxes.pop(j)
                    else:
                        boxes.pop(i)
                        i -= 1
                        break
                else:
                    j += 1
            i += 1

        return [tuple(map(int, box)) for box in boxes]

    @staticmethod
    def _calculate_iou(box1, box2) -> float:
        """Oblicza Intersection over Union dla dwóch bboxów"""
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / (area1 + area2 - intersection + 1e-6)

    @staticmethod
    def _box_area(box) -> float:
        """Oblicza powierzchnię boxa"""
        return (box[2] - box[0]) * (box[3] - box[1])