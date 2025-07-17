from image_loader import ImageLoader
from bounding_box_manager import BoundingBoxManager
from row_detector import RowDetector
from image_loader import ImageLoader
from bounding_box_manager import BoundingBoxManager
from row_detector import RowDetector
from input_handler import InputHandler
from auto_detector import AutoDetector
from gui import GUI
import sys
import os
import traceback

if __name__ == "__main__":
    try:
        # Inicjalizacja komponentów
        image_loader = ImageLoader("test_images") # Domyślny folder, może być pusty
        bbox_manager = BoundingBoxManager()
        row_detector = RowDetector(bbox_manager)
        input_handler = InputHandler(bbox_manager, row_detector)
        auto_detector = AutoDetector(model_path='yolo/weights/best.pt')

        # Uruchomienie GUI
        app = GUI(image_loader, bbox_manager, row_detector, input_handler, auto_detector)
        app.run()

    except Exception as e:
        print(f"\nBŁĄD SYSTEMU: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)