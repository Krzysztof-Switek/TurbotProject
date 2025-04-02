from image_loader import ImageLoader
from bounding_box_manager import BoundingBoxManager
from row_detector import RowDetector
from image_window import ImageWindow
from input_handler import InputHandler
import sys, os, traceback
from auto_detector import AutoDetector

if __name__ == "__main__":
    try:
        # 1. Inicjalizacja ścieżki do obrazów
        image_dir = "test_images"
        print(f"\nŁadowanie obrazów z: {image_dir}")

        # 2. Walidacja katalogu
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"Ścieżka '{image_dir}' nie jest katalogiem")
        if not os.access(image_dir, os.R_OK):
            raise PermissionError(f"Brak uprawnień do odczytu katalogu '{image_dir}'")

        # 3. Wczytanie pierwszego obrazu
        image_loader = ImageLoader(image_dir)
        first_image = image_loader.load_image()

        if first_image is None:
            raise FileNotFoundError(f"Nie znaleziono obrazów w {image_dir}")
        if len(first_image.shape) not in {2, 3}:
            raise ValueError("Nieobsługiwany format obrazu")

        print("\nInformacje o obrazie:")
        print(f"Kształt: {first_image.shape}")
        print(f"Typ danych: {first_image.dtype}")

        # 4. Inicjalizacja komponentów
        bbox_manager = BoundingBoxManager()
        row_detector = RowDetector(bbox_manager)
        input_handler = InputHandler(bbox_manager, row_detector)

        # 5. Automatyczne wykrywanie otolitów
        auto_detector = AutoDetector(model_path='yolo/weights/best.pt')
        if auto_detector.model:
            print("\nWykrywanie otolitów (tryb AUTO)...")
            auto_boxes = auto_detector.detect(first_image)

            if auto_boxes:
                print(f"Znaleziono {len(auto_boxes)} otolitów")
                for (x1, y1, x2, y2) in auto_boxes:
                    bbox_manager.add_box(x1, y1, x2, y2, label="auto")
            else:
                print("Nie wykryto otolitów - przechodzę w tryb manualny")
        else:
            print("\nUwaga: Model YOLO nie został znaleziony. Działam w trybie manualnym.")

        # 6. Informacje o sterowaniu
        print("\nSterowanie ręczne:")
        print("m - tryb manualny")
        print("v - przesuwanie boxów")
        print("r - zmiana rozmiaru")
        print("d - usuwanie")
        print("n - następny obraz")
        print("q - wyjście")

        # 7. Uruchomienie głównego okna
        ImageWindow(image_loader, bbox_manager, input_handler, auto_detector).show_image()

    except Exception as e:
        print(f"\nBŁĄD SYSTEMU: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)