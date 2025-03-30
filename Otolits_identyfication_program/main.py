from image_loader import ImageLoader
from bounding_box_manager import BoundingBoxManager
from row_detector import RowDetector
from image_window import ImageWindow
from input_handler import InputHandler
import sys, os, traceback

if __name__ == "__main__":
    try:
        image_dir = "test_images"
        print(f"\nŁadowanie obrazów z: {image_dir}")

        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"Ścieżka '{image_dir}' nie jest katalogiem")
        if not os.access(image_dir, os.R_OK):
            raise PermissionError(f"Brak uprawnień do odczytu katalogu '{image_dir}'")

        image_loader = ImageLoader(image_dir)
        first_image = image_loader.load_image()

        if first_image is None:
            raise FileNotFoundError(f"Nie znaleziono obrazów w {image_dir}")

        if len(first_image.shape) not in {2, 3}:
            raise ValueError("Nieobsługiwany format obrazu")

        print("\nInformacje o pierwszym obrazie:")
        print(f"Kształt: {first_image.shape}")
        print(f"Typ danych: {first_image.dtype}")

        bbox_manager = BoundingBoxManager()  # Tylko wysokość i szerokość
        row_detector = RowDetector(bbox_manager)
        input_handler = InputHandler(bbox_manager, row_detector)

        print("\nSterowanie:")
        print("m - tryb manualny (dodawanie boxów)")
        print("v - tryb przesuwania boxów")
        print("r - tryb zmiany rozmiaru")
        print("d - tryb usuwania boxów")
        print("n - następny obraz")
        print("PRAWY KLIK - wykryj wiersze")
        print("q - wyjście")

        ImageWindow(image_loader, bbox_manager, input_handler).show_image()


    except Exception as e:
        print(f"\nKRYTYCZNY BŁĄD: {str(e)}", file=sys.stderr)
        traceback.print_exc()  # Dodajemy ślad stosu
        sys.exit(1)

