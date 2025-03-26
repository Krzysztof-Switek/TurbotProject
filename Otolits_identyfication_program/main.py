from image_loader import ImageLoader
from bounding_box_manager import BoundingBoxManager
from row_manager import RowManager
from image_window import ImageWindow
from input_handler import InputHandler, Mode
import cv2
import sys

if __name__ == "__main__":
    try:
        image_dir = "test_images"
        print(f"\nŁadowanie obrazów z: {image_dir}")

        image_loader = ImageLoader(image_dir)
        first_image = image_loader.load_image()

        if first_image is None:
            raise FileNotFoundError(f"Nie znaleziono obrazów w {image_dir}")

        print("\nInformacje o pierwszym obrazie:")
        print(f"Kształt: {first_image.shape}")
        print(f"Typ danych: {first_image.dtype}")

        bbox_manager = BoundingBoxManager(first_image.shape)
        row_manager = RowManager()
        input_handler = InputHandler(bbox_manager, row_manager)

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
        print(f"\nBłąd: {str(e)}")
        sys.exit(1)