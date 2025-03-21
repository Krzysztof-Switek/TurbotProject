from Otolits_identyfication_program.image_loader import ImageLoader
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager
from Otolits_identyfication_program.row_manager import RowManager
from Otolits_identyfication_program.input_handler import InputHandler
from Otolits_identyfication_program.model_yolo import YOLOModel
from Otolits_identyfication_program.gui import GUI


if __name__ == "__main__":
    image_loader = ImageLoader("Otolits_identyfication_program/test_images")
    bbox_manager = BoundingBoxManager()
    row_manager = RowManager()
    input_handler = InputHandler(bbox_manager, row_manager)
    yolo_model = YOLOModel("path/to/yolo_model")

    gui = GUI(image_loader, bbox_manager, row_manager, input_handler, yolo_model)
    gui.run()




# import os
# from image_loader import ImageLoader
# from image_window import ImageWindow
# from bounding_box_manager import BoundingBoxManager
# from input_handler import MouseHandler
#
# if __name__ == "__main__":
#     image_dir = "test_images"  # Folder z obrazami
#     image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]  # Pobieramy pliki
#
#     if not image_files:
#         print("Brak obrazów w katalogu test_images!")
#         exit()
#
#     # Tworzymy obiekt zarządzający boxami
#     bbox_manager = BoundingBoxManager()
#     # Tworzymy obiekt obsługujący mysz
#     mouse_handler = MouseHandler(bbox_manager)
#     # Tworzymy okno i podłączamy obsługę myszy
#     image_window = ImageWindow(mouse_handler, window_name="Otoliths search window")
#
#     for image_file in image_files:
#         image_path = os.path.join(image_dir, image_file)
#         print(f"Ładowanie: {image_path}")
#
#         image_loader = ImageLoader(image_path)
#         scaled_image = image_loader.get_image(scaled=True)  # Pobieramy przeskalowany obraz
#
#         image_window.show(scaled_image)  # Wyświetlamy obraz w oknie
#
#         input("Naciśnij Enter, aby zobaczyć kolejny obraz...")  # Przechodzimy do następnego obrazu
