import os
import cv2

def get_screen_size():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

class ImageLoader:
    def __init__(self, image_dir):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Katalog '{image_dir}' nie istnieje. Program zostaje przerwany.")

        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.current_index = 0
        self.image = None
        self.original_image = None
        self.screen_width, self.screen_height = get_screen_size()

    def load_image(self):
        if self.current_index >= len(self.image_files):
            return None  # No more images

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        self.original_image = cv2.imread(image_path)
        self.image = self._resize_to_screen(self.original_image)
        return self.image

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return self.load_image()
        return None

    def _resize_to_screen(self, image):
        h, w = image.shape[:2]
        scale = min(self.screen_width / w, self.screen_height / h)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def get_original_image(self):
        return self.original_image
