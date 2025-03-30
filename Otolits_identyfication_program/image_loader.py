import os
import cv2
import numpy as np
from typing import Optional, Tuple  # Dodaj Tuple do importów

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
        self.scale = 1.0  # Domyślna wartość
        self.original_size = (0, 0)  # Inicjalizacja

    def load_image(self):
        if self.current_index >= len(self.image_files):
            return None

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Nie udało się załadować obrazu: {image_path}")
        self.image = self._resize_to_screen(self.original_image)
        return self.image

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return self.load_image()
        return None

    def _resize_to_screen(self, image):
        h, w = image.shape[:2]
        self.scale = min(self.screen_width / w, self.screen_height / h) * 0.9
        self.original_size = (w, h)
        new_size = (int(w * self.scale), int(h * self.scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def get_original_image(self) -> Optional[np.ndarray]:
        """Zwraca oryginalny, nieprzeskalowany obraz"""
        return self.original_image.copy() if self.original_image is not None else None

    def get_current_original_image(self) -> Optional[np.ndarray]:
        """Alias dla get_original_image() dla spójności interfejsu"""
        return self.get_original_image()

    def scale_coords_to_original(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """Przelicza współrzędne z przeskalowanego podglądu na oryginalny obraz"""
        scale_factor = 1 / self.scale
        return (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor),
            int(y2 * scale_factor)
        )

    @property
    def current_image_path(self) -> Optional[str]:
        """Zwraca ścieżkę do aktualnie załadowanego obrazu"""
        if 0 <= self.current_index < len(self.image_files):
            return os.path.join(self.image_dir, self.image_files[self.current_index])
        return None

    def scale_coords_to_original(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        """Przelicza współrzędne z przeskalowanego podglądu na oryginalny obraz"""
        if not hasattr(self, 'scale') or self.scale == 0:
            return x1, y1, x2, y2

        scale_factor = 1 / self.scale
        return (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor),
            int(y2 * scale_factor)
        )

