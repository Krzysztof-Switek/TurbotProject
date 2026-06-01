import os
import cv2
import numpy as np
from typing import Optional, Tuple
import exifread


class ImageLoader:
    """Ładuje obrazy z katalogu (sekwencyjnie), skaluje do podglądu z limitem
    `max_preview_px` (szer lub wys, w pikselach).

    Używany przez:
    - desktop (`main.py`, `image_window.py`) — sekwencyjna nawigacja przez `n`.
    - web (`web/services/image_service.py`) — pojedynczy plik przez `from_path`.

    Bez zależności GUI (tkinter usunięty) — działa w headless kontenerach Docker.
    """

    def __init__(self, image_dir: str, max_preview_px: int = 1920):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Katalog '{image_dir}' nie istnieje")

        self.image_dir = image_dir
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        self.current_index = 0
        self.image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.max_preview_px: int = max_preview_px
        self.scale: float = 1.0
        self.original_size: Tuple[int, int] = (0, 0)

    @classmethod
    def from_path(cls, image_path: str, max_preview_px: int = 1920) -> "ImageLoader":
        """Tworzy ImageLoader skierowany na konkretny plik (web usecase).

        Loader będzie miał tylko ten jeden plik jako `image_files[0]` i
        `current_index=0`. Wywołaj `load_image()` żeby załadować.
        """
        directory = os.path.dirname(image_path) or "."
        filename = os.path.basename(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Plik nie istnieje: {image_path}")

        loader = cls.__new__(cls)
        loader.image_dir = directory
        loader.image_files = [filename]
        loader.current_index = 0
        loader.image = None
        loader.original_image = None
        loader.max_preview_px = max_preview_px
        loader.scale = 1.0
        loader.original_size = (0, 0)
        return loader

    def load_image(self) -> Optional[np.ndarray]:
        """Ładuje obraz z pełną walidacją i obsługą błędów."""

        if self.current_index >= len(self.image_files):
            return None

        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])

        # 1. Wstępna walidacja pliku
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Plik nie istnieje: {image_path}")
        if os.path.getsize(image_path) > 100 * 1024 * 1024:  # 100MB
            raise ValueError(f"Plik przekracza 100MB: {image_path}")

        # 2. Ładowanie obrazu
        try:
            self.original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if self.original_image is None:
                raise ValueError("OpenCV nie może zdekodować pliku (może uszkodzony?)")

            # 3. Walidacja wymiarów
            h, w = self.original_image.shape[:2]
            if w * h > 50_000 * 50_000:
                raise ValueError(f"Obraz {w}x{h}px przekracza limit 50,000x50,000px")
            if w * h * 3 > 2 * 1024 * 1024 * 1024:  # ~2GB RAM
                print(f"UWAGA: Duży obraz {w}x{h} (~{(w * h * 3) / 1024 / 1024:.1f}MB RAM)")

            # 4. Skalowanie
            self.image = self._resize_to_preview(self.original_image)
            return self.image

        except MemoryError:
            raise ValueError("Brak pamięci do załadowania obrazu")
        except Exception as e:
            raise ValueError(f"Błąd ładowania {image_path}: {str(e)}")

    def next_image(self) -> Optional[np.ndarray]:
        """Ładuje następny obraz w kolejności"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return self.load_image()
        return None

    def _resize_to_preview(self, image: np.ndarray) -> np.ndarray:
        """Skaluje obraz do `max_preview_px` z zachowaniem proporcji i korektą orientacji EXIF.

        Args:
            image: Oryginalny obraz w formacie BGR (OpenCV)

        Returns:
            Przeskalowany obraz z zachowaniem proporcji (nie powiększa)
        """
        # 1. Korekta orientacji (tylko dla JPEG)
        if self.image_files[self.current_index].lower().endswith(('.jpg', '.jpeg')):
            try:
                orientation = self._get_exif_orientation()
                if orientation in [3, 6, 8]:
                    image = self._apply_orientation(image, orientation)
            except Exception as ex:
                print(f"Uwaga: Błąd EXIF - {str(ex)}")

        # 2. Obliczanie skali
        h, w = image.shape[:2]
        if w == 0 or h == 0:
            raise ValueError("Nieprawidłowe wymiary obrazu (szer/wys == 0)")

        self.scale = min(
            self.max_preview_px / w,
            self.max_preview_px / h,
            1.0  # Nie powiększamy
        )
        self.original_size = (w, h)

        # 3. Wybór interpolacji
        interpolation = (
            cv2.INTER_AREA if self.scale < 1.0
            else cv2.INTER_LINEAR
        )

        # 4. Skalowanie
        return cv2.resize(
            image,
            (int(w * self.scale), int(h * self.scale)),
            interpolation=interpolation
        )

    def _get_exif_orientation(self) -> Optional[int]:
        """Pobiera orientację EXIF z aktualnego pliku (optymalizacja)."""
        image_path = os.path.join(self.image_dir, self.image_files[self.current_index])

        try:
            with open(image_path, 'rb') as f:
                # Czytaj tylko znacznik EXIF Orientation
                tags = exifread.process_file(f, details=False, stop_tag='Orientation')
                if 'Image Orientation' in tags:
                    return int(tags['Image Orientation'].values[0])
        except Exception:
            return None

    def _apply_orientation(self, image: np.ndarray, orientation: int) -> np.ndarray:
        """Stosuje korektę orientacji do obrazu."""
        if orientation == 3:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 6:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def get_original_image(self, copy: bool = True) -> Optional[np.ndarray]:
        """Zwraca oryginalny obraz (opcjonalnie kopię).

        Args:
            copy: Jeśli True (domyślnie), zwraca kopię obrazu.
                  Jeśli False, zwraca referencję (uważaj na modyfikacje!)

        Returns:
            Oryginalny obraz w formacie BGR lub None jeśli brak
        """
        if self.original_image is None:
            return None
        return self.original_image.copy() if copy else self.original_image

    def scale_coords_to_original(self,
                                 x1: int,
                                 y1: int,
                                 x2: int,
                                 y2: int) -> Tuple[int, int, int, int]:
        """Transformuje współrzędne z podglądu do oryginału.

        Args:
            x1, y1: Lewy górny róg
            x2, y2: Prawy dolny róg

        Returns:
            Krotka ze współrzędnymi w oryginalnej skali

        Raises:
            ValueError: Jeśli współrzędne są nieprawidłowe lub skala niezainicjowana
        """
        if not hasattr(self, 'scale') or self.scale <= 0:
            raise ValueError("Skala nie została poprawnie zainicjowana")

        if x1 > x2 or y1 > y2:
            raise ValueError("Nieprawidłowe współrzędne (x1>x2 lub y1>y2)")

        return (
            max(0, int(x1 / self.scale)),
            max(0, int(y1 / self.scale)),
            min(self.original_size[0], int(x2 / self.scale)),
            min(self.original_size[1], int(y2 / self.scale))
        )

    def clear(self) -> None:
        """Zwalnia zasoby pamięci zajmowane przez obecny obraz."""
        self.image = None
        self.original_image = None
        if hasattr(self, 'scale'):
            del self.scale

    @property
    def current_image_path(self) -> Optional[str]:
        """Ścieżka do aktualnego obrazu"""
        if 0 <= self.current_index < len(self.image_files):
            return os.path.join(self.image_dir, self.image_files[self.current_index])
        return None
