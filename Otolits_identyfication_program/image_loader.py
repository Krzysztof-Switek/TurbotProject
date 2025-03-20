import cv2

class ImageLoader:
    def __init__(self, image_path, scale_factor=0.25):
        """
        Klasa do ładowania i skalowania obrazów.

        :param image_path: Ścieżka do pliku obrazu.
        :param scale_factor: Współczynnik skalowania obrazu (domyślnie 0.25).
        """
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.original_image = self._load_image()
        self.scaled_image = self._resize_image(self.original_image)

    def _load_image(self):
        """Wczytuje obraz z pliku."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {self.image_path}")
        return image

    def _resize_image(self, image):
        """Przeskalowuje obraz według współczynnika."""
        height, width = image.shape[:2]
        new_size = (int(width * self.scale_factor), int(height * self.scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def get_image(self, scaled=True):
        """
        Zwraca obraz: przeskalowany lub oryginalny.

        :param scaled: Jeśli True, zwraca obraz przeskalowany, inaczej oryginalny.
        """
        return self.scaled_image if scaled else self.original_image


# ************  TESTY   ************
if __name__ == "__main__":
    print("🔹 Testowanie ImageLoader...")

    test_image = "test_images/TUR_BITS_2015_Q1_1.jpg"

    # Test 1: Czy obraz się wczytuje?
    loader = None  # 👈 Ustawiamy jako None, żeby uniknąć NameError w razie błędu
    try:
        loader = ImageLoader(test_image)
        assert loader.original_image is not None, "Błąd: Obraz nie został wczytany!"
        print("✅ Obraz wczytany poprawnie!")
    except FileNotFoundError:
        print(f"❌ Błąd: Plik {test_image} nie istnieje!")

    # Test 2: Czy obraz jest poprawnie skalowany?
    if loader:
        expected_size = (int(loader.original_image.shape[1] * 0.25), int(loader.original_image.shape[0] * 0.25))
        assert loader.scaled_image.shape[:2] == expected_size[::-1], "Błąd: Niepoprawny rozmiar skalowanego obrazu!"
        print("✅ Skalowanie działa poprawnie!")

    # Test 3: Obsługa błędnej ścieżki
    try:
        ImageLoader("nie_istnieje.jpg")
        print("❌ Błąd: Powinien być wyjątek dla braku pliku!")
    except FileNotFoundError:
        print("✅ Obsługa błędnej ścieżki działa poprawnie!")