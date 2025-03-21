import os
import shutil
import cv2
import unittest
from Otolits_identyfication_program.image_loader import ImageLoader


class TestImageLoader(unittest.TestCase):
    TEST_DIR = "tests/test_images"
    TEST_IMAGE = "tests/test_images/test_image.jpg"

    @classmethod
    def setUpClass(cls):
        """ Tworzy katalog testowy i zapisuje w nim przykładowy obraz """
        os.makedirs(cls.TEST_DIR, exist_ok=True)

        # Tworzymy mały obraz testowy 100x100 px
        test_image = 255 * (cv2.randn(cv2.UMat(100, 100, cv2.CV_8UC3), 0, 255)).get()
        cv2.imwrite(cls.TEST_IMAGE, test_image)

    @classmethod
    def tearDownClass(cls):
        """ Usuwa katalog testowy po zakończeniu testów """
        shutil.rmtree(cls.TEST_DIR, ignore_errors=True)

    def setUp(self):
        """ Inicjalizuje ImageLoader dla każdego testu """
        self.loader = ImageLoader(self.TEST_DIR)

    def test_image_files_listed_correctly(self):
        """ Sprawdza, czy pliki są poprawnie wykrywane i sortowane """
        self.assertGreater(len(self.loader.image_files), 0, "Brak znalezionych plików")
        self.assertIn("test_image.jpg", self.loader.image_files, "Plik testowy nie został wykryty")

    def test_load_image_returns_valid_image(self):
        """ Sprawdza, czy załadowany obraz nie jest pusty """
        image = self.loader.load_image()
        self.assertIsNotNone(image, "Załadowany obraz jest None")
        self.assertGreater(image.shape[0], 0, "Obraz ma niepoprawną wysokość")
        self.assertGreater(image.shape[1], 0, "Obraz ma niepoprawną szerokość")

    def test_screen_scaling(self):
        """ Sprawdza, czy obraz po skalowaniu mieści się na ekranie """
        image = self.loader.load_image()
        screen_w, screen_h = self.loader.screen_width, self.loader.screen_height
        self.assertLessEqual(image.shape[0], screen_h, "Wysokość obrazu przekracza ekran")
        self.assertLessEqual(image.shape[1], screen_w, "Szerokość obrazu przekracza ekran")

    def test_next_image_behavior(self):
        """ Sprawdza, czy metoda next_image() przechodzi do następnego obrazu lub zwraca None na końcu """
        self.loader.load_image()  # Wczytujemy pierwszy obraz
        next_img = self.loader.next_image()
        self.assertIsNone(next_img, "next_image() powinien zwrócić None, ale zwrócił obraz")

    def test_empty_directory(self):
        """ Sprawdza, czy klasa obsługuje pusty katalog """
        empty_dir = "tests/empty_images"
        os.makedirs(empty_dir, exist_ok=True)

        empty_loader = ImageLoader(empty_dir)
        self.assertEqual(len(empty_loader.image_files), 0, "Katalog jest pusty, ale znaleziono pliki")
        self.assertIsNone(empty_loader.load_image(), "Pusty katalog powinien zwrócić None")

        shutil.rmtree(empty_dir, ignore_errors=True)

    def test_nonexistent_directory(self):
        """ Sprawdza, czy klasa obsłuży nieistniejący katalog i zgłosi wyjątek """
        with self.assertRaises(FileNotFoundError):
            ImageLoader("tests/non_existent_folder")

    def test_scale_factor_stored_correctly(self):
        """ Sprawdza, czy ImageLoader poprawnie zapisuje współczynnik skalowania """
        self.loader.load_image()

        original_h, original_w = self.loader.original_image.shape[:2]
        resized_h, resized_w = self.loader.image.shape[:2]

        expected_scale = min(self.loader.screen_width / original_w, self.loader.screen_height / original_h)
        actual_scale = resized_w / original_w  # Skala powinna być taka sama dla obu wymiarów

        self.assertAlmostEqual(actual_scale, expected_scale, places=3,
                               msg=f"Zapisana skala ({actual_scale}) nie zgadza się z oczekiwaną ({expected_scale})")


if __name__ == "__main__":
    unittest.main()
