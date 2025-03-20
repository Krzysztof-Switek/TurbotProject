import sys
import unittest
import os
import sys
import cv2
from unittest.mock import patch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_window import ImageWindow
from mouse_handler import MouseHandler


class TestImageWindow(unittest.TestCase):
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_show_scaled_image(self, mock_waitkey, mock_imshow):
        """
        Testuje, czy metoda show poprawnie wyświetla przeskalowany obraz.
        """
        mouse_handler = MouseHandler(None)
        image_window = ImageWindow(mouse_handler, "Test Window")

        original_image = np.zeros((100, 100, 3), dtype=np.uint8)
        scaled_image = cv2.resize(original_image, (50, 50))  # Przeskalowany obraz

        mock_waitkey.return_value = 27
        image_window.show(scaled_image)

        mock_imshow.assert_called_with("Test Window", scaled_image)

    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_show_original_image(self, mock_waitkey, mock_imshow):
        """
        Testuje, czy metoda show poprawnie wyświetla oryginalny obraz.
        """
        mouse_handler = MouseHandler(None)
        image_window = ImageWindow(mouse_handler, "Test Window")

        original_image = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_waitkey.return_value = 27
        image_window.show(original_image)

        mock_imshow.assert_called_with("Test Window", original_image)


if __name__ == '__main__':
    unittest.main()
