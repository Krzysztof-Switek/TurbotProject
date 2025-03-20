import cv2

class ImageWindow:
    def __init__(self, mouse_handler, window_name):
        """
        Klasa obsługująca okno do wyświetlania obrazów z obsługą myszy.

        :param mouse_handler: Obiekt obsługujący myszkę (MouseHandler).
        :param window_name: Nazwa okna.
        """
        self.window_name = window_name

        # Tworzymy okno i ustawiamy obsługę myszy
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, mouse_handler.mouse_callback)

    def show(self, image):
        """
        Wyświetla obraz i obsługuje interakcję użytkownika.

        :param image: Obraz w formacie numpy.ndarray.
        """
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(0)  # Czekamy na klawisz

        if key == 27:  # ESC - zamyka cały program
            print("Zamykanie programu...")
            cv2.destroyAllWindows()
            exit()
        else:
            cv2.destroyWindow(self.window_name)  # Zamykamy okno dla aktualnego obrazu

    def close(self):
        """Zamyka okno."""
        cv2.destroyWindow(self.window_name)
