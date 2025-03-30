import uuid
import math


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, label=None):
        """Inicjalizacja boxa z automatyczną normalizacją współrzędnych i walidacją"""
        self.x1, self.x2 = sorted([float(x1), float(x2)])
        self.y1, self.y2 = sorted([float(y1), float(y2)])
        self._validate_coordinates()
        self.label = label
        self.id = str(uuid.uuid4())  # Unikalny identyfikator
        self.color = (0, 255, 0)  # Domyślny kolor (zielony)
        self.selected = False

    def _validate_coordinates(self):
        """Walidacja współrzędnych boxa"""
        if self.x1 == self.x2 or self.y1 == self.y2:
            raise ValueError("Bounding box nie może mieć zerowej szerokości/wysokości")
        if self.x1 < 0 or self.y1 < 0 or self.x2 < 0 or self.y2 < 0:
            raise ValueError("Współrzędne nie mogą być ujemne")

    def __str__(self):
        return f"BoundingBox({self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}, label={self.label})"

    def __repr__(self):
        return self.__str__()

    def update(self, x1, y1, x2, y2):
        """Aktualizuje współrzędne boxa z normalizacją i walidacją"""
        self.x1, self.x2 = sorted([float(x1), float(x2)])
        self.y1, self.y2 = sorted([float(y1), float(y2)])
        self._validate_coordinates()

    def contains(self, x, y, tolerance=0):
        """Sprawdza czy punkt (x,y) jest w boxie z tolerancją marginesu"""
        return (self.x1 - tolerance <= x <= self.x2 + tolerance and
                self.y1 - tolerance <= y <= self.y2 + tolerance)

    def intersects(self, other):
        """Sprawdza czy boxy się nakładają"""
        return not (self.x2 < other.x1 or self.x1 > other.x2 or
                    self.y2 < other.y1 or self.y1 > other.y2)

    def get_coordinates(self):
        """Zwraca współrzędne jako krotkę (x1, y1, x2, y2)"""
        return self.x1, self.y1, self.x2, self.y2

    def get_center(self):
        """Zwraca środek boxa jako (x, y)"""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def area(self):
        """Oblicza pole powierzchni boxa"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self):
        """Zwraca szerokość boxa"""
        return self.x2 - self.x1

    def height(self):
        """Zwraca wysokość boxa"""
        return self.y2 - self.y1

    def aspect_ratio(self):
        """Oblicza stosunek szerokości do wysokości"""
        return self.width() / self.height()

    def move(self, dx, dy):
        """Przesuwa box o wektor (dx, dy)"""
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy

    def resize(self, corner, new_x, new_y):
        """
        Zmienia rozmiar boxa poprzez przeciąganie określonego rogu
        :param corner: który róg jest przeciągany (1:lewy-górny, 2:prawy-górny, 3:lewy-dolny, 4:prawy-dolny)
        :param new_x: nowa pozycja x rogu
        :param new_y: nowa pozycja y rogu
        """
        if corner == 1:  # lewy górny
            self.x1, self.y1 = new_x, new_y
        elif corner == 2:  # prawy górny
            self.x2, self.y1 = new_x, new_y
        elif corner == 3:  # lewy dolny
            self.x1, self.y2 = new_x, new_y
        elif corner == 4:  # prawy dolny
            self.x2, self.y2 = new_x, new_y

        # Normalizacja współrzędnych
        self.x1, self.x2 = sorted([self.x1, self.x2])
        self.y1, self.y2 = sorted([self.y1, self.y2])
        self._validate_coordinates()

    def scale(self, factor):
        """Skaluje box względem jego środka"""
        center_x, center_y = self.get_center()
        width = self.width() * factor
        height = self.height() * factor

        self.x1 = center_x - width / 2
        self.x2 = center_x + width / 2
        self.y1 = center_y - height / 2
        self.y2 = center_y + height / 2

    def get_corners(self):
        """Zwraca współrzędne wszystkich rogów boxa"""
        return [
            (self.x1, self.y1),  # lewy górny
            (self.x2, self.y1),  # prawy górny
            (self.x1, self.y2),  # lewy dolny
            (self.x2, self.y2)  # prawy dolny
        ]

    def get_nearest_corner(self, x, y):
        """Znajduje najbliższy róg do podanego punktu i zwraca jego indeks"""
        corners = self.get_corners()
        distances = [math.hypot(cx - x, cy - y) for cx, cy in corners]
        return distances.index(min(distances)) + 1  # +1 bo indeksy od 1 do 4

    def draw(self, image, color=None, thickness=2):
        """Rysuje box na obrazie"""
        draw_color = color if color else self.color
        if self.selected:
            thickness = max(3, thickness)  # Pogrubienie dla zaznaczonego boxa
            draw_color = (0, 0, 255)  # Czerwony dla zaznaczonego boxa

        cv2.rectangle(image,
                      (int(self.x1), int(self.y1)),
                      (int(self.x2), int(self.y2)),
                      draw_color, thickness)

    def to_dict(self):
        """Konwertuje box do słownika (do zapisu)"""
        return {
            'id': self.id,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'label': self.label,
            'color': self.color
        }

    @classmethod
    def from_dict(cls, data):
        """Tworzy box ze słownika (z pliku)"""
        box = cls(data['x1'], data['y1'], data['x2'], data['y2'], data.get('label'))
        if 'color' in data:
            box.color = tuple(data['color'])
        return box

    def copy(self):
        """Tworzy kopię boxa"""
        return BoundingBox(self.x1, self.y1, self.x2, self.y2, self.label)

