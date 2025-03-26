import uuid


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, label=None):
        """Inicjalizacja boxa z automatyczną normalizacją współrzędnych"""
        if x1 == x2 or y1 == y2:
            raise ValueError("Bounding box nie może mieć zerowej szerokości/wysokości")

        self.x1, self.x2 = sorted([x1, x2])
        self.y1, self.y2 = sorted([y1, y2])
        self.label = label
        self.id = str(uuid.uuid4())  # Unikalny identyfikator

    def __str__(self):
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2}, label={self.label})"

    def update(self, x1, y1, x2, y2):
        """Aktualizuje współrzędne boxa z normalizacją"""
        if x1 == x2 or y1 == y2:
            raise ValueError("Bounding box nie może mieć zerowej szerokości/wysokości")
        self.x1, self.x2 = sorted([x1, x2])
        self.y1, self.y2 = sorted([y1, y2])

    def contains(self, x, y, tolerance=0):
        """Sprawdza czy punkt (x,y) jest w boxie z tolerancją marginesu"""
        return (self.x1 - tolerance <= x <= self.x2 + tolerance and
                self.y1 - tolerance <= y <= self.y2 + tolerance)

    def get_coordinates(self):
        """Zwraca współrzędne jako krotkę (x1, y1, x2, y2)"""
        return self.x1, self.y1, self.x2, self.y2

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

    def resize(self, x1, y1, x2, y2):
        """Zmienia rozmiar zachowując środek boxa"""
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        new_width = x2 - x1
        new_height = y2 - y1
        self.x1 = center_x - new_width / 2
        self.x2 = center_x + new_width / 2
        self.y1 = center_y - new_height / 2
        self.y2 = center_y + new_height / 2

    def to_dict(self):
        """Konwertuje box do słownika (do zapisu)"""
        return {
            'id': self.id,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data):
        """Tworzy box ze słownika (z pliku)"""
        return cls(data['x1'], data['y1'], data['x2'], data['y2'], data.get('label'))