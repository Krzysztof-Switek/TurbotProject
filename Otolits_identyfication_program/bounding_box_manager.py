class BoundingBoxManager:
    def __init__(self):
        self.boxes = []  # Lista przechowująca współrzędne boxów

    def add_box(self, x1, y1, x2, y2):
        """Dodaje nowy box do listy"""
        self.boxes.append((x1, y1, x2, y2))

    def remove_box(self, x, y):
        """Usuwa box, jeśli kliknięto w jego obrębie"""
        for box in self.boxes:
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.boxes.remove(box)
                break

    def edit_box(self, index, new_x1, new_y1, new_x2, new_y2):
        """Edytuje istniejący box"""
        if 0 <= index < len(self.boxes):
            self.boxes[index] = (new_x1, new_y1, new_x2, new_y2)

    def get_boxes(self):
        """Zwraca listę boxów"""
        return self.boxes


############## TESTY ##############

if __name__ == "__main__":
    manager = BoundingBoxManager()

    # Test dodawania boxów
    manager.add_box(10, 10, 50, 50)
    manager.add_box(60, 60, 100, 100)
    assert len(manager.get_boxes()) == 2
    print("✅ Dodawanie boxów działa poprawnie!")

    # Test usuwania boxów
    manager.remove_box(15, 15)  # Wewnątrz pierwszego boxa
    assert len(manager.get_boxes()) == 1
    print("✅ Usuwanie boxów działa poprawnie!")

    # Test edycji boxów
    manager.edit_box(0, 70, 70, 120, 120)
    assert manager.get_boxes()[0] == (70, 70, 120, 120)
    print("✅ Edycja boxów działa poprawnie!")
