Instrukcja obsługi programu
Uruchomienie
Uruchom plik main.py.

Program automatycznie załaduje pierwsze zdjęcie z katalogu test_images i uruchomi detekcję obiektów przy użyciu YOLO.
Po detekcji otworzy się dodatkowe okno z wczytanym zdjęciem oraz naniesionymi czerwonymi ramkami (bounding box) wokół wykrytych obiektów (uwaga: za pierwszym razem okno może uruchomić się w zminimalizowanej formie).

Tryb manualny — obsługa
Po detekcji program przechodzi w tryb manualny "dodaj linie".
Jeśli bounding boxy są wyznaczone poprawnie, można od razu zaznaczać wiersze — wszystkie bounding boxy należące do wybranego wiersza zmienią kolor na zielony.
Uwaga: Program wycina tylko zielone bounding boxy.

Klawisze funkcyjne
ENTER — wycięcie obrazów z zaznaczonych (zielonych) bounding boxów.
N — wczytanie kolejnego zdjęcia z katalogu test_images (po wycięciu obrazów).
B — dodawanie nowych bounding boxów.
D — usuwanie bounding boxów oraz linii wyznaczających wiersze.
V — przesuwanie bounding boxów oraz linii wyznaczających wiersze.
R — zmiana rozmiaru bounding boxów.

Lokalizacja wyników
Wycięte zdjęcia zapisywane są w katalogu output_crops.

