# Plan: wycinki A/B, numerowanie wierszy, pseudo-labelling + YOLO compartments

## Kontekst

Każde zdjęcie z otolitami zawiera **dwa fizyczne wycinki** (kartki papieru) ułożone pionowo: **A na górze, B na dole**. Każdy wycinek może mieć **do 3 wierszy** otolitów (typowo 3, ale czasem 1 lub 2).

Obecne `image_cropper.py:crop_and_save` (linie 41–51) używa sztywnego mapowania **globalnego** indeksu wiersza:
- wiersze 1–3 globalnie → `A_1`, `A_2`, `A_3`
- wiersze 4–6 globalnie → `B_1`, `B_2`, `B_3`

**Problem**: jeśli A ma 2 wiersze a B ma 3, to globalne pozycje 1,2 trafiają do A jako `A_1, A_2` (zamiast `A_2, A_3` — bo brakuje "najwyższego" wiersza), a 3,4,5 do B (z czego pierwszy dostaje `B_1` zamiast `B_3`). Numeracja się rozłazi i przestaje odpowiadać fizycznej pozycji w wycinku.

**Celem** jest: (1) niezawodnie identyfikować wycinki A i B, (2) numerować wiersze konsekwentnie w obrębie wycinka niezależnie od liczby wierszy w drugim, (3) zrobić to powtarzalnie przez model ML zamiast heurystyk.

## Decyzje uzgodnione z userem

1. **Reguła numerowania "od dołu"**: najniższy wiersz w wycinku zawsze ma numer `_3`. Każdy kolejny "w górę" — numer o 1 mniejszy.
2. **>3 wierszy w jednym wycinku** → blokada zapisu, komunikat o błędzie. Konwencja sztywna, brak overflowu.
3. **Identyfikacja A/B**: jedna klasa `compartment` w YOLO + sortowanie A/B po Y (najwyższy bbox = A). Plus manual override w UI.
4. **Strategia czasowa: hybryda od początku** — kod ma dwie ścieżki (model + heurystyka) z automatycznym fallbackiem. Płynne przejście.
5. **Dataset do treningu wycinków**: pseudo-labelling. Każda sesja użytkownika produkuje training data automatycznie — heurystyka generuje wstępne bbox-y, user je poprawia, system zapisuje finalne bbox-y w formacie YOLO. Po nazbieraniu 50–100 obrazów → trening modelu.
6. **Format nazwy pliku** — *odłożone do osobnej decyzji*. Domyślnie: zachowujemy obecny wzorzec `<orig><prefix>_<n>.png` bez separatora między oryginałem a prefiksem (kompatybilność z istniejącym archiwum).

---

## Reguła numerowania — formalizacja

Per wycinek (A i B niezależnie):

```python
rows_in_compartment = sorted(rows_of_X, key=lambda r: r.top_y())   # top→bottom
n = len(rows_in_compartment)

if n > 3:
    raise CompartmentOverflowError(f"Wycinek {label} ma {n} wierszy (max 3)")

for offset, row in enumerate(rows_in_compartment):
    row_num = 3 - (n - 1 - offset)            # offset 0 → najwyższy
    # równoważnie: row_num = offset + (4 - n)
    prefix = f"{compartment_label}_{row_num}"
```

Interpretacja `(n - 1 - offset)`: odległość bieżącego wiersza od najniższego. Najniższy ma odległość 0 → numer 3.

### Tabela weryfikacyjna

| n  | offset=0 | offset=1 | offset=2 |
|----|----------|----------|----------|
| 1  | _3       | —        | —        |
| 2  | _2       | _3       | —        |
| 3  | _1       | _2       | _3       |

Przykłady kombinowane:

| Konfiguracja | A         | B         |
|--------------|-----------|-----------|
| A=3, B=3     | A_1,A_2,A_3 | B_1,B_2,B_3 |
| A=2, B=3     | A_2,A_3     | B_1,B_2,B_3 |
| A=3, B=1     | A_1,A_2,A_3 | B_3         |
| A=1, B=2     | A_3         | B_2,B_3     |

---

## Architektura: kaskada detekcji

```
zdjęcie ──► [CompartmentSource.detect()] ──► [Compartment(A, bbox), Compartment(B, bbox)]
                  │
                  ├── YoloCompartmentSource (jeśli model dostępny)
                  └── HeuristicCompartmentSource (fallback: największa luka po Y)

zdjęcie ──► [AutoDetector.detect()] ──► [otolith boxes ...]
                  │
                  ▼
       assign_otoliths_to_compartments(boxes, compartments):
           dla każdego box: jeśli box.center ∈ compartment.bbox → przypisz

       dla każdego compartment:
           rows = detect_rows(compartment.boxes)    # klastrowanie po Y / RANSAC
           # user może edytować w UI

       crop_and_save(compartments, rows, image):
           per compartment, per row → "od dołu" numerowanie
           save PNG do output_dir
           equivalent w formacie YOLO → zapis do annotations_dir
                                        (pseudo-label dla przyszłego treningu)
```

---

## Komponenty kodu

### `compartment.py` (nowy)

```python
from dataclasses import dataclass
from typing import List, Literal
from bounding_box import BoundingBox

@dataclass
class Compartment:
    label: Literal['A', 'B']
    bbox: BoundingBox                            # granice wycinka w obrazie
    boxes: List[BoundingBox]                     # otolity wewnątrz
    rows: List['Row']                            # wiersze wewnątrz (z row_detector)
    source: Literal['yolo', 'heuristic', 'manual']   # do logowania/diagnostyki

    def center_y(self) -> float:
        return (self.bbox.y1 + self.bbox.y2) / 2
```

### `compartment_source.py` (nowy)

```python
from abc import ABC, abstractmethod

class CompartmentSource(ABC):
    @abstractmethod
    def detect(self, image, otolith_boxes) -> List[Compartment]: ...

class YoloCompartmentSource(CompartmentSource):
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, image, otolith_boxes):
        results = self.model(image)[0]
        # filter, sort po Y
        # najwyższy → label='A', niższy → label='B'
        # jeśli mniej niż 2 → AUTO-FALLBACK do heurystyki
        ...

class HeuristicCompartmentSource(CompartmentSource):
    """Najwiksza luka po Y w boxach otolitów"""
    def detect(self, image, otolith_boxes):
        if len(otolith_boxes) < 2:
            # brak otolitów — zwróć pojedynczy compartment A na cały obraz
            return [self._fallback_full_image(image, label='A')]

        y_centers = sorted([b.get_center()[1] for b in otolith_boxes])
        gaps = [(y_centers[i+1] - y_centers[i], i) for i in range(len(y_centers)-1)]
        max_gap, split_idx = max(gaps)

        # bezpiecznik: jeśli max_gap < 1.5× drugiego co do wielkości → wszystkie w jednym
        sorted_gaps = sorted([g[0] for g in gaps], reverse=True)
        if len(sorted_gaps) >= 2 and max_gap < 1.5 * sorted_gaps[1]:
            return [self._fallback_full_image(image, label='A')]

        threshold_y = (y_centers[split_idx] + y_centers[split_idx+1]) / 2
        a_boxes = [b for b in otolith_boxes if b.get_center()[1] < threshold_y]
        b_boxes = [b for b in otolith_boxes if b.get_center()[1] >= threshold_y]

        # bbox compartmentu = obejmuje wszystkie swoje otolity + margin
        ...

def get_compartment_source(model_path=None) -> CompartmentSource:
    if model_path and os.path.exists(model_path):
        return YoloCompartmentSource(model_path)
    return HeuristicCompartmentSource()
```

### Modyfikacja `image_cropper.py`

```python
class CompartmentOverflowError(Exception):
    """Wycinek ma >3 wierszy — naruszenie konwencji"""

class ImageCropper:
    def crop_and_save(self, original_image, compartments: List[Compartment], ...):
        for c in sorted(compartments, key=lambda c: c.center_y()):    # A przed B
            rows_in_c = sorted(c.rows, key=lambda r: min(b.y1 for b in r.boxes))
            n = len(rows_in_c)
            if n > 3:
                raise CompartmentOverflowError(
                    f"Wycinek {c.label}: {n} wierszy (max 3)"
                )

            for offset, row in enumerate(rows_in_c):
                row_num = 3 - (n - 1 - offset)        # "od dołu"
                prefix = f"{c.label}_{row_num}"

                sorted_boxes = sorted(row.boxes, key=lambda b: (b.x1 + b.x2) / 2)
                for box_idx, box in enumerate(sorted_boxes, start=1):
                    filename = f"{orig_name}{prefix}_{box_idx}.png"
                    # ... reszta jak teraz
```

### Pseudo-labelling: zapis adnotacji

```python
def save_annotations_yolo_format(compartments, image_shape, annotations_path):
    """Zapisuje finalne bbox-y wycinków w formacie YOLO."""
    h, w = image_shape[:2]
    lines = []
    for c in compartments:
        # YOLO format: <class> <x_center> <y_center> <width> <height>, normalized 0-1
        cx = (c.bbox.x1 + c.bbox.x2) / 2 / w
        cy = (c.bbox.y1 + c.bbox.y2) / 2 / h
        bw = (c.bbox.x2 - c.bbox.x1) / w
        bh = (c.bbox.y2 - c.bbox.y1) / h
        # Klasa: 0 (jedna klasa "compartment", A/B z pozycji Y)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    annotations_path.write_text("\n".join(lines))
```

Wywoływane z `process_cropping` / `/api/crop` **po pomyślnym zapisie obrazków** — tylko gdy user nie odznaczył checkboxa "Zapisz adnotacje wycinków" (lub przez env `ENABLE_PSEUDO_LABELLING=true`).

---

## Layout danych po wprowadzeniu pseudo-labellingu

```
DATA_ROOT/
├─ images/                       (zdjęcia wejściowe)
│  └─ FLE_NPZDR_2025_3.jpg
├─ annotations/                  (NOWE — adnotacje wycinków w formacie YOLO)
│  ├─ FLE_NPZDR_2025_3.txt
│  └─ classes.txt                (jedna linia: "compartment")
└─ output_crops/                 (wycięte otolity)
   ├─ FLE_NPZDR_2025_3A_1_1.png
   ├─ FLE_NPZDR_2025_3A_1_2.png
   └─ ...
```

Po nazbieraniu 50–100 plików `.txt` w `annotations/` → wystarczy split train/val + uruchomienie skryptu treningowego.

---

## UI dla wycinków (webowy frontend)

W stosunku do planu `web_app.md` dochodzą elementy:

- **Niebieska ramka wokół wycinka A** (label "A" w lewym górnym rogu) i **fioletowa wokół B**.
- **Toggle "Pokaż wycinki"** w toolbarze — żeby nie zaśmiecać widoku.
- **Drag krawędzi ramki wycinka** = redefinicja granic A/B (pseudo-label edit).
- **Klawisz `c`** = włącz tryb edycji wycinków (analogicznie do `b` dla boxów, `l` dla linii).
- **Prawym klikiem na linię wiersza** → menu "Przepnij do A" / "Przepnij do B" (manual override gdy auto źle przypisało).
- **Status bar**: `Wycinek A: 3 wiersze, 9 otolitów | Wycinek B: 2 wiersze, 6 otolitów`.
- **Przed zapisem**: walidacja `n > 3` po stronie frontendu — wyświetlenie ostrzeżenia bez wysyłania `/api/crop`.
- **Modal "Preview"**: lista plików z prefiksami zanim user kliknie "Zapisz".
- **Checkbox "Zapisz adnotacje wycinków"** w toolbarze (domyślnie ON).

---

## Zmiany API webowego (rozszerzenie planu `web_app.md`)

### `POST /api/detect` (rozszerzony response)
```json
{
  "scale": 0.5,
  "compartments": [
    { "label": "A", "bbox": {"x1":..,"y1":..,"x2":..,"y2":..}, "source": "heuristic" },
    { "label": "B", "bbox": {"x1":..,"y1":..,"x2":..,"y2":..}, "source": "heuristic" }
  ],
  "boxes": [
    { "x1":..,"y1":..,"x2":..,"y2":..,"label":"auto","compartment":"A" },
    ...
  ]
}
```

### `POST /api/crop` (rozszerzony payload)
```json
{
  "image_path": "rel/img.jpg",
  "output_dir": "rel/out",
  "scale": 0.5,
  "compartments": [
    { "label": "A", "bbox": {...}, "rows": [
        { "line": {"p1":[..],"p2":[..]}, "boxes": [...] }
    ]},
    { "label": "B", "bbox": {...}, "rows": [...] }
  ],
  "save_annotations": true
}
```

Backend:
1. Per compartment: walidacja `len(rows) <= 3` → 400 `CompartmentOverflowError`.
2. Crop wg nowej reguły numerowania.
3. Jeśli `save_annotations` → zapis `.txt` w `DATA_ROOT/annotations/`.

### Nowy endpoint: `POST /api/detect_compartments` (opcjonalny)
- Body: `{ "path": "rel/img.jpg" }`
- Response: jak wyżej w `compartments`. Pozwala na re-detect po dodaniu manualnym otolitów (gdy user kliknął "Wykryj ponownie").

---

## Trening modelu wycinków (Etap 2)

### `YOLO/compartments_config.yaml`
```yaml
path: /data/annotations            # lub absolute path do datasetu
train: images/train
val: images/val
names:
  0: compartment
```

### `YOLO/train_compartments.py`
```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(
    data="compartments_config.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/compartments",
)
```

### Workflow treningu
1. Po nazbieraniu N obrazów z adnotacjami w `DATA_ROOT/annotations/`:
2. Skrypt podziału `split_dataset.py` (~80/20 train/val).
3. `python train_compartments.py` → wagi w `runs/compartments/.../weights/best.pt`.
4. Kopiowanie wag do `YOLO/weights/compartments.pt`.
5. Restart kontenera webowego — `YoloCompartmentSource` aktywuje się automatycznie (sprawdza `os.path.exists`).

---

## Roadmap implementacji

### Faza A: drobne porządki (~1h)
1. `image_cropper.py` — usunąć martwe `display_row_num`, naprawić type hint `List['Row']`, dodać guard pustego wiersza.

### Faza B: kaskada + heurystyka (~1–2 dni)
2. Nowa klasa `Compartment` i `CompartmentSource`.
3. `HeuristicCompartmentSource` z algorytmem największej luki + bezpiecznik.
4. `assign_otoliths_to_compartments`.
5. Zmiana `image_cropper.py` na nową regułę numerowania, z błędem przy n>3.
6. Detektor wierszy (klastrowanie po Y) — `RowDetector.auto_detect_rows(boxes)`.

### Faza C: web UI dla wycinków (~2–3 dni)
7. Rozszerzenie API (`/api/detect`, `/api/crop`).
8. Frontend: rendering ramek wycinków, klawisz `c`, prawym klikiem A/B toggle, status bar, modal preview.
9. Walidacja `n > 3` w UI z czytelnym komunikatem.

### Faza D: pseudo-labelling (~1 dzień)
10. Zapis adnotacji wycinków w `save_annotations_yolo_format`.
11. Checkbox "Zapisz adnotacje" w UI.
12. Env var `ANNOTATIONS_DIR`, w Dockerze volume mount.

### Faza E: trening i wymiana (off-line, ~tygodnie, równolegle do produkcji)
13. Skrypt `split_dataset.py`.
14. Trening `train_compartments.py`.
15. Walidacja jakości modelu (mAP, precision/recall na zbiorze trzymanym).
16. Wymiana — `MODEL_PATH_COMPARTMENTS` w env, `YoloCompartmentSource` aktywuje się.

### Faza F: stabilizacja
17. Auto-fallback `YoloCompartmentSource` → heurystyka gdy model zwraca <2 wycinków.
18. Telemetria (logowanie `source: yolo|heuristic|manual` per zdjęcie) — żeby wiedzieć kiedy model się myli.
19. Re-trening na poszerzonym datasecie (już z `source=manual` korektami).

---

## Otwarte pytania (do późniejszego rozstrzygnięcia)

1. **Format nazwy pliku**: zachować `<orig><prefix>_<n>.png` bez separatora, czy zmienić? (Domyślnie zachowujemy, ale można później dodać opcję konfiguracyjną.)
2. **Auto-bbox compartmentu z heurystyki**: czy obejmować tylko otolity z marginesem, czy próbować wykryć "fizyczne krawędzie kartonika"? Druga opcja wymaga preprocessing (Canny + kontury).
3. **Co gdy żaden otolit nie zostanie wykryty** (np. zdjęcie złej jakości): heurystyka nie ma czego klastrować. Fallback: cały obraz jako jeden compartment "A", user dorysowuje boxy manualnie.
4. **Co gdy compartment YOLO zwraca >2 bbox-ów** (rzadkie, ale możliwe): wybrać 2 największe + sort po Y.
5. **UI klawiszologia konflikty**: `c` dla compartment editing — czy nie koliduje z innymi planowanymi skrótami (`Esc` zostaje).
6. **Walidacja przy zapisie annotation**: czy nie zapisywać gdy `source='heuristic'` (bez user edit)? Filozofia: zapisuj tylko gdy user faktycznie potwierdził/poprawił.