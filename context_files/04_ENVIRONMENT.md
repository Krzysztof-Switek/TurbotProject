# 04 — Environment & Non-code Context

## Requirements — Otolits_identyfication_program
`Otolits_identyfication_program/requirements.txt`

```text
# Web framework
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart

# Computer vision + ML
# WAŻNE: headless wariant OpenCV (brak GUI deps, mniejszy obraz Docker).
# NIE instalować razem z opencv-python — paczki konfliktują.
# Dla developmentu desktop poza Dockerem zainstaluj regular opencv-python
# w innym virtualenvie.
opencv-python-headless>=4.9
numpy
Pillow
ultralytics>=8.1
exifread
```

## pyvenv.cfg
`pyvenv.cfg`

```ini
home = C:\Users\kswitek\AppData\Local\Programs\Python\Python310
include-system-site-packages = false
version = 3.10.7
```

## Config — Otolits_identyfication_program/YOLO/config.yaml
`Otolits_identyfication_program/YOLO/config.yaml`

```yaml

path: C:\Users\mtiurina\source\repos\airo\DataSets\otolith_cuts_marks_1280_labeled
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: mark
  1: otolith
```

## Config — Otolits_identyfication_program/YOLO/datasets/turbot.yaml
`Otolits_identyfication_program/YOLO/datasets/turbot.yaml`

```yaml
# path: C:\Users\kswitek\Documents\TurbotProject\Otolits_identyfication_program\YOLO\datasets\turbot_dataset

path: /home/kswitek/Documents/TurbotProject/Otolits_identyfication_program/YOLO/datasets/turbot_dataset

train: images/train
val: images/val
test: images/test

nc: 1  # Liczba klas
names: ['Otolit']  # Nazwa klasy

```

## Planning / audit documents (`audyty_plany/`)

| Document | Top heading |
|----------|-------------|
| `audyty_plany/19.06_skala_plan_TO_DO.md` | Skala — audyt + plan TO-DO (2026-06-19) |
| `audyty_plany/19.06_skala_podsumowanie_wdrozenia.md` | Skala — podsumowanie wdrożenia (2026-06-19) |
| `audyty_plany/STAN_WEB_2026_06_02.md` | Stan projektu — sesja 2026-06-02 |
| `audyty_plany/TO_DO_WEB.MD` | TO_DO_WEB — stan sesji + plan dalszych kroków |
| `audyty_plany/kontekst_projektu.md` | TurbotProject — Kontekst projektu (dokument referencyjny) |
| `audyty_plany/web_app.md` | Plan: konwersja TurbotProject na narzędzie webowe (FastAPI + vanilla JS + Docker) |
| `audyty_plany/wycinki_i_numerowanie.md` | Plan: wycinki A/B, numerowanie wierszy, pseudo-labelling + YOLO compartments |
