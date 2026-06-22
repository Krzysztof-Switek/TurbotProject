# Turbot — wycinanie otolitów

Narzędzie webowe (działa w przeglądarce) do **wycinania otolitów ze zdjęć**.
Program sam znajduje otolity na zdjęciu, układa je w wiersze i dwa wycinki
(górny **A** i dolny **B**), a następnie zapisuje każdy otolit jako osobny
obrazek PNG — opcjonalnie z paskiem skali.

---

## Część 1 — Instrukcja dla osoby pracującej przy komputerze

> Ta część jest dla Ciebie, jeśli siadasz do komputera i masz wyciąć otolity.
> Zakłada, że program został już uruchomiony przez informatyka (patrz Część 2).

### Jak otworzyć program

Otwórz przeglądarkę (Chrome, Edge, Firefox) i wpisz adres podany przez osobę,
która uruchomiła program — najczęściej:

```
http://localhost:8000
```

Zobaczysz jedno okno podzielone na trzy części:

- **Lewy panel** — wybór skali oraz przeglądanie folderów (skąd brać zdjęcia
  i gdzie zapisywać wyniki).
- **Środek** — duży obszar ze zdjęciem oraz pasek przycisków na górze.
- **Dół** — pasek stanu z informacjami (która skala, ile wierszy, ewentualne błędy).

### Najkrótsza droga: jak wyciąć otolity (krok po kroku)

1. **Wybierz zdjęcie.** W lewym panelu, w sekcji **Source**, klikaj foldery, aż
   dojdziesz do swoich zdjęć, i kliknij zdjęcie.
2. **Poczekaj chwilę — program zrobi resztę sam.** Po kliknięciu automatycznie:
   - obrysuje znalezione otolity ramkami,
   - podzieli zdjęcie na wycinek **A** (góra) i **B** (dół),
   - poukłada otolity w wiersze i podpisze je (`A_1`, `A_2`, `A_3`, `B_1`…),
   - ramki otolitów, które trafiły do wiersza, robią się **zielone**.

   > **Ważne: program wycina tylko zielone ramki.** Ramka **czerwona** = otolit
   > znaleziony, ale nie przypisany do żadnego wiersza — nie zostanie wycięty.

3. **Sprawdź wynik.** Jeśli wszystko wygląda dobrze (otolity obrysowane,
   pogrupowane, zielone) — przejdź od razu do punktu 5. Jeśli coś trzeba poprawić
   — użyj trybów ręcznych (punkt 4).
4. **Popraw ręcznie (jeśli trzeba).** Kliknij przycisk trybu na górze (albo wciśnij
   klawisz w nawiasie), potem działaj myszką na zdjęciu:
   - **`b` add box** — domaluj brakującą ramkę: przeciągnij myszką wokół otolitu,
     którego program nie złapał.
   - **`l` add line** — narysuj linię wiersza: przeciągnij ją wzdłuż rzędu otolitów.
     Otolity przecięte linią dołączają do tego wiersza i zmieniają kolor na zielony.
   - **`v` move** — przesuń ramkę lub linię (chwyć i przeciągnij).
   - **`r` resize** — zmień rozmiar ramki (chwyć za róg i przeciągnij).
   - **`d` del** — usuń ramkę lub linię (kliknij ją).
   - **`e` edit label** — kliknij linię wiersza, aby ręcznie wskazać, czy należy
     do wycinka **A**, **B**, czy ma zdecydować program (`auto`).
5. **Wskaż, gdzie zapisać.** W lewym panelu, w sekcji **Destination dir**, wejdź do
   folderu, w którym mają się pojawić wycięte otolity, i kliknij **save here**.
   Wybrany folder pokaże się jako bieżący cel zapisu.
6. **Wytnij i zapisz.** Kliknij **Crop and save** na górze (lub wciśnij **Enter**).
   Program wytnie każdy zielony otolit do osobnego pliku PNG i pokaże podsumowanie
   (ile plików zapisał).
7. **Następne zdjęcie.** Po prostu kliknij kolejne zdjęcie w sekcji **Source**.

### Przyciski na górnym pasku — ściąga

**Tryby pracy myszką** (po lewej):

| Przycisk | Klawisz | Co robi |
|----------|:-------:|---------|
| add box    | `b` | dorysowanie ramki wokół otolitu |
| add line   | `l` | dorysowanie linii wiersza (grupuje otolity) |
| move       | `v` | przesuwanie ramki lub linii |
| resize     | `r` | zmiana rozmiaru ramki |
| del        | `d` | usuwanie ramki lub linii (kliknięciem) |
| edit label | `e` | ręczne przypisanie wiersza do wycinka A / B / auto |

**Akcje** (po prawej):

| Przycisk | Co robi |
|----------|---------|
| **swap slices** (`w`) | zamienia wycinki miejscami (A↔B) — gdy program odwrotnie rozpoznał górę i dół |
| **Advanced rows: OFF/ON** | włącz, gdy otolity leżą **na skos** i wiersze się zlewają — program wykryje wtedy pochylone wiersze. Ustawienie działa do zamknięcia programu |
| **Clear rows** | usuwa wszystkie linie wierszy i komunikat błędu; **ramki zostają** |
| **Reload** | wczytuje zdjęcie od nowa (ponawia automatyczne wykrywanie, przywraca skasowane ramki) |
| **Auto-detect (YOLO)** | ponawia automatyczne wykrywanie i dorzuca ramki (przydatne po ręcznych poprawkach) |
| **Save annotations** (haczyk) | zostaw zaznaczony — zapisuje dane pomocnicze do dalszego uczenia programu |
| **Crop and save** (`Enter`) | wycina zielone otolity do wybranego folderu |

### Skala — pasek skali na wycinkach (opcjonalnie)

Jeśli chcesz, by na każdym wyciętym otolicie pojawił się pasek skali (np. „500 μm”),
ustaw raz skalę i wybieraj ją z listy:

1. W lewym panelu, sekcja **Scale**, kliknij **＋ New**.
2. Wgraj zdjęcie linijki / wzorca (kliknij pole lub przeciągnij plik).
3. W razie potrzeby powiększ widok (`−` / `＋` / `Fit`), żeby precyzyjnie kliknąć.
4. Kliknij **dwa punkty** na znanym odcinku linijki (np. końce 1 cm).
5. Wpisz **nazwę**, **długość rzeczywistą** odcinka i jednostkę (cm / mm / μm),
   ewentualnie powiększenie mikroskopu → **Save scale**.
6. Skala stanie się aktywna i będzie zapamiętana. Z listy możesz przełączać skale,
   a 🗑 usuwa zaznaczoną.

> Skala jest poprawna tylko dla zdjęć robionych przy **tym samym powiększeniu**, co
> wzorzec. Bez wybranej skali wycinanie nadal działa — tylko bez paska skali.

### Gdzie trafiają wyniki

Wycięte otolity zapisują się w folderze wskazanym przyciskiem **save here**.
Nazwa każdego pliku mówi, skąd otolit pochodzi, np.:

```
<nazwa_zdjęcia>A_2_1.png
                │ │ └ numer otolitu w wierszu (od lewej)
                │ └── numer wiersza w wycinku
                └──── wycinek: A (górny) lub B (dolny)
```

### Co zrobić, gdy coś nie wygląda dobrze

- **Za dużo / za mało ramek** → domaluj (`b`) lub usuń (`d`) ręcznie.
- **Otolity zostają czerwone (nie wycinają się)** → nie trafiły do wiersza:
  dorysuj linię (`l`) przez ich rząd albo przesuń istniejącą linię (`v`).
- **Wiersze się zlewają / otolity leżą na skos** → włącz **Advanced rows: ON**.
- **Góra i dół zamienione (A zamiast B)** → kliknij **swap slices**.
- **Czerwony komunikat o błędzie na dole** (np. „za dużo wierszy”) → usuń nadmiarowe
  linie (`d`) lub kliknij **Clear rows** i rozłóż wiersze od nowa.
- **Chcesz zacząć zdjęcie od zera** → kliknij **Reload**.

---

## Część 2 — Uruchomienie i wdrożenie (dla informatyka)

Aplikacja to serwer **FastAPI + Uvicorn** z frontendem w czystym HTML/JS
(bez kroku budowania). Cały kod jest w `Otolits_identyfication_program/`,
a wytrenowany model YOLO (`YOLO/weights/best.pt`) jest dołączony do repozytorium.

### Wymagania

- Python **3.10+** (testowane na 3.10–3.13).
- Zależności w `Otolits_identyfication_program/requirements.txt`
  (FastAPI, Uvicorn, OpenCV-headless, NumPy, Pillow, Ultralytics/YOLO, exifread).

### Instalacja

```powershell
cd Otolits_identyfication_program
python -m venv ..\.venv-web
..\.venv-web\Scripts\python.exe -m pip install -r requirements.txt
```

(na Linux/macOS analogicznie: `python3 -m venv ../.venv-web` i `source ../.venv-web/bin/activate`)

### Uruchomienie

Serwer **musi** być uruchamiany z katalogu `Otolits_identyfication_program/`
(importy zakładają ten katalog jako roboczy):

```powershell
# Windows / PowerShell
$env:DATA_ROOT     = "C:\sciezka\do\zdjec"            # bazowy katalog danych
$env:ALLOWED_ROOTS = "Dane=C:\sciezka\do\zdjec"        # katalogi widoczne w przeglądarce plików
Set-Location "C:\...\TurbotProject\Otolits_identyfication_program"
..\.venv-web\Scripts\python.exe -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

```bash
# Linux / macOS
export DATA_ROOT=/dane/zdjecia
export ALLOWED_ROOTS="Dane=/dane/zdjecia"
cd TurbotProject/Otolits_identyfication_program
../.venv-web/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Następnie otwórz `http://<adres-serwera>:8000`. Przy starcie aplikacja wypisuje w
logu swoją konfigurację oraz informację, czy model YOLO został wczytany.

### Zmienne środowiskowe

| Zmienna | Domyślnie | Znaczenie |
|---------|-----------|-----------|
| `DATA_ROOT` | `/data` | Bazowy katalog danych; w nim powstaje też `_scales/` (biblioteka skal). |
| `ALLOWED_ROOTS` | `data=DATA_ROOT` | Lista nazwanych katalogów dostępnych w przeglądarce plików, format `Nazwa=ścieżka;Nazwa2=ścieżka2`. Tylko te katalogi (i ich podkatalogi) są widoczne — zabezpieczenie przed wyjściem poza dozwolony obszar. |
| `MODEL_PATH` | `YOLO/weights/best.pt` | Ścieżka do wytrenowanego modelu YOLO. |
| `MAX_PREVIEW_PX` | `1920` | Maks. rozmiar podglądu wysyłanego do przeglądarki (większy = ostrzejszy, ale wolniejszy). |
| `SCALES_DIR` | `DATA_ROOT/_scales` | Katalog biblioteki skal i kopii zdjęć wzorców. |
| `ALLOWED_EXTS` | `.jpg,.jpeg,.png` | Obsługiwane formaty zdjęć. |

### Uwagi wdrożeniowe

- Aplikacja jest **bezstanowa** (stan pracy żyje w przeglądarce) i przewidziana
  dla pojedynczego użytkownika — bez logowania. Do udostępnienia w sieci postaw ją
  za reverse-proxy / w kontenerze i ogranicz dostęp.
- Przeglądarka plików operuje na dysku **serwera**, nie komputera użytkownika.
- Wyniki cropu zapisywane są w katalogu docelowym wskazanym w interfejsie (musi
  leżeć w obrębie `ALLOWED_ROOTS`).

---

## Struktura projektu (skrót)

```
Otolits_identyfication_program/
├─ web/                  # aplikacja webowa (FastAPI + frontend)
│  ├─ app.py             # punkt wejścia serwera
│  ├─ config.py          # konfiguracja ze zmiennych środowiskowych
│  ├─ routers/           # endpointy API (pliki, podgląd, detekcja, crop, skale)
│  ├─ services/          # logika (przeglądarka plików, obrazy, skale)
│  └─ static/            # interfejs: index.html, app.js, style.css
├─ auto_detector.py      # wykrywanie otolitów modelem YOLO
├─ image_cropper.py      # wycinanie otolitów + pasek skali
├─ image_loader.py       # wczytywanie i skalowanie zdjęć
├─ row_detector.py       # wykrywanie wierszy
├─ YOLO/weights/best.pt  # wytrenowany model (dołączony)
└─ requirements.txt      # zależności
```