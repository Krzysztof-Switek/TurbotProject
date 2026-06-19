"""Konfiguracja aplikacji webowej — odczyt env vars z domyślnymi.

Wszystkie wartości są niezmienne po starcie procesu (czytane raz). FastAPI app
i routery importują stałe zdefiniowane tu zamiast wołać os.environ bezpośrednio.
"""

import os
from pathlib import Path

# Katalog główny — fallback root nawigacji + domyślna lokalizacja SCALES_DIR.
# W Dockerze montowany jako volume (np. -v /host/data:/data).
DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", "/data")).resolve()


def _parse_allowed_roots(raw: str) -> dict[str, Path]:
    """Parsuje 'Nazwa=ścieżka;Nazwa2=ścieżka2' → {nazwa: Path}. Puste/niepełne pomija."""
    roots: dict[str, Path] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, path = chunk.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or not path:
            continue
        roots[name] = Path(path).resolve()
    return roots


# Allow-lista nazwanych katalogów-rootów dostępnych w przeglądarce plików.
# Każdy root = bezpieczna „wyspa": nawigacja confined pod root (guard
# fs_browser.safe_resolve), absolutne ścieżki serwera nie wyciekają do klienta.
# Ścieżki w API mają format `<RootName>/pod/katalog`; pusty `` = wirtualny top
# listujący roots. Domyślnie jeden root `data` = DATA_ROOT (zgodność wstecz).
# Przykład: ALLOWED_ROOTS="Documents=C:\\Users\\me\\Documents;Crops=D:\\crops"
ALLOWED_ROOTS: dict[str, Path] = (
    _parse_allowed_roots(os.environ.get("ALLOWED_ROOTS", "")) or {"data": DATA_ROOT}
)

# Ścieżka do wytrenowanego modelu YOLO (otolity). Ładowany raz przy starcie
# aplikacji do app.state.detector.
MODEL_PATH: Path = Path(
    os.environ.get(
        "MODEL_PATH",
        # Default: model w katalogu programu, ścieżka relatywna do cwd uruchomienia.
        str(Path(__file__).parent.parent / "YOLO" / "weights" / "best.pt"),
    )
).resolve()

# Maksymalna szerokość/wysokość obrazu preview wysyłanego do przeglądarki.
# ImageLoader skaluje do tego limitu zachowując proporcje. Większe wartości
# = ostrzejszy podgląd, ale wolniejsze ładowanie.
MAX_PREVIEW_PX: int = int(os.environ.get("MAX_PREVIEW_PX", "1920"))

# Katalog biblioteki nazwanych skal (presetów μm/px) + kopii zdjęć wzorców.
# Trzymany POD DATA_ROOT, żeby istniejący endpoint /api/image/preview mógł
# serwować wgrane zdjęcia linijek (potrzebny współczynnik X-Scale do przeliczenia
# pomiaru z przestrzeni preview na oryginał). Pliki: `<slug>.json` + `<uuid>.<ext>`.
SCALES_DIR: Path = Path(
    os.environ.get("SCALES_DIR", str(DATA_ROOT / "_scales"))
).resolve()

# Rozszerzenia plików obrazowych pokazywanych w listingu katalogów.
# Dopuszczalne formaty wejściowe.
ALLOWED_EXTS: frozenset[str] = frozenset(
    ext.strip().lower()
    for ext in os.environ.get("ALLOWED_EXTS", ".jpg,.jpeg,.png").split(",")
    if ext.strip()
)


def describe() -> str:
    """Tekstowy dump konfiguracji do logów startupu — sanity check."""
    return (
        f"DATA_ROOT={DATA_ROOT}\n"
        f"ALLOWED_ROOTS={ {n: str(p) for n, p in ALLOWED_ROOTS.items()} }\n"
        f"MODEL_PATH={MODEL_PATH} (exists={MODEL_PATH.exists()})\n"
        f"MAX_PREVIEW_PX={MAX_PREVIEW_PX}\n"
        f"SCALES_DIR={SCALES_DIR}\n"
        f"ALLOWED_EXTS={sorted(ALLOWED_EXTS)}"
    )
