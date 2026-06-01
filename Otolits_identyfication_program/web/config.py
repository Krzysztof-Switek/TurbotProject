"""Konfiguracja aplikacji webowej — odczyt env vars z domyślnymi.

Wszystkie wartości są niezmienne po starcie procesu (czytane raz). FastAPI app
i routery importują stałe zdefiniowane tu zamiast wołać os.environ bezpośrednio.
"""

import os
from pathlib import Path

# Katalog główny dla nawigacji userskiej. Wszystkie ścieżki w API są relatywne
# do tego katalogu. Path traversal blokowany przez fs_browser.safe_resolve().
# W Dockerze montowany jako volume (np. -v /host/data:/data).
DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", "/data")).resolve()

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
        f"MODEL_PATH={MODEL_PATH} (exists={MODEL_PATH.exists()})\n"
        f"MAX_PREVIEW_PX={MAX_PREVIEW_PX}\n"
        f"ALLOWED_EXTS={sorted(ALLOWED_EXTS)}"
    )
