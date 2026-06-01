"""Wrapper na ImageLoader dla web — ładowanie pojedynczego obrazu po ścieżce
relatywnej do DATA_ROOT, encode preview do PNG, ekspozycja oryginalnej skali.

Używany przez `web/routers/image.py`, `web/routers/detect.py`, `web/routers/crop.py`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from image_loader import ImageLoader
from web.config import MAX_PREVIEW_PX
from web.services import fs_browser


@dataclass
class LoadedImage:
    """Wynik load_for_preview / load_for_detect.

    - `loader`: ImageLoader skierowany na pojedynczy plik. `loader.scale`,
      `loader.original_size`, `loader.image` (preview), `loader.original_image`
      są dostępne po wywołaniu load_image() w środku.
    - `abs_path`: absolutna ścieżka pliku (po safe_resolve).
    - `rel_path`: ścieżka relatywna do DATA_ROOT (do response'ów).
    """
    loader: ImageLoader
    abs_path: Path
    rel_path: str


def load(rel_path: str) -> LoadedImage:
    """Ładuje obraz wskazany ścieżką relatywną do DATA_ROOT.

    1. `safe_resolve(rel_path)` — guard path traversal.
    2. `ImageLoader.from_path(abs_path, max_preview_px=MAX_PREVIEW_PX)` —
       konstruktor dla pojedynczego pliku.
    3. `loader.load_image()` — wczytuje oryginał + tworzy preview.

    Rzuca `PermissionError` (path traversal), `FileNotFoundError` (brak pliku),
    `ValueError` (nieprawidłowy obraz / przekroczenie limitów rozmiaru).
    """
    abs_path = fs_browser.safe_resolve(rel_path)
    if not abs_path.is_file():
        raise FileNotFoundError(f"Plik nie istnieje: '{rel_path}'")

    loader = ImageLoader.from_path(str(abs_path), max_preview_px=MAX_PREVIEW_PX)
    loader.load_image()
    return LoadedImage(
        loader=loader,
        abs_path=abs_path,
        rel_path=fs_browser.to_rel(abs_path),
    )


def encode_png(image: np.ndarray) -> bytes:
    """Koduje obraz BGR (numpy ndarray) do bajtów PNG.

    Używane przez endpoint /api/image/preview do zwrotu pliku jako binary
    response. Rzuca `RuntimeError` jeśli cv2 nie zakodowało.
    """
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Nie udało się zakodować obrazu do PNG")
    return buf.tobytes()


def get_preview_meta(loaded: LoadedImage) -> dict:
    """Metadane preview do nagłówków HTTP (X-Original-Width itp.).

    Frontend używa tych wartości żeby przeliczać współrzędne preview → oryginał
    przy crop. Wartości dokładnie odpowiadają temu co przelicza
    `ImageLoader.scale_coords_to_original`.
    """
    w, h = loaded.loader.original_size
    return {
        "X-Original-Width": str(w),
        "X-Original-Height": str(h),
        "X-Scale": f"{loaded.loader.scale:.6f}",
    }
