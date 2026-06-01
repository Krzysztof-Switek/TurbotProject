"""Kalibracja skali (μm/px) per katalog — sidecar `calibration.json`.

Wszystkie zdjęcia w katalogu mają jedną wartość skali (założenie userskie:
identyczne powiększenie mikroskopu na całej sesji). Plik `calibration.json`
leży **w katalogu zdjęć** obok plików `.jpg`, czyli np.:

    DATA_ROOT/test_images/calibration.json
    DATA_ROOT/test_images/FLE_NPZDR_2025_1.jpg
    DATA_ROOT/test_images/FLE_NPZDR_2025_2.jpg
    ...

Format JSON:
```json
{
  "um_per_px": 12.5,
  "magnification": "4x",
  "reference_image": "FLE_NPZDR_2025_1.jpg",
  "p1": [100, 200],
  "p2": [300, 200],
  "length_um": 2500.0,
  "created_at": "2026-06-01T15:30:00Z"
}
```
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from web.services import fs_browser

CALIBRATION_FILENAME = "calibration.json"


@dataclass
class Calibration:
    """Pełne dane kalibracji + metadane do wyświetlenia w UI / logu."""
    um_per_px: float
    magnification: str
    reference_image: str       # nazwa pliku (bez ścieżki) z którego user kliknął wzorzec
    p1: tuple[float, float]    # współrzędne preview pierwszego klika
    p2: tuple[float, float]    # współrzędne preview drugiego klika
    length_um: float           # długość rzeczywista odcinka p1-p2 w μm
    created_at: str            # ISO timestamp UTC

    def to_dict(self) -> dict:
        return {
            "um_per_px": self.um_per_px,
            "magnification": self.magnification,
            "reference_image": self.reference_image,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "length_um": self.length_um,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(
            um_per_px=float(data["um_per_px"]),
            magnification=str(data.get("magnification", "")),
            reference_image=str(data.get("reference_image", "")),
            p1=tuple(data.get("p1", [0, 0])),
            p2=tuple(data.get("p2", [0, 0])),
            length_um=float(data.get("length_um", 0)),
            created_at=str(data.get("created_at", "")),
        )


def _calibration_path(rel_dir: str) -> Path:
    """Lokalizacja sidecar JSON. Rzuca PermissionError przy path traversal."""
    target_dir = fs_browser.safe_resolve(rel_dir)
    return target_dir / CALIBRATION_FILENAME


def load(rel_dir: str) -> Optional[Calibration]:
    """Wczytuje kalibrację z `<rel_dir>/calibration.json`. None jeśli brak."""
    path = _calibration_path(rel_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Calibration.from_dict(data)
    except (OSError, ValueError, KeyError) as e:
        # Zepsuty plik nie powinien blokować pracy — traktujemy jak brak.
        print(f"calibration.load: błąd parsowania {path}: {e}")
        return None


def save(
    rel_dir: str,
    um_per_px: float,
    magnification: str,
    reference_image: str,
    p1: tuple[float, float],
    p2: tuple[float, float],
    length_um: float,
) -> Calibration:
    """Zapisuje kalibrację do `<rel_dir>/calibration.json`. Nadpisuje istniejącą."""
    if um_per_px <= 0:
        raise ValueError(f"um_per_px musi być > 0 (jest {um_per_px})")
    if length_um <= 0:
        raise ValueError(f"length_um musi być > 0 (jest {length_um})")

    calib = Calibration(
        um_per_px=float(um_per_px),
        magnification=str(magnification),
        reference_image=str(reference_image),
        p1=(float(p1[0]), float(p1[1])),
        p2=(float(p2[0]), float(p2[1])),
        length_um=float(length_um),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    path = _calibration_path(rel_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib.to_dict(), indent=2), encoding="utf-8")
    return calib
