"""Biblioteka nazwanych skal (presetów μm/px) — zastępuje kalibrację per-katalog.

Każda skala to osobny plik `SCALES_DIR/<slug>.json` + kopia wgranego zdjęcia
wzorca `SCALES_DIR/<uuid>.<ext>`. Skala jest **globalna** (nie związana z katalogiem
zdjęć) i wybierana przez użytkownika z menu. `um_per_px` jest zapisywane w
rozdzielczości ORYGINAŁU — poprawne tylko dla zdjęć wykonanych przy tym samym
powiększeniu co wzorzec (pole `magnification` + nazwa pomagają to utrzymać).

Format JSON:
```json
{
  "name": "microscope_4x",
  "slug": "microscope_4x",
  "um_per_px": 12.5,
  "length_real": 2.0,
  "unit": "cm",
  "magnification": "4x",
  "source_filename": "_scales/ab12cd34.jpg",
  "p1": [100, 200],
  "p2": [300, 200],
  "dist_preview_px": 200.0,
  "created_at": "2026-06-19T15:30:00+00:00"
}
```
"""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from web.config import SCALES_DIR
from web.services import fs_browser

# Przelicznik jednostek wejściowych na mikrometry (kanoniczna jednostka skali).
_UNIT_TO_UM = {"um": 1.0, "μm": 1.0, "mm": 1000.0, "cm": 10000.0}


@dataclass
class ScalePreset:
    name: str
    slug: str
    um_per_px: float
    length_real: float
    unit: str
    magnification: str
    source_filename: str        # ścieżka relatywna do DATA_ROOT (kopia wzorca)
    p1: tuple[float, float]     # punkt 1 w przestrzeni preview
    p2: tuple[float, float]     # punkt 2 w przestrzeni preview
    dist_preview_px: float
    created_at: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "um_per_px": self.um_per_px,
            "length_real": self.length_real,
            "unit": self.unit,
            "magnification": self.magnification,
            "source_filename": self.source_filename,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "dist_preview_px": self.dist_preview_px,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScalePreset":
        return cls(
            name=str(data["name"]),
            slug=str(data["slug"]),
            um_per_px=float(data["um_per_px"]),
            length_real=float(data.get("length_real", 0)),
            unit=str(data.get("unit", "um")),
            magnification=str(data.get("magnification", "")),
            source_filename=str(data.get("source_filename", "")),
            p1=tuple(data.get("p1", [0, 0])),
            p2=tuple(data.get("p2", [0, 0])),
            dist_preview_px=float(data.get("dist_preview_px", 0)),
            created_at=str(data.get("created_at", "")),
        )


def _to_um(value: float, unit: str) -> float:
    """Przelicza długość rzeczywistą na mikrometry. Rzuca ValueError dla nieznanej jednostki."""
    if unit not in _UNIT_TO_UM:
        raise ValueError(f"Nieznana jednostka '{unit}' (dozwolone: cm, mm, um)")
    return float(value) * _UNIT_TO_UM[unit]


def slugify(name: str) -> str:
    """Nazwa → bezpieczny slug (a-z0-9_). Rzuca ValueError gdy wynik pusty."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"Nazwa '{name}' nie zawiera dozwolonych znaków")
    return slug


def _scales_dir() -> Path:
    """Zwraca SCALES_DIR, tworząc go gdy nie istnieje."""
    SCALES_DIR.mkdir(parents=True, exist_ok=True)
    return SCALES_DIR


def _preset_path(slug: str) -> Path:
    return _scales_dir() / f"{slug}.json"


def list_scales() -> list[ScalePreset]:
    """Wszystkie presety z SCALES_DIR, posortowane po nazwie. Zepsute pliki pomijane."""
    presets: list[ScalePreset] = []
    for path in sorted(_scales_dir().glob("*.json")):
        try:
            presets.append(ScalePreset.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError) as e:
            print(f"scales.list_scales: pomijam zepsuty {path}: {e}")
    presets.sort(key=lambda p: p.name.lower())
    return presets


def get_scale(slug: str) -> Optional[ScalePreset]:
    """Wczytuje pojedynczy preset po slugu. None gdy brak/zepsuty."""
    path = _preset_path(slug)
    if not path.is_file():
        return None
    try:
        return ScalePreset.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError) as e:
        print(f"scales.get_scale: błąd parsowania {path}: {e}")
        return None


def save_uploaded_photo(orig_filename: str, data: bytes) -> str:
    """Zapisuje wgrane zdjęcie wzorca pod unikalną nazwą w SCALES_DIR.

    Zwraca ścieżkę relatywną do DATA_ROOT (do użycia przez /api/image/preview).
    Rzuca ValueError przy niedozwolonym rozszerzeniu.
    """
    ext = Path(orig_filename).suffix.lower()
    from web.config import ALLOWED_EXTS
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Niedozwolone rozszerzenie '{ext}' (dozwolone: {sorted(ALLOWED_EXTS)})")
    target = _scales_dir() / f"{uuid.uuid4().hex}{ext}"
    target.write_bytes(data)
    return fs_browser.to_rel(target)


def _delete_photo(source_filename: str) -> None:
    """Usuwa kopię zdjęcia wzorca, jeśli leży bezpiecznie w SCALES_DIR."""
    if not source_filename:
        return
    try:
        abs_path = fs_browser.safe_resolve(source_filename)
    except (PermissionError, ValueError):
        return
    if abs_path.parent.resolve() == SCALES_DIR and abs_path.is_file():
        try:
            abs_path.unlink()
        except OSError as e:
            print(f"scales._delete_photo: nie udało się usunąć {abs_path}: {e}")


def save_scale(
    name: str,
    um_per_px: float,
    length_real: float,
    unit: str,
    magnification: str,
    source_filename: str,
    p1: tuple[float, float],
    p2: tuple[float, float],
    dist_preview_px: float,
) -> ScalePreset:
    """Zapisuje preset (nadpisuje gdy slug już istnieje = edycja). Walidacja w środku."""
    if um_per_px <= 0:
        raise ValueError(f"um_per_px musi być > 0 (jest {um_per_px})")
    slug = slugify(name)

    # Nadpisanie: usuń stare zdjęcie jeśli zmieniono wzorzec (uniknij sierot).
    existing = get_scale(slug)
    if existing is not None and existing.source_filename != source_filename:
        _delete_photo(existing.source_filename)

    preset = ScalePreset(
        name=name.strip(),
        slug=slug,
        um_per_px=float(um_per_px),
        length_real=float(length_real),
        unit=str(unit),
        magnification=str(magnification),
        source_filename=str(source_filename),
        p1=(float(p1[0]), float(p1[1])),
        p2=(float(p2[0]), float(p2[1])),
        dist_preview_px=float(dist_preview_px),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _preset_path(slug).write_text(json.dumps(preset.to_dict(), indent=2), encoding="utf-8")
    return preset


def delete_scale(slug: str) -> bool:
    """Usuwa preset + jego kopię zdjęcia. Zwraca True jeśli plik JSON istniał."""
    path = _preset_path(slug)
    if not path.is_file():
        return False
    existing = get_scale(slug)
    if existing is not None:
        _delete_photo(existing.source_filename)
    try:
        path.unlink()
    except OSError as e:
        print(f"scales.delete_scale: nie udało się usunąć {path}: {e}")
        return False
    return True
