"""Bezpieczna nawigacja po katalogach w obrębie DATA_ROOT.

Wszystkie ścieżki przychodzące z API są **relatywne** do DATA_ROOT.
Funkcje tu zdefiniowane (`safe_resolve`, `list_dir`, `make_dir`):
- przepuszczają wejście przez `Path.is_relative_to()` żeby zablokować
  path traversal (`../../etc`),
- nie ujawniają absolutnej ścieżki serwera do klienta (response operuje
  na ścieżkach relatywnych).
"""

from dataclasses import dataclass
from pathlib import Path

from web.config import ALLOWED_EXTS, DATA_ROOT


def safe_resolve(rel_path: str) -> Path:
    """Resolve relatywnej ścieżki do absolutnej z guardem przeciw path traversal.

    Zwraca absolutną ścieżkę pod DATA_ROOT. Rzuca `PermissionError` gdy wejście
    próbuje wyjść poza DATA_ROOT (np. `../foo`, `/etc`, symlink poza root).
    """
    target = (DATA_ROOT / (rel_path or "")).resolve()
    if target != DATA_ROOT and not target.is_relative_to(DATA_ROOT):
        raise PermissionError(f"Path '{rel_path}' wychodzi poza DATA_ROOT")
    return target


def to_rel(abs_path: Path) -> str:
    """Konwersja absolutnej ścieżki na string relatywny do DATA_ROOT.

    Używane przy budowaniu response'ów żeby nie ujawniać absolutnej ścieżki
    serwera klientowi (oprócz path-traversal hardening to też operational
    security — log/screenshot nie zdradza struktury hosta).
    """
    abs_path = abs_path.resolve()
    if abs_path == DATA_ROOT:
        return ""
    return abs_path.relative_to(DATA_ROOT).as_posix()


@dataclass
class DirEntry:
    name: str


@dataclass
class FileEntry:
    name: str
    size: int


@dataclass
class ListResult:
    path: str          # relatywna do DATA_ROOT
    parent: str | None # relatywna; None gdy jesteśmy w DATA_ROOT
    dirs: list[DirEntry]
    images: list[FileEntry]


def list_dir(rel_path: str) -> ListResult:
    """Lista podkatalogów + plików obrazowych w `rel_path` (relatywnym do DATA_ROOT).

    Filtruje pliki po `ALLOWED_EXTS` (case-insensitive). Sortuje alfabetycznie.
    Rzuca `PermissionError` przy path traversal, `FileNotFoundError` gdy katalog
    nie istnieje, `NotADirectoryError` gdy ścieżka wskazuje na plik.
    """
    target = safe_resolve(rel_path)

    if not target.exists():
        raise FileNotFoundError(f"Katalog nie istnieje: '{rel_path}'")
    if not target.is_dir():
        raise NotADirectoryError(f"Ścieżka nie jest katalogiem: '{rel_path}'")

    dirs: list[DirEntry] = []
    images: list[FileEntry] = []
    for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir():
            dirs.append(DirEntry(name=entry.name))
        elif entry.is_file() and entry.suffix.lower() in ALLOWED_EXTS:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            images.append(FileEntry(name=entry.name, size=size))

    parent: str | None
    if target == DATA_ROOT:
        parent = None
    else:
        parent = to_rel(target.parent)

    return ListResult(
        path=to_rel(target),
        parent=parent,
        dirs=dirs,
        images=images,
    )


def make_dir(rel_path: str) -> Path:
    """Tworzy katalog (z parentami) pod DATA_ROOT.

    Idempotentne: jeśli katalog już istnieje, nie rzuca. Rzuca `PermissionError`
    przy path traversal i `FileExistsError` gdy ścieżka wskazuje na istniejący
    plik (kolizja typu).
    """
    target = safe_resolve(rel_path)
    if target.exists() and not target.is_dir():
        raise FileExistsError(
            f"Ścieżka '{rel_path}' istnieje już jako plik, nie katalog"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target
