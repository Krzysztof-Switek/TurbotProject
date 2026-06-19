"""Bezpieczna nawigacja po katalogach w obrębie dozwolonych rootów.

Model ścieżek (string w API): `<RootName>/pod/katalog`, gdzie `<RootName>` to
nazwa z allow-listy `config.ALLOWED_ROOTS`. Pusty string `""` = **wirtualny top**
listujący dostępne roots. Każdy root jest osobną „wyspą": nawigacja jest confined
pod ten root (`Path.is_relative_to`), a absolutne ścieżki serwera nie wyciekają do
klienta (response operuje na `<RootName>/...`).

Dodatkowo istnieje **zarezerwowany** root `_scales` → `config.SCALES_DIR`
(biblioteka skal). Jest rozwiązywalny (preview wgranych wzorców), ale **nie**
pokazywany na wirtualnym topie.
"""

from dataclasses import dataclass
from pathlib import Path

from web.config import ALLOWED_EXTS, ALLOWED_ROOTS, SCALES_DIR

# Zarezerwowana nazwa roota dla biblioteki skal (rozwiązywalna, ukryta w listingu).
SCALES_ROOT_NAME = "_scales"


def _resolvable_roots() -> dict[str, Path]:
    """Wszystkie roots dające się rozwiązać: nawigacyjne + zarezerwowany `_scales`."""
    roots = dict(ALLOWED_ROOTS)
    roots[SCALES_ROOT_NAME] = SCALES_DIR
    return roots


def _split_root(path: str) -> tuple[str, str]:
    """'Root/sub/dir' → ('Root', 'sub/dir'); 'Root' → ('Root', ''); '' → ('', '')."""
    p = (path or "").strip("/")
    if not p:
        return "", ""
    first, _, rest = p.partition("/")
    return first, rest


def safe_resolve(rel_path: str) -> Path:
    """Resolve ścieżki `<Root>/...` do absolutnej z guardem przeciw path traversal.

    Rzuca `PermissionError` gdy: brak roota (pusta ścieżka / sam wirtualny top),
    nieznany root, albo wejście próbuje wyjść poza root.
    """
    root_name, rest = _split_root(rel_path)
    if not root_name:
        raise PermissionError("Brak roota w ścieżce — wybierz katalog z listy")
    base = _resolvable_roots().get(root_name)
    if base is None:
        raise PermissionError(f"Nieznany root '{root_name}'")
    target = (base / rest).resolve()
    if target != base and not target.is_relative_to(base):
        raise PermissionError(f"Path '{rel_path}' wychodzi poza root '{root_name}'")
    return target


def to_rel(abs_path: Path) -> str:
    """Konwersja absolutnej ścieżki na `<RootName>/...` (relatywnie do roota).

    `_scales` sprawdzany pierwszy, żeby pliki biblioteki skal mapowały się na krótką,
    stabilną formę `_scales/<plik>` niezależnie od tego czy SCALES_DIR leży też pod
    którymś rootem nawigacyjnym. Rzuca `ValueError` gdy poza wszystkimi rootami.
    """
    abs_path = abs_path.resolve()
    candidates = [(SCALES_ROOT_NAME, SCALES_DIR), *ALLOWED_ROOTS.items()]
    for name, base in candidates:
        if abs_path == base:
            return name
        if abs_path.is_relative_to(base):
            return f"{name}/{abs_path.relative_to(base).as_posix()}"
    raise ValueError(f"Ścieżka {abs_path} nie jest pod żadnym dozwolonym rootem")


@dataclass
class DirEntry:
    name: str


@dataclass
class FileEntry:
    name: str
    size: int


@dataclass
class ListResult:
    path: str          # `<Root>/...` lub "" dla wirtualnego topu
    parent: str | None # `<Root>/...` / "" (top); None gdy jesteśmy na topie
    dirs: list[DirEntry]
    images: list[FileEntry]


def list_dir(rel_path: str) -> ListResult:
    """Lista podkatalogów + plików obrazowych.

    - `rel_path == ""` → **wirtualny top**: zwraca roots nawigacyjne jako `dirs`
      (bez `_scales`), `parent=None`, brak obrazów.
    - inaczej → zawartość katalogu pod wskazanym rootem.

    Filtruje pliki po `ALLOWED_EXTS` (case-insensitive), sortuje alfabetycznie,
    ukrywa katalog biblioteki skal. Rzuca `PermissionError` (traversal / brak roota),
    `FileNotFoundError`, `NotADirectoryError`.
    """
    root_name, rest = _split_root(rel_path)

    if not root_name:
        # Wirtualny top — lista rootów nawigacyjnych jako "katalogi".
        return ListResult(
            path="",
            parent=None,
            dirs=[DirEntry(name=n) for n in ALLOWED_ROOTS],
            images=[],
        )

    target = safe_resolve(rel_path)

    if not target.exists():
        raise FileNotFoundError(f"Katalog nie istnieje: '{rel_path}'")
    if not target.is_dir():
        raise NotADirectoryError(f"Ścieżka nie jest katalogiem: '{rel_path}'")

    dirs: list[DirEntry] = []
    images: list[FileEntry] = []
    for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir():
            # Ukryj wewnętrzny katalog biblioteki skal (gdyby leżał pod rootem nav).
            if entry.resolve() == SCALES_DIR:
                continue
            dirs.append(DirEntry(name=entry.name))
        elif entry.is_file() and entry.suffix.lower() in ALLOWED_EXTS:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            images.append(FileEntry(name=entry.name, size=size))

    # Parent: na szczycie roota (rest=="") wracamy do wirtualnego topu (""),
    # głębiej — do katalogu nadrzędnego.
    parent = "" if rest == "" else to_rel(target.parent)

    return ListResult(
        path=to_rel(target),
        parent=parent,
        dirs=dirs,
        images=images,
    )


def make_dir(rel_path: str) -> Path:
    """Tworzy katalog (z parentami) pod wskazanym rootem.

    Idempotentne (`exist_ok=True`). Rzuca `PermissionError` (traversal / brak roota)
    i `FileExistsError` gdy ścieżka wskazuje na istniejący plik (kolizja typu).
    """
    target = safe_resolve(rel_path)
    if target.exists() and not target.is_dir():
        raise FileExistsError(
            f"Ścieżka '{rel_path}' istnieje już jako plik, nie katalog"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target
