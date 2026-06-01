"""API endpoints do nawigacji po katalogach + tworzenia nowych.

Routes:
- GET  /api/fs/list?path=<rel>  -> ListResponse
- POST /api/fs/mkdir            -> MkdirResponse  (body: {path})

Wszystkie ścieżki w request/response są **relatywne do DATA_ROOT**.
Wewnętrznie wszystko przechodzi przez `fs_browser.safe_resolve()`.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from web.services import fs_browser

router = APIRouter(prefix="/api/fs", tags=["fs"])


class DirEntryOut(BaseModel):
    name: str


class FileEntryOut(BaseModel):
    name: str
    size: int


class ListResponse(BaseModel):
    path: str
    parent: str | None
    dirs: list[DirEntryOut]
    images: list[FileEntryOut]


class MkdirRequest(BaseModel):
    path: str = Field(..., min_length=1, description="Ścieżka katalogu relatywna do DATA_ROOT")


class MkdirResponse(BaseModel):
    path: str
    created: bool


@router.get("/list", response_model=ListResponse)
def list_directory(path: str = Query("", description="Relatywna do DATA_ROOT; pusty = root")) -> ListResponse:
    """Lista podkatalogów + plików obrazowych w danym katalogu."""
    try:
        result = fs_browser.list_dir(path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ListResponse(
        path=result.path,
        parent=result.parent,
        dirs=[DirEntryOut(name=d.name) for d in result.dirs],
        images=[FileEntryOut(name=f.name, size=f.size) for f in result.images],
    )


@router.post("/mkdir", response_model=MkdirResponse)
def make_directory(req: MkdirRequest) -> MkdirResponse:
    """Tworzy nowy katalog pod DATA_ROOT (idempotentne: `exist_ok=True`)."""
    try:
        already_existed = fs_browser.safe_resolve(req.path).exists()
        target = fs_browser.make_dir(req.path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return MkdirResponse(
        path=fs_browser.to_rel(target),
        created=not already_existed,
    )
