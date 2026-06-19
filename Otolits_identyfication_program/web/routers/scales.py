"""API biblioteki nazwanych skal (presetów μm/px).

Routes:
- GET    /api/scales            -> list[ScaleResponse]
- POST   /api/scales/upload     -> {"path": "<rel>"}   (multipart, wgranie wzorca)
- POST   /api/scales            -> ScaleResponse        (utworzenie/edycja presetu)
- DELETE /api/scales/{slug}     -> {"deleted": bool}

Model globalny (nie per-katalog): preset wybierany przez użytkownika z menu w UI.
Backend liczy `um_per_px` z surowych danych (p1/p2/length_real/unit/scale) jako
single source of truth — eliminuje rozjazdy preview↔original.
"""

import math

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from web.services import scales as scales_service

router = APIRouter(prefix="/api/scales", tags=["scales"])


# ---------- schemas ----------

class ScaleResponse(BaseModel):
    name: str
    slug: str
    um_per_px: float
    length_real: float
    unit: str
    magnification: str
    source_filename: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    dist_preview_px: float
    created_at: str


class UploadResponse(BaseModel):
    path: str = Field(..., description="Ścieżka wgranego wzorca, relatywna do DATA_ROOT.")


class SaveScaleRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Nazwa skali wybrana przez użytkownika.")
    magnification: str = Field("", description="Powiększenie mikroskopu, np. '4x' — informacyjne.")
    p1: tuple[float, float] = Field(..., description="Punkt 1 wzorca (przestrzeń preview).")
    p2: tuple[float, float] = Field(..., description="Punkt 2 wzorca (przestrzeń preview).")
    length_real: float = Field(..., gt=0, description="Rzeczywista długość |p1-p2|.")
    unit: str = Field(..., description="Jednostka length_real: cm | mm | um.")
    scale: float = Field(..., gt=0, description="Współczynnik preview→original (X-Scale wzorca).")
    source_filename: str = Field(..., description="Ścieżka wgranego wzorca (z /upload).")


def _to_response(p: scales_service.ScalePreset) -> ScaleResponse:
    return ScaleResponse(**p.to_dict())


@router.get("", response_model=list[ScaleResponse])
def list_scales() -> list[ScaleResponse]:
    """Zwraca wszystkie zapisane skale (posortowane po nazwie)."""
    return [_to_response(p) for p in scales_service.list_scales()]


@router.post("/upload", response_model=UploadResponse)
async def upload_photo(file: UploadFile = File(...)) -> UploadResponse:
    """Wgrywa zdjęcie wzorca; zapisuje w SCALES_DIR; zwraca jego ścieżkę relatywną."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Pusty plik")
    try:
        rel = scales_service.save_uploaded_photo(file.filename or "upload.png", data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return UploadResponse(path=rel)


@router.post("", response_model=ScaleResponse)
def create_scale(req: SaveScaleRequest) -> ScaleResponse:
    """Tworzy/edytuje preset skali. Liczy μm/px do oryginału z surowych danych."""
    dist_preview_px = math.hypot(req.p2[0] - req.p1[0], req.p2[1] - req.p1[1])
    if dist_preview_px < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Punkty wzorca są za blisko (dystans {dist_preview_px:.2f} px). Wybierz dłuższy odcinek.",
        )
    try:
        length_um = scales_service._to_um(req.length_real, req.unit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # μm/px DO ORYGINAŁU = length_um / (dist_preview_px / scale) = length_um * scale / dist_preview_px.
    um_per_px_original = length_um * req.scale / dist_preview_px

    try:
        preset = scales_service.save_scale(
            name=req.name,
            um_per_px=um_per_px_original,
            length_real=req.length_real,
            unit=req.unit,
            magnification=req.magnification,
            source_filename=req.source_filename,
            p1=req.p1,
            p2=req.p2,
            dist_preview_px=dist_preview_px,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _to_response(preset)


@router.delete("/{slug}")
def delete_scale(slug: str) -> dict:
    """Usuwa preset (+ kopię zdjęcia wzorca)."""
    deleted = scales_service.delete_scale(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Skala '{slug}' nie istnieje")
    return {"deleted": True}
