"""API endpointy kalibracji skali (μm/px) per katalog.

Routes:
- GET  /api/calibration?dir=<rel>  -> CalibrationResponse | 404
- POST /api/calibration            -> CalibrationResponse

Per-katalog: jedna kalibracja używana przez wszystkie zdjęcia w `<rel_dir>/`.
Frontend pobiera kalibrację po wczytaniu obrazu, decyzja o blokadzie crop
bez kalibracji jest po stronie UI.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from web.services import calibration as calibration_service

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


class CalibrationResponse(BaseModel):
    um_per_px: float
    magnification: str
    reference_image: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    length_um: float
    created_at: str


class SaveRequest(BaseModel):
    dir: str = Field(..., description="Katalog (relatywny do DATA_ROOT) gdzie zapisać sidecar")
    magnification: str = Field("", description="Powiększenie mikroskopu, np. '4x' — informacyjne")
    reference_image: str = Field(..., description="Nazwa zdjęcia z którego user wykonał kalibrację")
    p1: tuple[float, float] = Field(..., description="Punkt 1 wzorca (w przestrzeni preview)")
    p2: tuple[float, float] = Field(..., description="Punkt 2 wzorca (w przestrzeni preview)")
    length_um: float = Field(..., gt=0, description="Rzeczywista długość |p1-p2| w mikrometrach")
    scale: float = Field(..., gt=0, description="Skala preview→original z X-Scale (do przeliczenia μm/px na oryginał)")


def _calibration_to_response(c: calibration_service.Calibration) -> CalibrationResponse:
    return CalibrationResponse(
        um_per_px=c.um_per_px,
        magnification=c.magnification,
        reference_image=c.reference_image,
        p1=c.p1,
        p2=c.p2,
        length_um=c.length_um,
        created_at=c.created_at,
    )


@router.get("", response_model=Optional[CalibrationResponse])
def get_calibration(dir: str = Query("", description="Katalog relatywny do DATA_ROOT")):
    """Zwraca kalibrację dla katalogu lub null (200 z body=null) jeśli brak.

    Brak kalibracji nie jest błędem — frontend po prostu nie ma jeszcze
    wykonanej kalibracji, normalny stan przy pierwszym wejściu do katalogu.
    """
    try:
        calib = calibration_service.load(dir)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if calib is None:
        return None
    return _calibration_to_response(calib)


@router.post("", response_model=CalibrationResponse)
def post_calibration(req: SaveRequest):
    """Zapisuje kalibrację dla katalogu.

    Frontend liczy `um_per_px` po stronie klienta: `length_um / hypot(p2-p1) / scale`,
    ale backend dodatkowo przeliczy to samo z surowych danych (p1/p2/length_um/scale)
    jako single source of truth — eliminuje rozjazdy.
    """
    import math
    # Przestrzeń preview: dist_preview_px = hypot(p2-p1).
    dx = req.p2[0] - req.p1[0]
    dy = req.p2[1] - req.p1[1]
    dist_preview_px = math.hypot(dx, dy)
    if dist_preview_px < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Punkty wzorca są za blisko (dystans {dist_preview_px:.2f} px). Wybierz dłuższy odcinek.",
        )
    # Skala preview→original: jeśli preview ma scale=0.05, to 1 px preview = 20 px oryginał.
    # μm/px DO ORYGINAŁU = length_um / (dist_preview_px / scale) = length_um * scale / dist_preview_px.
    um_per_px_original = req.length_um * req.scale / dist_preview_px

    try:
        calib = calibration_service.save(
            rel_dir=req.dir,
            um_per_px=um_per_px_original,
            magnification=req.magnification,
            reference_image=req.reference_image,
            p1=req.p1,
            p2=req.p2,
            length_um=req.length_um,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _calibration_to_response(calib)
