"""API endpoint do automatycznej detekcji otolitów przez YOLO.

Route:
- POST /api/detect  (body: {path})  -> DetectResponse

YOLO model trzymany w `app.state.detector` (ładowany raz przy starcie
w `web/app.py`). Backend ładuje oryginalny obraz, wykonuje detekcję
na pełnej rozdzielczości, przelicza wyniki do przestrzeni preview
(× scale) — frontend dostaje boxy gotowe do narysowania na canvasie.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from web.services import image_service

router = APIRouter(prefix="/api/detect", tags=["detect"])


class DetectRequest(BaseModel):
    path: str = Field(..., description="Ścieżka pliku relatywna do DATA_ROOT")


class DetectedBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "auto"


class DetectResponse(BaseModel):
    boxes: list[DetectedBox]
    scale: float = Field(..., description="preview_dim / original_dim, 0–1")


@router.post("", response_model=DetectResponse)
def detect(req: DetectRequest, request: Request) -> DetectResponse:
    """Wykrywa otolity YOLO na obrazie wskazanym ścieżką relatywną do DATA_ROOT.

    Returns boxy w przestrzeni **preview** (przeliczone przez `scale`) —
    frontend nie martwi się skalą do crop bo robimy odwrotną transformację
    po stronie backendu w `/api/crop`.
    """
    detector = getattr(request.app.state, "detector", None)
    if detector is None or detector.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model YOLO nie został załadowany na serwerze (sprawdź MODEL_PATH)",
        )

    try:
        loaded = image_service.load(req.path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    original = loaded.loader.get_original_image(copy=False)
    if original is None:
        raise HTTPException(status_code=500, detail="Loader nie ma oryginalnego obrazu")

    # Detect na pełnej rozdzielczości (najlepsza jakość modelu).
    original_boxes = detector.detect(original)

    # Przelicz do przestrzeni preview — frontend rysuje na canvasie podglądu.
    scale = loaded.loader.scale
    preview_boxes = [
        DetectedBox(
            x1=int(x1 * scale),
            y1=int(y1 * scale),
            x2=int(x2 * scale),
            y2=int(y2 * scale),
            label="auto",
        )
        for (x1, y1, x2, y2) in original_boxes
    ]

    return DetectResponse(boxes=preview_boxes, scale=scale)
