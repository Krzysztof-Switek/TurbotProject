"""API endpoint do pobrania obrazu (preview, przeskalowanego do MAX_PREVIEW_PX).

Route:
- GET /api/image/preview?path=<rel>  -> PNG response + nagłówki X-Original-* i X-Scale
"""

from fastapi import APIRouter, HTTPException, Query, Response

from web.services import image_service

router = APIRouter(prefix="/api/image", tags=["image"])


@router.get("/preview", responses={200: {"content": {"image/png": {}}}})
def preview(path: str = Query(..., description="Ścieżka pliku relatywna do DATA_ROOT")) -> Response:
    """Zwraca obraz preview (PNG, BGR przez OpenCV) przeskalowany do `MAX_PREVIEW_PX`.

    Nagłówki:
    - `X-Original-Width`  — szerokość oryginalnego obrazu (px).
    - `X-Original-Height` — wysokość oryginalnego obrazu (px).
    - `X-Scale`           — `preview_dim / original_dim` (0–1).

    Frontend używa tych wartości żeby przeliczać współrzędne preview <-> oryginał
    przy crop. Cache-Control: brak (każdy fetch ładuje plik z dysku — prosty
    model, można dodać LRU cache w przyszłości jeśli będzie potrzeba).
    """
    try:
        loaded = image_service.load(path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if loaded.loader.image is None:
        raise HTTPException(status_code=500, detail="Loader nie zwrócił obrazu preview")

    png_bytes = image_service.encode_png(loaded.loader.image)
    headers = image_service.get_preview_meta(loaded)
    return Response(content=png_bytes, media_type="image/png", headers=headers)
