"""API endpoint do cropowania boxów + pseudo-labellingu.

Route:
- POST /api/crop  -> CropResponse

Backend:
1. Walidacja override values (None/'A'/'B').
2. safe_resolve(image_path, output_dir).
3. Load oryginalny obraz przez image_service (ImageLoader.from_path).
4. Rekonstrukcja BoundingBox/RowLine/Row z payloadu (z compartment_override).
5. Walidacja przez compute_row_labels — błąd → HTTP 400.
6. mkdir output_dir (idempotentne).
7. ImageCropper.crop_and_save — pliki PNG do output_dir.
8. Jeśli save_annotations → _save_compartment_annotations (plik .txt obok zdjęcia).
9. Response z listą plików, ścieżką .txt (jeśli powstał), licznikami A/B.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bounding_box import BoundingBox
from image_cropper import ImageCropper, compute_row_labels
from row_detector import RowDetector, RowLine
from web.services import fs_browser, image_service

router = APIRouter(prefix="/api/crop", tags=["crop"])


# ---------- request schemas ----------

class BoxIn(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class LineIn(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]


class RowIn(BaseModel):
    line: LineIn
    boxes: list[BoxIn]
    compartment_override: Optional[str] = Field(
        default=None,
        description="'A' / 'B' = manualny override; null = przypisanie przez geometrię.",
    )


class CropRequest(BaseModel):
    image_path: str = Field(..., description="Ścieżka zdjęcia źródłowego relatywna do DATA_ROOT.")
    output_dir: str = Field(..., description="Katalog wyjściowy dla PNG, relatywny do DATA_ROOT.")
    rows: list[RowIn]
    save_annotations: bool = Field(
        default=True,
        description="Czy zapisać adnotację YOLO `.txt` obok zdjęcia (pseudo-labelling).",
    )
    um_per_px: Optional[float] = Field(
        default=None,
        gt=0,
        description="Skala oryginalnego obrazu w μm/px (z kalibracji per-katalog). "
                    "Gdy podane, na każdym wycinku rysowany jest pasek skali.",
    )


# ---------- response schemas ----------

class SavedFile(BaseModel):
    filename: str
    row_index: int
    box_index: int


class CropResponse(BaseModel):
    saved: list[SavedFile]
    annotations_path: Optional[str] = Field(
        default=None,
        description="Ścieżka .txt relatywna do DATA_ROOT, gdy zapis pseudo-label się powiódł.",
    )
    compartments: dict[str, int] = Field(
        default_factory=dict,
        description="Liczniki wierszy per wycinek: {'A': N, 'B': M}.",
    )


@router.post("", response_model=CropResponse)
def crop(req: CropRequest) -> CropResponse:
    """Cropuje boxy z wierszy do output_dir; opcjonalnie zapisuje pseudo-label."""

    # 1. Walidacja override
    for row_in in req.rows:
        if row_in.compartment_override not in (None, "A", "B"):
            raise HTTPException(
                status_code=400,
                detail=f"compartment_override musi być 'A', 'B' lub null (jest '{row_in.compartment_override}')",
            )

    # 2. Load source image (path traversal + I/O w środku image_service.load)
    try:
        loaded = image_service.load(req.image_path)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    original = loaded.loader.get_original_image(copy=False)
    if original is None:
        raise HTTPException(status_code=500, detail="Loader nie ma oryginalnego obrazu")

    # 3. Resolve + mkdir output_dir
    try:
        output_abs = fs_browser.make_dir(req.output_dir)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # 4. Rekonstrukcja obiektów Python z payloadu
    rows = []
    for idx, row_in in enumerate(req.rows, start=1):
        if not row_in.boxes:
            continue
        boxes = [BoundingBox(b.x1, b.y1, b.x2, b.y2, label="user") for b in row_in.boxes]
        line = RowLine(
            p1=tuple(row_in.line.p1),
            p2=tuple(row_in.line.p2),
            id=str(uuid.uuid4()),
        )
        rows.append(
            RowDetector.Row(
                id=idx,
                line=line,
                boxes=boxes,
                compartment_override=row_in.compartment_override,
            )
        )

    # 5. Walidacja semantyczna przez compute_row_labels (>6 globalnie / >3 per wycinek)
    labels, err = compute_row_labels(rows)
    if err is not None:
        raise HTTPException(status_code=400, detail=err)

    # 6. Crop + save (z opcjonalnym paskiem skali jeśli um_per_px podane)
    cropper = ImageCropper(output_dir=str(output_abs), image_loader=loaded.loader)
    crop_results = cropper.crop_and_save(original, rows, [], um_per_px=req.um_per_px)

    # 7. Pseudo-labelling
    annotations_rel: Optional[str] = None
    if req.save_annotations:
        cropper._save_compartment_annotations(original, rows)
        expected_txt = loaded.abs_path.with_suffix(".txt")
        if expected_txt.is_file():
            annotations_rel = fs_browser.to_rel(expected_txt)

    # 8. Liczniki A/B
    counts = {"A": 0, "B": 0}
    for label_str in labels.values():
        counts[label_str[0]] += 1

    return CropResponse(
        saved=[
            SavedFile(filename=cr.filename, row_index=cr.row_index, box_index=cr.box_index)
            for cr in crop_results
        ],
        annotations_path=annotations_rel,
        compartments=counts,
    )
