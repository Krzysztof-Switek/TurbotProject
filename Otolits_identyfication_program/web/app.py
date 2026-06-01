"""FastAPI app — entry point dla uvicorn.

Run lokalnie:
    cd Otolits_identyfication_program
    DATA_ROOT=./test_images uvicorn web.app:app --reload --port 8000

Run w Dockerze: patrz Dockerfile + docker-compose.yml.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from auto_detector import AutoDetector
from web import config
from web.routers import calibration as calibration_router
from web.routers import crop as crop_router
from web.routers import detect as detect_router
from web.routers import fs as fs_router
from web.routers import image as image_router

logger = logging.getLogger("turbot.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ładuje model YOLO do app.state.detector. Shutdown: nic — model
    zwalniany przez GC.
    """
    logger.info("Konfiguracja aplikacji:\n%s", config.describe())

    if not config.DATA_ROOT.is_dir():
        logger.warning(
            "DATA_ROOT=%s nie istnieje lub nie jest katalogiem — endpointy /api/fs/* będą zwracać błędy",
            config.DATA_ROOT,
        )

    detector = AutoDetector(model_path=str(config.MODEL_PATH))
    if detector.model is None:
        logger.warning(
            "Model YOLO nie został wczytany (MODEL_PATH=%s nie istnieje) — /api/detect zwróci 503",
            config.MODEL_PATH,
        )
    else:
        logger.info("Model YOLO załadowany: %s", config.MODEL_PATH)
    app.state.detector = detector

    yield

    # Shutdown — placeholder dla przyszłych zasobów (LRU cache, sesje itp.).
    logger.info("Shutting down")


app = FastAPI(
    title="Turbot — Otolith Annotation Tool",
    description="Webowy frontend do anotacji otolitów (port z desktop).",
    version="0.1.0",
    lifespan=lifespan,
)

# API routery.
app.include_router(fs_router.router)
app.include_router(image_router.router)
app.include_router(detect_router.router)
app.include_router(crop_router.router)
app.include_router(calibration_router.router)

# Statyczny frontend (HTML/JS/CSS). Mountowany na końcu żeby nie przejmować
# ścieżek /api/*. `html=True` — index.html wyświetla się na "/".
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
else:
    logger.warning("Brak katalogu static (%s) — frontend niedostępny", _STATIC_DIR)
