# 03 — Dependencies

## Internal import graph

Each module and the project modules it imports.

| Module | Imports (internal) |
|--------|--------------------|
| `Otolits_identyfication_program.YOLO.datasets.split_dataset` | — |
| `Otolits_identyfication_program.YOLO.predict` | — |
| `Otolits_identyfication_program.YOLO.predict_file` | — |
| `Otolits_identyfication_program.YOLO.train` | — |
| `Otolits_identyfication_program.YOLO.yolo_trainer` | `Otolits_identyfication_program.bounding_box_manager` |
| `Otolits_identyfication_program.__init__` | — |
| `Otolits_identyfication_program.auto_detector` | — |
| `Otolits_identyfication_program.bounding_box` | — |
| `Otolits_identyfication_program.bounding_box_manager` | `Otolits_identyfication_program.bounding_box` |
| `Otolits_identyfication_program.image_cropper` | — |
| `Otolits_identyfication_program.image_loader` | — |
| `Otolits_identyfication_program.image_window` | `Otolits_identyfication_program.image_cropper`, `Otolits_identyfication_program.input_handler`, `Otolits_identyfication_program.row_detector` |
| `Otolits_identyfication_program.input_handler` | `Otolits_identyfication_program.bounding_box`, `Otolits_identyfication_program.image_cropper`, `Otolits_identyfication_program.row_detector` |
| `Otolits_identyfication_program.main` | `Otolits_identyfication_program.auto_detector`, `Otolits_identyfication_program.bounding_box_manager`, `Otolits_identyfication_program.image_loader`, `Otolits_identyfication_program.image_window`, `Otolits_identyfication_program.input_handler` |
| `Otolits_identyfication_program.row_detector` | `Otolits_identyfication_program.bounding_box` |
| `Otolits_identyfication_program.tests.__init__` | — |
| `Otolits_identyfication_program.tests.bounding_box_manager_test` | `Otolits_identyfication_program.bounding_box`, `Otolits_identyfication_program.bounding_box_manager` |
| `Otolits_identyfication_program.tests.image_loader_test` | `Otolits_identyfication_program.image_loader` |
| `Otolits_identyfication_program.web.__init__` | — |
| `Otolits_identyfication_program.web.app` | `Otolits_identyfication_program.auto_detector` |
| `Otolits_identyfication_program.web.config` | — |
| `Otolits_identyfication_program.web.routers.__init__` | — |
| `Otolits_identyfication_program.web.routers.crop` | `Otolits_identyfication_program.bounding_box`, `Otolits_identyfication_program.image_cropper`, `Otolits_identyfication_program.row_detector` |
| `Otolits_identyfication_program.web.routers.detect` | — |
| `Otolits_identyfication_program.web.routers.fs` | — |
| `Otolits_identyfication_program.web.routers.image` | — |
| `Otolits_identyfication_program.web.routers.scales` | — |
| `Otolits_identyfication_program.web.services.__init__` | — |
| `Otolits_identyfication_program.web.services.fs_browser` | `Otolits_identyfication_program.web.config` |
| `Otolits_identyfication_program.web.services.image_service` | `Otolits_identyfication_program.image_loader`, `Otolits_identyfication_program.web.config` |
| `Otolits_identyfication_program.web.services.scales` | `Otolits_identyfication_program.web.config` |
| `Picks_modification_scripts.Resize` | — |

## External libraries

Third-party top-level packages imported across the project:

- `PIL`
- `cv2`
- `exifread`
- `fastapi`
- `numpy`
- `pydantic`
- `pytest`
- `ultralytics`

## Entry points and their fan-out

- `Otolits_identyfication_program/YOLO/yolo_trainer.py` → `Otolits_identyfication_program.bounding_box_manager`
- `Otolits_identyfication_program/main.py` → `Otolits_identyfication_program.auto_detector`, `Otolits_identyfication_program.bounding_box_manager`, `Otolits_identyfication_program.image_loader`, `Otolits_identyfication_program.image_window`, `Otolits_identyfication_program.input_handler`
- `Otolits_identyfication_program/tests/image_loader_test.py` → `Otolits_identyfication_program.image_loader`
- `Picks_modification_scripts/Resize.py` → —
