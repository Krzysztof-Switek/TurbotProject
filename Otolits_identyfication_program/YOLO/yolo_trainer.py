import subprocess
from ultralytics import YOLO
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager


class YoloTrainer:
    def __init__(self, model, data, imgsz=640, device='cpu', workers=0, batch=2, epochs=20, patience=5,
                 name='turbot_results', amp=False, bounding_box_manager=None):
        self.model = model
        self.data = data
        self.imgsz = imgsz
        self.device = device
        self.workers = workers
        self.batch = batch
        self.epochs = epochs
        self.patience = patience
        self.name = name
        self.amp = amp
        self.bounding_box_manager = bounding_box_manager  # Obiekt do zarządzania boxami

    def train(self):
        # Polecenie treningu YOLOv8
        command = [
            'yolo', 'train',
            f'model={self.model}',
            f'data={self.data}',
            f'imgsz={self.imgsz}',
            f'device={self.device}',
            f'workers={self.workers}',
            f'batch={self.batch}',
            f'epochs={self.epochs}',
            f'patience={self.patience}',
            f'name={self.name}',
            f'amp={str(self.amp).lower()}'
        ]

        try:
            subprocess.run(command, check=True)
            print("Training completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during training: {e}")

    def detect_objects(self, image_path):
        # Wykrywanie obiektów na podstawie wytrenowanego modelu
        model = YOLO(self.model)  # Załaduj wytrenowany model YOLO
        results = model(image_path)  # Wykrywanie obiektów na obrazie

        # Przetwarzanie wyników detekcji
        for box in results.xyxy[0].cpu().numpy():  # Iterujemy po wykrytych boxach
            x1, y1, x2, y2, conf, cls = box  # Współrzędne boxa, pewność, klasa
            self.bounding_box_manager.add_box(int(x1), int(y1), int(x2), int(y2))  # Dodajemy box do BoundingBoxManager

        print(f"Detected {len(results.xyxy[0])} objects.")


# Test integracji z BoundingBoxManager
if __name__ == "__main__":
    bounding_box_manager = BoundingBoxManager()  # Utwórz obiekt BoundingBoxManager
    trainer = YoloTrainer(model='yolo11l.pt', data='datasets/turbot.yaml', bounding_box_manager=bounding_box_manager)

    # Trenuj model (to uruchomi polecenie treningowe)
    trainer.train()

    # Wykonaj detekcję na obrazie (można podać ścieżkę do obrazu, np. "datasets/test_image.jpg")
    trainer.detect_objects('datasets/test_image.jpg')

    # Wyświetl wykryte boxy
    print(f"Detected bounding boxes: {bounding_box_manager.get_boxes()}")
