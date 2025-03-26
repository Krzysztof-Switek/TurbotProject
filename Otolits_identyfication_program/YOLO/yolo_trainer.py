import os
import cv2
from ultralytics import YOLO
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager

class YoloTrainer:
    def __init__(self, model, data, imgsz=640, device='cpu', workers=0, batch=4, epochs=200, patience=50,
                 name='turbot_results', amp=False, single_cls=True, bounding_box_manager=None):
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
        self.single_cls = single_cls
        self.bounding_box_manager = bounding_box_manager

    def train(self):
        try:
            model = YOLO(self.model)
            model.train(
                data=self.data,
                imgsz=self.imgsz,
                device=self.device,
                workers=self.workers,
                batch=self.batch,
                epochs=self.epochs,
                patience=self.patience,
                name=self.name,
                amp=self.amp,
                single_cls=self.single_cls,
                save_conf=False,
                save_txt=False
            )
            print("Training completed successfully.")
        except Exception as e:
            print(f"Training failed: {e}")

    def detect_objects(self, image_path):
        try:
            model = YOLO(self.model)
            results = model(image_path)
            result_dir = os.path.join(os.getcwd(), self.name)

            # Sprawdź, czy katalog istnieje, jeśli nie, utwórz go
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            for r in results:
                im = r.plot(labels=False)

                # Sprawdzanie formatu obrazu
                print(f"Image type: {type(im)}")

                # Ścieżka do zapisu obrazu
                output_image_path = os.path.join(result_dir, "no_labels_pred.jpg")

                # Zapisz obraz
                if cv2.imwrite(output_image_path, im):
                    print(f"Image saved successfully at: {output_image_path}")
                else:
                    print(f"Failed to save the image at: {output_image_path}")

                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    if self.bounding_box_manager:
                        self.bounding_box_manager.add_box(x1, y1, x2, y2)

            print(f"Detected {len(results[0].boxes)} objects.")
        except Exception as e:
            print(f"Detection failed: {e}")


if __name__ == "__main__":
    bounding_box_manager = BoundingBoxManager()
    trainer = YoloTrainer(model='yolo11l.pt', data='datasets/turbot.yaml', bounding_box_manager=bounding_box_manager)

    # Trenuj model
    trainer.train()

    # Wykonaj detekcję na obrazie
    trainer.detect_objects('/home/kswitek/Documents/TurbotProject/Otolits_identyfication_program/test_images/TUR_BITS_2016_Q1_1.jpg')
