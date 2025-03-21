class GUI:
    def __init__(self, image_loader, bbox_manager, row_manager, input_handler, yolo_model=None):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.row_manager = row_manager
        self.input_handler = input_handler
        self.yolo_model = yolo_model
        self.mode = "manual"

    def set_mode(self, mode):
        self.mode = mode

    def run(self):
        pass