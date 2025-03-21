class InputHandler:
    def __init__(self, bounding_box_manager, row_manager):
        self.bounding_box_manager = bounding_box_manager
        self.row_manager = row_manager
        self.mode = "manual"

    def set_mode(self, mode):
        self.mode = mode

    def mouse_callback(self, event, x, y, flags, param):
        pass

    def keyboard_callback(self, key):
        pass


# import cv2
#
# class MouseHandler:
#     def __init__(self, bounding_box_manager):
#         self.bbox_manager = bounding_box_manager
#         self.drawing = False
#         self.start_x = None
#         self.start_y = None
#
#     def mouse_callback(self, event, x, y, flags, param):
#         """Obsługuje zdarzenia myszy."""
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.drawing = True
#             self.start_x, self.start_y = x, y
#
#         elif event == cv2.EVENT_MOUSEMOVE:
#             if self.drawing:
#                 self.temp_box = (self.start_x, self.start_y, x, y)
#
#         elif event == cv2.EVENT_LBUTTONUP:
#             self.drawing = False
#             if self.start_x != x and self.start_y != y:
#                 self.bbox_manager.add_box(self.start_x, self.start_y, x, y)
#
#         elif event == cv2.EVENT_RBUTTONDOWN:
#             self.bbox_manager.remove_box(x, y)
