import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

class GUI:
    def __init__(self, image_loader, bbox_manager, row_manager, input_handler, yolo_model=None):
        self.image_loader = image_loader
        self.bbox_manager = bbox_manager
        self.row_manager = row_manager
        self.input_handler = input_handler
        self.yolo_model = yolo_model
        self.mode = "manual"  # Domyślny tryb
        self.current_image = None
        self.tk_image = None

        # Inicjalizacja głównego okna
        self.root = tk.Tk()
        self.root.title("Otolit - Identyfikacja")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Główne ramki
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Przyciski
        self.btn_load = tk.Button(self.top_frame, text="Wczytaj obraz", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT)

        self.btn_prev = tk.Button(self.top_frame, text="Poprzedni", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(self.top_frame, text="Następny", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT)

        self.btn_auto_detect = tk.Button(self.top_frame, text="Auto Detect", command=self.auto_detect)
        self.btn_auto_detect.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(self.top_frame, text="Zapisz", command=self.save_results)
        self.btn_save.pack(side=tk.RIGHT)

        # Canvas do wyświetlania obrazu
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Zmienne do rysowania
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # Etykieta statusu
        self.status_label = tk.Label(self.bottom_frame, text="Gotowy", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

    def load_image(self):
        """Wczytuje nowy obraz i wyświetla go na canvasie."""
        image_path = filedialog.askopenfilename()
        if not image_path:
            return

        try:
            # Używamy ImageLoader do wczytania obrazu
            self.image_loader.image_paths = [image_path]
            self.image_loader.current_image_index = 0
            self.current_image = self.image_loader.load_image()

            if self.current_image is not None:
                self.display_image(self.current_image)
                self.status_label.config(text=f"Wczytano: {os.path.basename(image_path)}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać obrazu: {e}")

    def display_image(self, image_cv):
        """Konwertuje obraz z formatu OpenCV do formatu Tkinter i wyświetla go."""
        self.bbox_manager.clear_boxes() # Wyczyść stare prostokąty
        # Konwersja kolorów z BGR do RGB
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        # Konwersja do formatu PIL
        image_pil = Image.fromarray(image_rgb)
        # Konwersja do formatu Tkinter
        self.tk_image = ImageTk.PhotoImage(image_pil)

        # Aktualizacja canvasu
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.draw_boxes() # Narysuj prostokąty po wyświetleniu obrazu

    def prev_image(self):
        """Wyświetla poprzedni obraz."""
        self.current_image = self.image_loader.prev_image()
        if self.current_image is not None:
            self.display_image(self.current_image)

    def next_image(self):
        """Wyświetla następny obraz."""
        self.current_image = self.image_loader.next_image()
        if self.current_image is not None:
            self.display_image(self.current_image)

    def save_results(self):
        """Zapisuje wyniki (do implementacji)."""
        # Tutaj logika zapisywania bounding boxów
        messagebox.showinfo("Zapisano", "Wyniki zostały zapisane.")

    def on_closing(self):
        """Obsługa zamknięcia okna."""
        if messagebox.askokcancel("Wyjście", "Czy na pewno chcesz wyjść?"):
            self.root.destroy()

    def on_mouse_press(self, event):
        """Rozpoczyna rysowanie prostokąta."""
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_mouse_drag(self, event):
        """Aktualizuje rozmiar rysowanego prostokąta."""
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_release(self, event):
        """Kończy rysowanie prostokąta i dodaje go do managera."""
        if self.rect_id:
            x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
            self.bbox_manager.add_box(x1, y1, x2, y2)
            self.canvas.delete(self.rect_id) # Usuń tymczasowy prostokąt
            self.rect_id = None
            self.draw_boxes() # Narysuj wszystkie prostokąty od nowa

    def draw_boxes(self):
        """Rysuje wszystkie bounding boxy na obrazie."""
        self.canvas.delete("box") # Usuń stare prostokąty
        for box in self.bbox_manager.boxes:
            x1, y1, x2, y2 = box.get_coords()
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="box")

    def auto_detect(self):
        """Uruchamia automatyczne wykrywanie otolitów."""
        if self.current_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        if self.yolo_model and self.yolo_model.model:
            self.status_label.config(text="Wykrywanie otolitów...")
            self.root.update_idletasks() # Odśwież etykietę

            auto_boxes = self.yolo_model.detect(self.current_image)

            if auto_boxes:
                for (x1, y1, x2, y2) in auto_boxes:
                    self.bbox_manager.add_box(x1, y1, x2, y2, label="auto")
                self.draw_boxes()
                self.status_label.config(text=f"Znaleziono {len(auto_boxes)} otolitów.")
            else:
                self.status_label.config(text="Nie wykryto otolitów.")
        else:
            messagebox.showerror("Błąd", "Model YOLO nie jest załadowany.")

    def run(self):
        """Uruchamia główną pętlę aplikacji."""
        self.root.mainloop()