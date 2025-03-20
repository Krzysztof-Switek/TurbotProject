import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Ścieżki katalogów
INPUT_DIR = "C:\\Users\\kswitek\\Documents\\Turbot\\TUR_images"
OUTPUT_DIR = "TUR_resized"
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB

# Tworzenie katalogu wyjściowego, jeśli nie istnieje
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        quality = 95  # Początkowa jakość JPG
        scale_factor = 1.0  # Początkowy współczynnik skalowania

        while True:
            img_resized = img.resize(
                (int(img.width * scale_factor), int(img.height * scale_factor)),
                Image.LANCZOS
            )

            img_resized.save(output_path, "JPEG", quality=quality)

            # Jeśli plik jest mniejszy niż 1MB lub jakość spadła za bardzo, przerywamy
            if os.path.getsize(output_path) <= MAX_FILE_SIZE or quality < 30:
                break

            # Zmniejszamy jakość i współczynnik skalowania
            quality -= 5
            scale_factor *= 0.95


def process_images():
    for filename in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            resize_image(input_path, output_path)
            print(f"Przetworzono: {filename}")


if __name__ == "__main__":
    process_images()
