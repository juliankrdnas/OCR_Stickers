import cv2
import pytesseract
import pandas as pd
import os
from rapidfuzz import fuzz, process

# Configuración de Tesseract (ajusta si es necesario en tu PC)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------
# Preprocesamiento de imagen + detección de regiones
# -------------------------
def extract_sticker_numbers(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ No se pudo abrir la imagen: {image_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Umbral adaptativo para resaltar texto
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # Encontrar contornos de posibles regiones con texto
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stickers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar regiones muy pequeñas o muy grandes
        if h > 20 and w > 20 and h < 300 and w < 300:
            roi = image[y:y+h, x:x+w]

            # OCR en esa región
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(roi, config=config)

            # Mantener solo números
            text = "".join([c for c in text if c.isdigit()])
            if text:
                stickers.append(text)

    return stickers

# -------------------------
# Extraer ID de nombre archivo
# -------------------------
def get_id_from_filename(filename):
    parts = filename.split("_")
    if len(parts) >= 3:
        return parts[2]  # El tercer bloque
    return None

# -------------------------
# Comparar con RapidFuzz
# -------------------------
def find_best_match(ocr_number, database_numbers, threshold=85):
    if not ocr_number:
        return None, 0

    match, score, _ = process.extractOne(
        ocr_number, database_numbers, scorer=fuzz.ratio
    )

    return (match, score) if score >= threshold else (None, score)

# -------------------------
# Procesar todas las imágenes
# -------------------------
def process_images(image_folder, excel_path, output_path):
    # Leer la base de datos desde Excel
    df = pd.read_excel(excel_path)
    if "Sticker" not in df.columns:
        raise ValueError("El Excel debe tener una columna 'Sticker'")

    database_numbers = df["Sticker"].astype(str).tolist()

    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_folder, filename)

            # OCR en la imagen (varias detecciones posibles)
            detected_stickers = extract_sticker_numbers(image_path)

            # ID desde el nombre de archivo
            file_id = get_id_from_filename(filename)

            if detected_stickers:
                for ocr_number in detected_stickers:
                    best_match, score = find_best_match(ocr_number, database_numbers)

                    results.append({
                        "Archivo": filename,
                        "ID_Tarea": file_id,
                        "Numero_OCR": ocr_number,
                        "Coincidencia": best_match,
                        "Similitud(%)": score
                    })
            else:
                results.append({
                    "Archivo": filename,
                    "ID_Tarea": file_id,
                    "Numero_OCR": "No detectado",
                    "Coincidencia": None,
                    "Similitud(%)": 0
                })

    # Guardar resultados en un nuevo Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)
    print(f"✅ Resultados guardados en: {output_path}")


# -------------------------
# EJEMPLO DE USO
# -------------------------
if __name__ == "__main__":
    carpeta_imagenes = r"C:\Users\Julian\OneDrive\Documents\Job\Ilumina\LUIS GONZALO LOPEZ RIVERA - SUPERVISOR 11"
    base_excel = r"C:\Users\Julian\OneDrive\Documents\Job\Ilumina\OCR\ODT_0154_VILLAVICENCIO_01.xlsx"
    salida_excel = r"C:\Users\Julian\OneDrive\Documents\Job\Ilumina\resultados_stickers.xlsx"

    process_images(carpeta_imagenes, base_excel, salida_excel)
