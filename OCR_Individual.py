import argparse
import os
import re
import cv2
import pytesseract
import numpy as np

def set_tesseract_path(path_arg: str | None):
    """Configura la ruta del ejecutable de Tesseract en Windows si no está en PATH."""
    if path_arg:
        pytesseract.pytesseract.tesseract_cmd = path_arg
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win

def preprocess_roi(roi):
    """
    Preprocesa una ROI recortada aplicando varias técnicas de mejora de contraste,
    binarización y morfología. Devuelve una lista de variantes procesadas.
    """
    variants = []

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # --- Escalar (aumenta tamaño de dígitos pequeños) ---
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # --- Mejorar contraste ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    gray_eq = cv2.equalizeHist(gray)

    # --- Suavizado de ruido ---
    blur = cv2.medianBlur(gray_clahe, 3)

    # --- Binarizaciones ---
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    th_adapt_inv = cv2.bitwise_not(th_adapt)

    # --- Morfología ligera (conectar huecos en dígitos) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph_otsu = cv2.morphologyEx(th_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph_adapt = cv2.morphologyEx(th_adapt, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph_adapt_inv = cv2.morphologyEx(th_adapt_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Guardar variantes para probar en OCR
    variants.extend([gray_clahe, gray_eq, morph_otsu, morph_adapt, morph_adapt_inv])

    return variants

def ocr_digits(img, save_debug=None, base="debug"):
    """
    Ejecuta OCR sobre varias variantes de la imagen.
    Devuelve el texto numérico más largo encontrado.
    """
    variants = preprocess_roi(img)
    best_text = ""
    best_len = -1

    for i, variant in enumerate(variants):
        # Guardar para depuración
        if save_debug:
            cv2.imwrite(os.path.join(save_debug, f"{base}_variant_{i}.png"), variant)

        # Probar diferentes configuraciones de Tesseract
        for psm in (7, 6, 11):  # línea, bloque, texto disperso
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(variant, config=config)
            text = re.sub(r"\D+", "", text)  # mantener solo dígitos

            if len(text) > best_len:
                best_len = len(text)
                best_text = text

    return best_text if best_text else None

def select_roi_interactive(image_path: str):
    """
    Permite seleccionar manualmente una ROI en la imagen.
    Si no se selecciona, se devuelve la imagen completa.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    clone = img.copy()
    cv2.namedWindow("Selecciona ROI (ENTER para aceptar, C para cancelar)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selecciona ROI (ENTER para aceptar, C para cancelar)", 1280, 720)
    r = cv2.selectROI("Selecciona ROI (ENTER para aceptar, C para cancelar)", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecciona ROI (ENTER para aceptar, C para cancelar)")

    x, y, w, h = r
    if w == 0 or h == 0:
        return None, img
    roi = img[y:y+h, x:x+w]
    return roi, img

def main():
    parser = argparse.ArgumentParser(description="OCR de stickers numéricos en imágenes con recorte manual.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen (JPG/PNG).")
    parser.add_argument("--expected", default=None, help="(Opcional) Número esperado para comparar.")
    parser.add_argument("--tesseract", default=None, help="(Opcional) Ruta al ejecutable de Tesseract en Windows.")
    parser.add_argument("--save_debug", default=None, help="(Opcional) Carpeta para guardar variantes procesadas.")

    args = parser.parse_args()
    set_tesseract_path(args.tesseract)

    # Seleccionar ROI
    roi, full = select_roi_interactive(args.image)
    target = roi if roi is not None else full

    # Guardar depuración opcional
    save_debug = args.save_debug
    if save_debug:
        os.makedirs(save_debug, exist_ok=True)

    # OCR
    detected = ocr_digits(target, save_debug=save_debug, base=os.path.splitext(os.path.basename(args.image))[0])

    print("====================================")
    print(f"Imagen: {args.image}")
    if roi is not None:
        print("ROI: seleccionado por el usuario ✅")
    else:
        print("ROI: no seleccionado; se usó la imagen completa ⚠️")
    print(f"Detectado (solo dígitos): {detected if detected else '— vacío —'}")

    if args.expected:
        try:
            from rapidfuzz import fuzz
            score = fuzz.ratio(args.expected, detected or "")
            print(f"Esperado: {args.expected} | Similitud (RapidFuzz): {score}%")
        except Exception:
            print("RapidFuzz no está instalado; omitiendo comparación. Instala con: pip install rapidfuzz")

    print("====================================")

if __name__ == "__main__":
    main()
