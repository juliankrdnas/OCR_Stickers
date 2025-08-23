
import argparse
import os
import re
import cv2
import pytesseract

def set_tesseract_path(path_arg: str | None):
    if path_arg:
        pytesseract.pytesseract.tesseract_cmd = path_arg
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win

def preprocess_roi(roi):
    """Return two binarized versions (normal and inverted) to try with OCR."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # De-noise a bit
    gray = cv2.medianBlur(gray, 3)

    # Boost contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Scale up (helps OCR on small digits)
    scale = 2.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold (two variants)
    th_norm = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    th_inv = cv2.bitwise_not(th_norm)

    # Light morphology to close small gaps in digits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_norm = cv2.morphologyEx(th_norm, cv2.MORPH_CLOSE, kernel, iterations=1)
    th_inv = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    return th_norm, th_inv

def ocr_digits(img):
    """Run Tesseract constrained to digits; try multiple PSMs and both binarizations."""
    # Try both normal and inverted
    variants = preprocess_roi(img)

    best_text = ""
    best_conf_len = -1

    for variant in variants:
        for psm in (7, 6):  # 7: single line, 6: block of text
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(variant, config=config)
            text = re.sub(r"\D+", "", text)  # keep only digits

            # Heuristic score: prefer longer digit strings
            if len(text) > best_conf_len:
                best_conf_len = len(text)
                best_text = text

    return best_text if best_text else None

def select_roi_interactive(image_path: str):
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
        return None, img  # nothing selected; return full image for fallback
    roi = img[y:y+h, x:x+w]
    return roi, img

def main():
    parser = argparse.ArgumentParser(description="OCR de una sola imagen con recorte manual (ROI) para stickers numéricos.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen (JPG/PNG).")
    parser.add_argument("--expected", default=None, help="(Opcional) Número esperado para comparar.")
    parser.add_argument("--tesseract", default=None, help="(Opcional) Ruta al ejecutable de Tesseract en Windows.")
    parser.add_argument("--save_debug", default=None, help="(Opcional) Carpeta para guardar ROI y binarizaciones.")

    args = parser.parse_args()
    set_tesseract_path(args.tesseract)

    roi, full = select_roi_interactive(args.image)

    # If user canceled ROI, fallback to full image
    target = roi if roi is not None else full

    th_norm, th_inv = preprocess_roi(target)

    # Optional debug saves
    if args.save_debug:
        os.makedirs(args.save_debug, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        if roi is not None:
            cv2.imwrite(os.path.join(args.save_debug, f"{base}_roi.jpg"), roi)
        cv2.imwrite(os.path.join(args.save_debug, f"{base}_th_norm.png"), th_norm)
        cv2.imwrite(os.path.join(args.save_debug, f"{base}_th_inv.png"), th_inv)

    detected = ocr_digits(target)

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
