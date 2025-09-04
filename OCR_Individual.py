import argparse
import os
import re
import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz

# -----------------------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# -----------------------------------------------------------------------------
def set_tesseract_path(path_arg: str | None):
    """Configura la ruta del ejecutable de Tesseract en Windows."""
    if path_arg:
        pytesseract.pytesseract.tesseract_cmd = path_arg
    else:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win

# -----------------------------------------------------------------------------
# PIPELINE DE PREPROCESAMIENTO AVANZADO
# -----------------------------------------------------------------------------
def create_processing_variants(roi, save_debug=None, base="debug"):
    """
    Crea una lista de imágenes candidatas para el OCR usando diferentes
    estrategias de preprocesamiento para maximizar las chances de éxito.
    """
    variants = []
    if roi is None or roi.size == 0:
        return variants

    # Convertir a escala de grises como base para la mayoría de las técnicas
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # --- ESTRATEGIA 1: Mejora de Contraste con CLAHE ---
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) es excelente para
    # realzar detalles en imágenes con brillos y sombras sin sobreexponer.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    variants.append(contrast_enhanced)

    # --- ESTRATEGIA 2: Binarización Adaptativa Invertida ---
    # Ideal para condiciones de iluminación no uniformes.
    # Se invierte (bitwise_not) porque los números son oscuros sobre fondo claro.
    # Tesseract a menudo funciona mejor con texto negro sobre fondo blanco.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(cv2.bitwise_not(adaptive_thresh))

    # --- ESTRATEGIA 3: Binarización de Otsu sobre imagen con contraste mejorado ---
    # Una binarización simple pero efectiva después de haber mejorado el contraste.
    _, otsu_thresh = cv2.threshold(
        contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(cv2.bitwise_not(otsu_thresh))

    # --- Guardar imágenes de depuración ---
    if save_debug:
        os.makedirs(save_debug, exist_ok=True)
        cv2.imwrite(os.path.join(save_debug, f"{base}_0_roi_gray.png"), gray)
        cv2.imwrite(os.path.join(save_debug, f"{base}_1_clahe.png"), variants[0])
        cv2.imwrite(os.path.join(save_debug, f"{base}_2_adaptive.png"), variants[1])
        cv2.imwrite(os.path.join(save_debug, f"{base}_3_otsu.png"), variants[2])
        print(f"⚙️  Imágenes de depuración guardadas en: {save_debug}")

    return variants

# -----------------------------------------------------------------------------
# EJECUCIÓN DE OCR
# -----------------------------------------------------------------------------
def run_ocr_on_variants(variants):
    """
    Ejecuta Tesseract en todas las variantes de imagen y devuelve el mejor
    resultado posible, priorizando los que tienen una longitud esperada.
    """
    best_text = ""
    
    for i, variant in enumerate(variants):
        # Escalar la variante para darle a Tesseract dígitos más grandes
        scaled_variant = cv2.resize(variant, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Probar con los modos de segmentación más relevantes
        for psm in [7, 8, 6]:
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(scaled_variant, config=config)
            cleaned_text = re.sub(r"\D", "", text) # Limpiar nuevamente por si acaso

            # Lógica de selección del "mejor" texto:
            # Si encontramos un texto de 6 dígitos, es casi seguro el correcto.
            if len(cleaned_text) == 6:
                return cleaned_text # Devolver inmediatamente
            # Si no, nos quedamos con el más largo que hayamos encontrado hasta ahora.
            if len(cleaned_text) > len(best_text):
                best_text = cleaned_text

    return best_text

# -----------------------------------------------------------------------------
# INTERFAZ Y EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def select_roi_interactive(image_path: str):
    """Permite al usuario seleccionar manualmente una ROI en la imagen."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    window_name = "Selecciona el sticker (ENTER para confirmar, C para cancelar)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Ajustar tamaño de ventana a la pantalla si la imagen es muy grande
    screen_h, screen_w = 1080, 1920 # Valores típicos, ajusta si es necesario
    img_h, img_w, _ = img.shape
    scale = min(screen_w / img_w, screen_h / img_h, 1) # No escalar más grande que 1
    win_w, win_h = int(img_w * scale), int(img_h * scale)
    cv2.resizeWindow(window_name, win_w, win_h)

    r = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = r
    return img[y:y+h, x:x+w] if w > 0 and h > 0 else None

def main():
    parser = argparse.ArgumentParser(description="OCR de stickers v3: Enfoque Múltiple y Robusto.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen para procesar.")
    parser.add_argument("--expected", default=None, help="(Opcional) Número esperado para comparar.")
    parser.add_argument("--tesseract", default=None, help="(Opcional) Ruta al ejecutable de Tesseract.")
    parser.add_argument("--save_debug", default=None, help="(Opcional) Carpeta para guardar imágenes de preprocesamiento.")

    args = parser.parse_args()
    set_tesseract_path(args.tesseract)

    try:
        # 1. Selección interactiva
        roi = select_roi_interactive(args.image)
        if roi is None:
            print("ℹ️  No se seleccionó una ROI. Saliendo.")
            return

        # 2. Generar variantes de preprocesamiento
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        variants = create_processing_variants(roi, save_debug=args.save_debug, base=base_name)

        # 3. Ejecutar OCR y obtener el mejor resultado
        detected_text = run_ocr_on_variants(variants)

        # 4. Mostrar resultados
        print("\n" + "="*40)
        print(f"📄 Resultados para: {os.path.basename(args.image)}")
        print(f"✅ Número detectado: {detected_text if detected_text else '--- No se detectó ningún número ---'}")

        if args.expected:
            score = fuzz.ratio(args.expected, detected_text or "")
            print(f"🎯 Número esperado:   {args.expected}")
            print(f"✨ Similitud (RapidFuzz): {score:.2f}%")
            if score > 90:
                print("👍 ¡Coincidencia Alta!")
            else:
                print("👎 Coincidencia Baja o Nula.")
        print("="*40)

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()