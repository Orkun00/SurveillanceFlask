import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# (Optional) If using Windows, set the tesseract cmd path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def order_points(pts):
    # Order the four points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # Order points and compute the perspective transform
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Define destination points for a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_license_plate(image):
    # Convert image to grayscale, blur and edge detect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)

    # Find and sort contours by area (largest first)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # Look for a contour with four corners
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        print("No plate contour detected.")
        return None

    pts = plate_contour.reshape(4, 2)
    warped_plate = four_point_transform(image, pts)
    return warped_plate

def upscale_image(image, model_path="EDSR_x4.pb", model_name="edsr", scale=4):
    # Initialize and use OpenCV's dnn_superres to upscale the image
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, scale)
    upscaled = sr.upsample(image)
    return upscaled

def preprocess_for_ocr(image):
    # Convert to grayscale and apply Otsu's thresholding for binary image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_plate_text(image_path, super_res_model_path="EDSR_x4.pb", super_res_model_name="edsr", scale=4, tesseract_config="--psm 7"):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return None

    # Detect and align the license plate
    plate = detect_license_plate(image)
    if plate is None:
        print("License plate not detected!")
        return None

    # Upscale the detected plate image
    upscaled_plate = upscale_image(plate, model_path=super_res_model_path, model_name=super_res_model_name, scale=scale)

    # Preprocess for OCR (thresholding)
    ocr_ready = preprocess_for_ocr(upscaled_plate)

    # (Optional) Display intermediate result
    plt.imshow(ocr_ready, cmap='gray')
    plt.title("Preprocessed Plate for OCR")
    plt.axis('off')
    plt.show()

    # Extract text using Tesseract
    text = pytesseract.image_to_string(ocr_ready, config=tesseract_config)
    # Clean up the text. It can only contain numbers and letters
    text = ''.join(filter(str.isalnum, text))
    # Remove any leading/trailing whitespace
    text = text.strip()
    # Optionally, you can also convert to uppercase
    text = text.upper()

    return text
