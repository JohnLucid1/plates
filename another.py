from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')

# Load the replacement image
replacement_image = cv2.imread('plate.png')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', "webp"}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_smooth_mask(mask, kernel_size=15):
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

def replace_plate(image, plate_coords, replacement_image, padding=20, kernel_size=25):
    x1, y1, x2, y2 = map(int, plate_coords)

    # Add padding to the plate coordinates
    y1 = max(0, y1 - padding)
    y2 = min(image.shape[0], y2 + padding)

    # Create a binary mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Smooth the mask with a larger kernel
    smooth_mask = apply_smooth_mask(mask, kernel_size=kernel_size)

    # Normalize the mask
    smooth_mask = smooth_mask.astype(float) / 255.0
    # Resize replacement image to match the plate size
    replacement_resized = cv2.resize(replacement_image, (x2-x1, y2-y1))

    # Create a 3-channel mask
    mask_3channel = np.repeat(smooth_mask[:, :, np.newaxis], 3, axis=2)

    # Replace the plate area
    result = image.copy()
    result[y1:y2, x1:x2] = (1 - mask_3channel[y1:y2, x1:x2]) * result[y1:y2, x1:x2] + \
                           mask_3channel[y1:y2, x1:x2] * replacement_resized

    return result.astype(np.uint8)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'car_image' not in request.files or 'car_plate' not in request.files:
        return jsonify({'error': 'Both car_image and car_plate are required'}), 400

    car_image = request.files['car_image']
    car_plate = request.files['car_plate']

    if car_image.filename == '' or car_plate.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(car_image.filename) or not allowed_file(car_plate.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Read images from bytes
    car_image_bytes = car_image.read()
    car_plate_bytes = car_plate.read()

    try:
        # Convert bytes to OpenCV images
        car_image = cv2.imdecode(np.frombuffer(car_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        car_plate = cv2.imdecode(np.frombuffer(car_plate_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)

        # Convert back to color if necessary
        if len(car_image.shape) == 3:
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        results = model(enhanced_image, conf=0.70)

        # Process the results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                replaced_image = replace_plate(car_image, box, car_plate, padding=8, kernel_size=25)

                # Convert the processed image to bytes
                is_success, buffer = cv2.imencode(".jpg", replaced_image)
                io_buf = io.BytesIO(buffer)

                return io_buf.getvalue(), 200, {'Content-Type': 'image/jpeg'}

        return jsonify({'error': 'No car plate detected'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')