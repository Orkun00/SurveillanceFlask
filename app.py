from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    # Read the image file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'success': False, 'error': 'Image decode failed'}), 500

    # Now you can process the image
    print("Image shape:", image.shape)

    return jsonify({'success': True, 'shape': image.shape}), 200

if __name__ == '__main__':
    app.run("0,0.0.0", 5000, debug=True, threaded=True)
