from flask import Flask, request, jsonify
import cv2
import numpy as np
import traceback
from face_recog_lib import get_face_embedding, compare_faces

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Validate that an image file is provided in the request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file part in the request. Make sure you include an image file.'
            }), 400

        file = request.files['image']
        # Check if the file is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected for uploading. Please choose an image file.'
            }), 400

        # Attempt to read the file as bytes
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Failed to read file as bytes. The file might be corrupted or unreadable. Error: ' + str(e)
            }), 500

        # Attempt to decode the image from the byte array
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({
                'success': False,
                'error': ('Image decode failed. This might be because the uploaded file '
                          'is not a valid image format or it is corrupted.')
            }), 500

        # How to use insightface
        embedding1 = get_face_embedding(image)
        embedding2 = get_face_embedding(image)

        return jsonify({
            'success': True,
            'message': 'Image successfully received and processed.',
            'is_same': bool(compare_faces(embedding1, embedding2)),
        }), 200

    except Exception as e:
        # Print the traceback for debugging purposes
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred: ' + str(e)
        }), 500

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file part in the request. Make sure you include an image file.'
            }), 400

        files = request.files.getlist('image')
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files selected for uploading. Please choose at least one image file.'
            }), 400

        results = []
        for file in files:
            file_results = {'filename': file.filename}
            if file.filename == '':
                file_results['error'] = 'No file selected for this entry.'
                results.append(file_results)
                continue
            try:
                file_bytes = np.frombuffer(file.read(), np.uint8)
            except Exception as e:
                file_results['error'] = 'Failed to read file as bytes. The file might be corrupted or unreadable. Error: ' + str(e)
                results.append(file_results)
                continue

            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                file_results['error'] = 'Image decode failed. This might be because the uploaded file is not a valid image format or it is corrupted.'
                results.append(file_results)
                continue

            try:
                embedding = get_face_embedding(image)
                file_results['embedding'] = embedding.tolist()
            except Exception as e:
                file_results['error'] = 'Face detection or embedding failed. Error: ' + str(e)

            results.append(file_results)

        return jsonify({
            'success': True,
            'message': 'Batch processing completed.',
            'results': results
        }), 200
    except Exception as e:
        # Print the traceback for debugging purposes
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred: ' + str(e)
        }), 500


if __name__ == '__main__':
    app.run("0,0.0.0", 5000, debug=True, threaded=True)
