from flask import Flask, request, jsonify
import cv2
import numpy as np
import traceback
from face_recog_lib import get_face_embedding, compare_faces
from sklearn.cluster import DBSCAN
import zipfile
import io
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
        if 'zipfile' not in request.files:
            return jsonify({'success': False, 'error': 'No zip file found in request.'}), 400

        zip_file = request.files['zipfile']
        zip_buffer = io.BytesIO(zip_file.read())

        embeddings = []
        filenames = []
        results = []

        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                try:
                    file_data = zip_ref.read(filename)
                    file_bytes = np.frombuffer(file_data, np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if image is None:
                        results.append({'filename': filename, 'error': 'Image decode failed.'})
                        continue

                    embedding = get_face_embedding(image)
                    embeddings.append(embedding)
                    filenames.append(filename)

                except Exception as e:
                    results.append({'filename': filename, 'error': str(e)})

        if not embeddings:
            return jsonify({'success': False, 'error': 'No valid embeddings extracted.'}), 400

        embeddings_array = np.vstack(embeddings)  # shape (N, 512)

        clustering_model = DBSCAN(eps=0.6, min_samples=1, metric='cosine')
        clustering_model.fit(embeddings_array)

        labels = clustering_model.labels_

        clustered_results = []
        for filename, label in zip(filenames, labels):
            clustered_results.append({'filename': filename, 'cluster_id': int(label)})

        return jsonify({
            'success': True,
            'message': 'Clustering completed.',
            'num_clusters': len(set(labels)),
            'clustered_results': clustered_results
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Unexpected error: ' + str(e)}), 500


if __name__ == '__main__':
    app.run("0,0.0.0", 5000, debug=True, threaded=True)
