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

        all_embeddings = []
        face_info_list = []

        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                try:
                    file_data = zip_ref.read(filename)
                    file_bytes = np.frombuffer(file_data, np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if image is None:
                        face_info_list.append({'filename': filename, 'error': 'Image decode failed.'})
                        continue

                    faces = get_face_embedding(image)
                    if not isinstance(faces, list):
                        raise ValueError("get_face_embedding must return a list of face objects")

                    for idx, face in enumerate(faces):
                        embedding = face.embedding  # ✅ Extract the actual embedding from the face object

                        if not isinstance(embedding, np.ndarray):
                            raise ValueError(f"Face {idx} in {filename} does not contain a valid NumPy embedding")

                        all_embeddings.append(embedding)
                        face_info_list.append({
                            'filename': filename,
                            'face_index': idx
                        })

                except Exception as e:
                    face_info_list.append({'filename': filename, 'error': str(e)})

        if not all_embeddings:
            return jsonify({'success': False, 'error': 'No face embeddings extracted.'}), 400

        embeddings_array = np.vstack(all_embeddings)

        clustering_model = DBSCAN(eps=0.6, min_samples=1, metric='cosine')
        clustering_model.fit(embeddings_array)
        labels = clustering_model.labels_

        clustered_results = []
        for face_info, cluster_id in zip(face_info_list, labels):
            result = face_info.copy()
            result['cluster_id'] = int(cluster_id)
            clustered_results.append(result)

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
