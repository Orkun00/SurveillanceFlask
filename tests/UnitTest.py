import os
import unittest
import io
import zipfile

import cv2
import numpy as np
from unittest.mock import patch
from app import app


class FlaskImageTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    @patch('app.compare_faces')
    @patch('app.get_face_embedding')
    def test_valid_image_upload_with_mocks(self, mock_get_face_embedding, mock_compare_faces):
        # Dummy embedding for mocking
        dummy_embedding = np.zeros((128,))
        # Simulate the same embedding returned twice
        mock_get_face_embedding.side_effect = [dummy_embedding, dummy_embedding]
        # Simulate a successful face comparison
        mock_compare_faces.return_value = True

        # Create a dummy valid image (a 100x100 black image)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', dummy_image)
        self.assertTrue(ret, "Image encoding failed in test setup")
        image_bytes = io.BytesIO(buffer.tobytes())

        data = {
            'image': (image_bytes, 'dummy.jpg')
        }
        response = self.client.post('/detect', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['message'], 'Image successfully received and processed.')
        self.assertIn('is_same', response.json)
        self.assertTrue(response.json['is_same'])
        # Verify that the mocked functions were called the expected number of times.
        self.assertEqual(mock_get_face_embedding.call_count, 2)
        mock_compare_faces.assert_called_once_with(dummy_embedding, dummy_embedding)

    def test_no_file_uploaded(self):
        # No file provided in the request.
        response = self.client.post('/detect', data={})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json['success'])
        self.assertIn('error', response.json)

    def test_empty_file_field(self):
        # File field exists but the filename is empty.
        data = {
            'image': (io.BytesIO(b''), '')
        }
        response = self.client.post('/detect', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json['success'])
        self.assertIn('error', response.json)

    def test_invalid_image_data(self):
        # Providing data that cannot be decoded into an image.
        fake_image = io.BytesIO(b'not_a_real_image_file')
        data = {
            'image': (fake_image, 'invalid.jpg')
        }
        response = self.client.post('/detect', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 500)
        self.assertFalse(response.json['success'])
        self.assertIn('error', response.json)

    def test_valid_face_detection_integration(self):
        """
        Integration test using a valid face image.
        This test does not patch the face recognition functions so that
        the real detection and embedding are executed.
        Ensure that 'test_image.jpg' is a valid face image available in your test directory.
        """
        try:
            with open("../photos/test_image.jpg", "rb") as img:
                data = {
                    'image': (io.BytesIO(img.read()), 'test_image.jpg')
                }
                response = self.client.post('/detect', content_type='multipart/form-data', data=data)
        except FileNotFoundError:
            self.skipTest("test_image.jpg not found. Please add a valid face image for integration testing.")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['message'], 'Image successfully received and processed.')
        # Since the same image is processed twice, the embeddings should be similar,
        # and the compare_faces function should return True.
        self.assertIn('is_same', response.json)
        self.assertTrue(response.json['is_same'])

    def test_batch_detect_with_valid_image(self):
        # Try to read your test image from disk
        try:
            with open("../photos/test_image.jpg", "rb") as f:
                image_data = f.read()
        except FileNotFoundError:
            self.skipTest("test_image.jpg not found. Please add a valid face image for testing.")

        # Prepare a batch of images using the same test image file
        data = {
            'image': [
                (io.BytesIO(image_data), 'test_image.jpg'),
                (io.BytesIO(image_data), 'test_image.jpg'),
                (io.BytesIO(image_data), 'test_image.jpg')
            ]
        }

        # Post the batch request to the endpoint
        response = self.client.post('/batch_detect', content_type='multipart/form-data', data=data)

        # Verify the response status code and JSON response
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data.get('success'))
        self.assertEqual(json_data.get('message'), 'Batch processing completed.')
        self.assertIn('results', json_data)
        self.assertEqual(len(json_data['results']), 3)

        # Validate each result in the batch response
        for result in json_data['results']:
            self.assertIn('filename', result)
            # Each result should have either an 'embedding' (if processing succeeded)
            # or an 'error' if something went wrong.
            print(result)
            self.assertTrue('embedding' in result or 'error' in result)

    def test_clustering_three_unique_faces(self):
        image_paths = [
            "../photos/person1.jpg",  # Unique person
            "../photos/person0.jpg",  # Same person as...
            "../photos/person0same.jpg"  # ...this one
        ]

        # Make sure all images exist
        for path in image_paths:
            if not os.path.exists(path):
                self.skipTest(f"Image not found: {path}")

        # Create in-memory zip of the images
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for idx, path in enumerate(image_paths):
                with open(path, 'rb') as img_file:
                    zipf.writestr(f"person{idx + 1}.jpg", img_file.read())
        zip_buffer.seek(0)

        # Send the zip file to the batch_detect endpoint
        data = {
            'zipfile': (zip_buffer, 'test_faces.zip')
        }

        response = self.client.post('/batch_detect', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)

        json_data = response.get_json()
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['message'], 'Clustering completed.')
        self.assertIn('clustered_results', json_data)

        clustered_results = json_data['clustered_results']
        print("Clustered Results:", clustered_results)

        # Check that each result has a filename, face_index, and cluster_id
        for result in clustered_results:
            self.assertIn('filename', result)
            self.assertIn('face_index', result)
            self.assertIn('cluster_id', result)

        # Extract all cluster IDs
        cluster_ids = [result['cluster_id'] for result in clustered_results]
        unique_clusters = set(cluster_ids)

        print("Cluster IDs:", cluster_ids)

        # Since person0.jpg and person0same.jpg should belong to the same person,
        # we expect 2 unique clusters total
        self.assertEqual(len(unique_clusters), 2)


if __name__ == '__main__':
    unittest.main()
