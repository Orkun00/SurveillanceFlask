import unittest
import io
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
            with open("test_image.jpg", "rb") as img:
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

if __name__ == '__main__':
    unittest.main()
