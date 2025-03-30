import unittest
import io
from app import app

class FlaskImageTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_valid_image_upload(self):
        with open("test_image.jpg", "rb") as img:
            data = {
                'image': (io.BytesIO(img.read()), 'test_image.jpg')
            }
            response = self.client.post('/detect', content_type='multipart/form-data', data=data)

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json['success'])
            self.assertIn('shape', response.json)

    def test_no_file_uploaded(self):
        response = self.client.post('/detect', data={})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json['success'])
        self.assertIn('error', response.json)

    def test_empty_file_field(self):
        data = {
            'image': (io.BytesIO(b''), '')
        }
        response = self.client.post('/detect', content_type='multipart/form-data', data=data)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json['success'])
        self.assertIn('error', response.json)

    def test_invalid_image_data(self):
        fake_image = io.BytesIO(b'not_a_real_image_file')
        data = {
            'image': (fake_image, 'invalid.jpg')
        }
        response = self.client.post('/detect', content_type='multipart/form-data', data=data)

        self.assertEqual(response.status_code, 500)
        self.assertFalse(response.json['success'])
        self.assertIn('error', response.json)

if __name__ == '__main__':
    unittest.main()
