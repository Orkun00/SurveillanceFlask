import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize the face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU; use 0 for GPU if available

def get_face_embedding(image_path):
    """
    Extract face embedding from an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The embedding of the first detected face in the image.

    Raises:
        ValueError: If the image cannot be read or no faces are detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.65):
    """
    Compare two face embeddings using cosine similarity.

    Args:
        emb1 (np.ndarray): The first face embedding.
        emb2 (np.ndarray): The second face embedding.
        threshold (float, optional): The similarity threshold to determine if the faces match. Defaults to 0.65.

    Returns:
        tuple: A tuple containing the similarity score (float) and a boolean indicating if the similarity exceeds the threshold.
    """
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold
