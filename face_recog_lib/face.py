import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize the face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU; use 0 for GPU if available

def get_face_embedding(image_input):
    """
    Extract face embedding from an image.

    Args:
        image_input (str or np.ndarray): Either the path to the image file or the image array itself.

    Returns:
        np.ndarray: The embedding of the first detected face in the image.

    Raises:
        ValueError: If the image cannot be read or no faces are detected.
    """
    # Check if input is a file path (string)
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image: {image_input}")
    # Otherwise, assume it's a numpy array
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("Invalid input type: expected a file path (str) or image array (np.ndarray)")

    # Process the image to detect faces and extract embeddings
    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    return faces

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
    return similarity > threshold
