import face_recognition
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_known_faces(self, faces_dir: str):
        """Load known faces from directory structure"""
        print("Loading known faces...")
        for person_name in os.listdir(faces_dir):
            person_dir = os.path.join(faces_dir, person_name)
            if os.path.isdir(person_dir):
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.known_face_encodings.append(encoding[0])
                        self.known_face_names.append(person_name)
                        print(f"Loaded {person_name} from {image_file}")
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def recognize_faces(self, image_path: str) -> List[Dict]:
        """Recognize faces in an image"""
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Find faces and encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding
            )
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
            else:
                confidence = 1 - face_distances[best_match_index]
            
            top, right, bottom, left = face_location
            results.append({
                "name": name,
                "confidence": round(confidence, 2),
                "location": (top, right, bottom, left),
                "is_authorized": name != "Unknown"
            })
        
        return results

# Example usage
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    # This will be used in our demo
    print("Face Recognizer initialized!")