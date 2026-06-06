import os
import cv2
import numpy as np
import urllib.request
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# File to store the extracted face data
DATA_FILE = "known_faces.pkl"

# URLs for OpenCV's highly optimized and lightweight models
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

YUNET_PATH = "face_detection_yunet.onnx"
SFACE_PATH = "face_recognition_sface.onnx"

def download_models():
    if not os.path.exists(YUNET_PATH):
        print("Downloading YuNet face detection model...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    if not os.path.exists(SFACE_PATH):
        print("Downloading SFace recognition model...")
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)

# Download models on startup
download_models()

# Initialize models
detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

def load_known_faces():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_known_faces(known_faces):
    with open(DATA_FILE, "wb") as f:
        pickle.dump(known_faces, f)

known_faces = load_known_faces()

def get_face_feature(img):
    height, width, _ = img.shape
    detector.setInputSize((width, height))
    faces = detector.detect(img)
    
    if faces[1] is None:
        return None
        
    # Get the first face
    face = faces[1][0]
    
    # Align and extract feature
    aligned_face = recognizer.alignCrop(img, face)
    feature = recognizer.feature(aligned_face)
    return feature

@app.route("/")
def index():
    return jsonify({"status": "Face API is running successfully. Models loaded."})

@app.route("/register", methods=["POST"])
def register():
    """
    Upload an image along with a student ID to register.
    Expected form data: 'id' and 'file'
    """
    if 'file' not in request.files or 'id' not in request.form:
        return jsonify({"error": "Missing 'file' or 'id'"}), 400
        
    student_id = str(request.form['id']).strip()
    file = request.files['file']
    
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
        
    feature = get_face_feature(img)
    if feature is None:
        return jsonify({"error": "No face detected in the image"}), 400
        
    known_faces[student_id] = feature
    save_known_faces(known_faces)
    
    return jsonify({"message": f"Successfully registered ID: {student_id}"}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Upload an image to recognize who it is.
    Expected form data: 'file'
    Returns: The matching student ID if successful.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file'"}), 400
        
    if not known_faces:
        return jsonify({"error": "No faces registered in the system yet"}), 400
        
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
        
    feature = get_face_feature(img)
    if feature is None:
        return jsonify({"error": "No face detected in the image"}), 400
        
    best_match_id = None
    best_score = 0.0
    
    # SFace cosine similarity threshold (>= 0.363 is generally a match)
    THRESHOLD = 0.363
    
    for student_id, known_feature in known_faces.items():
        score = recognizer.match(known_feature, feature, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = score
            best_match_id = student_id
            
    if best_score >= THRESHOLD:
        return jsonify({"id": best_match_id}), 200
    else:
        return jsonify({"error": "Unknown face"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
