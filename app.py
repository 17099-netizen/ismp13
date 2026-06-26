import os
import cv2
import numpy as np
import urllib.request
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # อนุญาตให้เว็บจำลองเรียกใช้งานได้

# ---------------------------------------------------------
# 🔒 ตั้งค่า API KEY (รหัสผ่านสำหรับเข้าใช้ API)
# คุณสามารถเปลี่ยนข้อความ "my_secret_key_12345" เป็นรหัสของคุณเองได้เลย
# ---------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "my_secret_key_12345")

DATA_FILE = "known_faces.pkl"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET_PATH = "face_detection_yunet.onnx"
SFACE_PATH = "face_recognition_sface.onnx"

def download_models():
    if not os.path.exists(YUNET_PATH):
        print("Downloading YuNet...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    if not os.path.exists(SFACE_PATH):
        print("Downloading SFace...")
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)

download_models()

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

# ---------------------------------------------------------
# ฟังก์ชันตรวจสอบ API Key
# ---------------------------------------------------------
def require_api_key(req):
    # ตรวจสอบ Key จาก Headers หรือ Form Data
    key = req.headers.get('x-api-key')
    if not key:
        key = req.form.get('api_key')
    return key == API_KEY

# ---------------------------------------------------------
# ฟังก์ชันจำลองการใส่หน้ากากอนามัย (Virtual Mask)
# ---------------------------------------------------------
def add_virtual_mask(img, face_data):
    masked_img = img.copy()
    x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
    nose_y = int(face_data[9])
    
    mask_top = max(0, nose_y - int(h * 0.05))
    mask_bottom = min(img.shape[0], y + h)
    x_start = max(0, x)
    x_end = min(img.shape[1], x + w)
    
    cv2.rectangle(masked_img, (x_start, mask_top), (x_end, mask_bottom), (255, 255, 255), -1)
    return masked_img

def get_face_data(img):
    height, width, _ = img.shape
    detector.setInputSize((width, height))
    faces = detector.detect(img)
    if faces[1] is None:
        return None
    return faces[1][0]

@app.route("/")
def index():
    return jsonify({"status": "Face API is running securely. API Key required."})

@app.route("/register", methods=["POST"])
def register():
    # ตรวจสอบรหัสผ่านก่อนทำงาน
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 401
        
    if 'file' not in request.files or 'id' not in request.form:
        return jsonify({"error": "Missing 'file' or 'id'"}), 400
        
    student_id = str(request.form['id']).strip()
    file = request.files['file']
    
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
        
    face = get_face_data(img)
    if face is None:
        return jsonify({"error": "No face detected in the image"}), 400
        
    if student_id not in known_faces:
        known_faces[student_id] = []
        
    # 1. สกัดจุดเด่นหน้าปกติ
    aligned_normal = recognizer.alignCrop(img, face)
    feature_normal = recognizer.feature(aligned_normal)
    known_faces[student_id].append(feature_normal)
    
    # 2. จำลองใส่แมสและสกัดจุดเด่น
    masked_img = add_virtual_mask(img, face)
    aligned_masked = recognizer.alignCrop(masked_img, face)
    feature_masked = recognizer.feature(aligned_masked)
    known_faces[student_id].append(feature_masked)
    
    save_known_faces(known_faces)
    return jsonify({"message": f"Successfully registered ID: {student_id} (Normal & Masked saved)"}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    # ตรวจสอบรหัสผ่านก่อนทำงาน
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 401
        
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file'"}), 400
        
    if not known_faces:
        return jsonify({"error": "No faces registered"}), 400
        
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
        
    face = get_face_data(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400
        
    aligned_face = recognizer.alignCrop(img, face)
    feature = recognizer.feature(aligned_face)
        
    best_match_id = None
    best_score = 0.0
    THRESHOLD = 0.28  # ลดจาก 0.34 → 0.28 รองรับแสงและมุมที่ต่างจากตอน register

    for student_id, features_list in known_faces.items():
        if isinstance(features_list, list):
            for known_feature in features_list:
                score = recognizer.match(known_feature, feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_match_id = student_id
        else:
            score = recognizer.match(features_list, feature, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_match_id = student_id

    if best_score >= THRESHOLD:
        return jsonify({"id": best_match_id, "score": round(float(best_score), 4)}), 200
    else:
        # คืน 200 พร้อม error field — ไม่ใช่ 404 เพื่อไม่ให้ PHP เข้าใจผิดว่า server error
        return jsonify({"error": "Unknown face", "score": round(float(best_score), 4)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
