import os
import cv2
import numpy as np
import urllib.request
import pymysql
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# 🔒 ตั้งค่า API KEY
# ---------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "my_secret_key_12345")

# ---------------------------------------------------------
# 🗄️ ตั้งค่า Database MySQL (Aiven)
# ---------------------------------------------------------
DB_HOST = os.environ.get("DB_HOST", "mysql-36feea8e-chaiyanan-18aa.l.aivencloud.com")
DB_PORT = int(os.environ.get("DB_PORT", 16338)) # Aiven ใช้ Port พิเศษ
DB_USER = os.environ.get("DB_USER", "avnadmin")
DB_PASS = os.environ.get("DB_PASS", "AVNS_mWh0__PWvXDxgp88P0u")
DB_NAME = os.environ.get("DB_NAME", "defaultdb")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET_PATH = "face_detection_yunet.onnx"
SFACE_PATH = "face_recognition_sface.onnx"

def download_models():
    if not os.path.exists(YUNET_PATH):
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    if not os.path.exists(SFACE_PATH):
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)

download_models()

detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

# ---------------------------------------------------------
# ฟังก์ชันเชื่อมต่อ Aiven MySQL (บังคับ SSL)
# ---------------------------------------------------------
def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        ssl={'ssl': {}} # Aiven ต้องการการเชื่อมต่อผ่าน SSL
    )

def init_db():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    feature_data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        conn.close()
        print("Aiven Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing Aiven DB: {e}")

def load_known_faces():
    faces_dict = {}
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id, feature_data FROM face_embeddings")
            rows = cursor.fetchall()
            for row in rows:
                uid = row['user_id']
                feat = np.array(json.loads(row['feature_data']), dtype=np.float32)
                if uid not in faces_dict:
                    faces_dict[uid] = []
                faces_dict[uid].append(feat)
        conn.close()
        print(f"Loaded faces for {len(faces_dict)} users from Aiven.")
    except Exception as e:
        print(f"Error loading faces: {e}")
    return faces_dict

def save_face_to_db(user_id, feature):
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            feature_json = json.dumps(feature.tolist())
            cursor.execute(
                "INSERT INTO face_embeddings (user_id, feature_data) VALUES (%s, %s)",
                (user_id, feature_json)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving to Aiven DB: {e}")

# ถ้าเริ่มเซิร์ฟเวอร์ ให้เชื่อมต่อและโหลดข้อมูลทันที
if DB_HOST:
    init_db()
    known_faces = load_known_faces()
else:
    known_faces = {}
    print("Waiting for Database Environment Variables...")

def require_api_key(req):
    key = req.headers.get('x-api-key') or req.form.get('api_key')
    return key == API_KEY

def add_virtual_mask(img, face_data):
    masked_img = img.copy()
    x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
    nose_y = int(face_data[9])
    mask_top = max(0, nose_y - int(h * 0.05))
    mask_bottom = min(img.shape[0], y + h)
    cv2.rectangle(masked_img, (max(0, x), mask_top), (min(img.shape[1], x + w), mask_bottom), (255, 255, 255), -1)
    return masked_img

def get_face_data(img):
    height, width, _ = img.shape
    detector.setInputSize((width, height))
    faces = detector.detect(img)
    return faces[1][0] if faces[1] is not None else None

@app.route("/")
def index():
    return jsonify({"status": f"Face API is running on Aiven. Loaded {len(known_faces)} users."})

@app.route("/register", methods=["POST"])
def register():
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401
    if 'file' not in request.files or 'id' not in request.form:
        return jsonify({"error": "Missing data"}), 400
        
    student_id = str(request.form['id']).strip()
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    face = get_face_data(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400
        
    if student_id not in known_faces:
        known_faces[student_id] = []
        
    feature_normal = recognizer.feature(recognizer.alignCrop(img, face))
    feature_masked = recognizer.feature(recognizer.alignCrop(add_virtual_mask(img, face), face))
    
    save_face_to_db(student_id, feature_normal)
    save_face_to_db(student_id, feature_masked)
    
    known_faces[student_id].append(feature_normal)
    known_faces[student_id].append(feature_masked)
    
    return jsonify({"message": f"Successfully registered ID: {student_id} to Aiven Database"}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400
    if not known_faces:
        return jsonify({"error": "No faces registered"}), 400
        
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    face = get_face_data(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400
        
    feature = recognizer.feature(recognizer.alignCrop(img, face))
    best_match_id, best_score, THRESHOLD = None, 0.0, 0.34
    
    for uid, features_list in known_faces.items():
        for known_feature in features_list:
            score = recognizer.match(known_feature, feature, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score, best_match_id = score, uid
            
    if best_score >= THRESHOLD:
        return jsonify({"id": best_match_id}), 200
    return jsonify({"error": "Unknown face"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
