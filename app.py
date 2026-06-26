import os
import cv2
import numpy as np
import urllib.request
import requests
import binascii
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# 🔒 secrets — ใส่ใน Render Environment Variables
# ---------------------------------------------------------
API_KEY       = os.environ.get("API_KEY",       "my_secret_key_12345")
RENDER_SECRET = os.environ.get("RENDER_SECRET", "REPLACE_WITH_LONG_RANDOM_SECRET_64CHARS")
PHP_API_URL   = os.environ.get("PHP_API_URL",   "https://dkt.gt.tc/face_vectors_api")

# ---------------------------------------------------------
# โหลดโมเดล
# ---------------------------------------------------------
YUNET_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
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

detector   = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

# ---------------------------------------------------------
# Helper headers สำหรับเรียก PHP API
# ---------------------------------------------------------
def php_headers():
    return {
        "Authorization": f"Bearer {RENDER_SECRET}",
        "User-Agent":    "python-requests/face-api",
        "Content-Type":  "application/json",
    }

# ---------------------------------------------------------
# แปลง numpy ↔ hex
# ---------------------------------------------------------
def vec_to_hex(arr):
    return binascii.hexlify(arr.astype(np.float32).tobytes()).decode('ascii')

def hex_to_vec(h):
    return np.frombuffer(binascii.unhexlify(h), dtype=np.float32)

# ---------------------------------------------------------
# known_faces cache (in-memory)
# ---------------------------------------------------------
known_faces = {}

def load_from_db():
    global known_faces
    try:
        resp = requests.get(PHP_API_URL, headers=php_headers(), timeout=30)
        if resp.status_code != 200:
            print(f"[WARN] PHP API {resp.status_code}: {resp.text[:200]}")
            return
        data = resp.json().get("data", {})
        known_faces = {uid: [hex_to_vec(h) for h in entry["vectors"]] for uid, entry in data.items()}
        total = sum(len(v) for v in known_faces.values())
        print(f"[INFO] Loaded {total} vectors for {len(known_faces)} users from MySQL")
    except Exception as e:
        print(f"[ERROR] load_from_db: {e}")

def save_to_db(user_id, role_type, vectors):
    payload = {"user_id": user_id, "role_type": role_type, "vectors": [vec_to_hex(v) for v in vectors]}
    resp = requests.post(PHP_API_URL, json=payload, headers=php_headers(), timeout=30)
    return resp.status_code == 200, resp.json()

# โหลดตอน startup
load_from_db()

# ---------------------------------------------------------
def require_api_key(req):
    key = req.headers.get('x-api-key') or req.form.get('api_key')
    return key == API_KEY

def add_virtual_mask(img, face_data):
    masked = img.copy()
    x, y, w, h = int(face_data[0]), int(face_data[1]), int(face_data[2]), int(face_data[3])
    nose_y = int(face_data[9])
    cv2.rectangle(masked, (max(0,x), max(0, nose_y - int(h*0.05))),
                  (min(img.shape[1], x+w), min(img.shape[0], y+h)), (255,255,255), -1)
    return masked

def get_face(img):
    detector.setInputSize((img.shape[1], img.shape[0]))
    faces = detector.detect(img)
    return None if faces[1] is None else faces[1][0]

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return jsonify({"status": "running", "users": len(known_faces),
                    "vectors": sum(len(v) for v in known_faces.values())})

@app.route("/reload", methods=["POST"])
def reload_route():
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401
    load_from_db()
    return jsonify({"status": "reloaded", "users": len(known_faces)})

@app.route("/register", methods=["POST"])
def register():
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401
    if 'file' not in request.files or 'id' not in request.form:
        return jsonify({"error": "Missing file or id"}), 400

    uid       = str(request.form['id']).strip()
    role_type = str(request.form.get('role', 'student')).strip().lower()
    if role_type not in ('student', 'teacher'):
        role_type = 'student'

    nparr = np.frombuffer(request.files['file'].read(), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    face = get_face(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400

    f_normal = recognizer.feature(recognizer.alignCrop(img, face))
    f_masked = recognizer.feature(recognizer.alignCrop(add_virtual_mask(img, face), face))
    vectors  = [f_normal, f_masked]

    ok, api_resp = save_to_db(uid, role_type, vectors)
    if not ok:
        return jsonify({"error": f"DB save failed: {api_resp}"}), 500

    known_faces[uid] = vectors
    return jsonify({"message": f"Successfully registered ID: {uid} (Normal & Masked saved)",
                    "saved_to": "MySQL via PHP API"}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    if not require_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400
    if not known_faces:
        return jsonify({"error": "No faces registered"}), 400

    nparr = np.frombuffer(request.files['file'].read(), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    face = get_face(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400

    feat = recognizer.feature(recognizer.alignCrop(img, face))

    best_id, best_score = None, 0.0
    THRESHOLD = 0.28

    for uid, vecs in known_faces.items():
        for v in vecs:
            s = recognizer.match(v, feat, cv2.FaceRecognizerSF_FR_COSINE)
            if s > best_score:
                best_score, best_id = s, uid

    score = round(float(best_score), 4)
    if best_score >= THRESHOLD:
        return jsonify({"id": best_id, "score": score}), 200
    return jsonify({"error": "Unknown face", "score": score}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
