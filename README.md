# Face Recognition API

This is a lightweight and robust Face Recognition API built with Python, Flask, and OpenCV's deep learning face models (YuNet and SFace). It's specially designed to run smoothly on free-tier hosting platforms like [Render.com](https://render.com) without running into memory or compilation issues.

## How to use on Render
1. Upload this entire unzipped folder to a new repository on your GitHub.
2. Log in to [Render.com](https://render.com) and click **New+** -> **Web Service**.
3. Connect the GitHub repository you just created.
4. Render will automatically detect the Python environment and install everything. Wait for the deployment to finish.
5. Your API URL will look something like `https://face-recognition-api-xxxx.onrender.com`.

## API Endpoints

### 1. Register a Person (`POST /register`)
Upload a picture along with the person's ID (e.g., student ID `17099`).
- **Form Data:**
  - `id`: The ID number (text)
  - `file`: The image file (image/jpeg or image/png)
- **Example Response:** `{"message": "Successfully registered ID: 17099"}`

### 2. Recognize a Person (`POST /recognize`)
Upload a picture to check who it is. If the person matches the database, it returns their ID.
- **Form Data:**
  - `file`: The image file
- **Example Response:** `{"id": "17099"}` (if it matches) OR `{"error": "Unknown face"}` (if no match).

## Important Note
The face data is saved in a file named `known_faces.pkl`. In a real production environment with persistent storage, you should consider saving these embeddings to a database like PostgreSQL or a Cloud Storage bucket, because Render's free web services automatically restart and lose local files periodically.
