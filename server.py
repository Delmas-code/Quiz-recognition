from fastapi import FastAPI, UploadFile, HTTPException, File, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import threading
import json
from typing import Dict
import shutil
import os
import cv2
import base64
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from collections import defaultdict

app = FastAPI()

# Allow all origins for testing (for production, specify specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, but you can specify domains for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Dictionary to keep track of connected clients
connected_clients: Dict[str, WebSocket] = {}

# Global variables to track unknown user counts
unknown_user_count = defaultdict(int)
unknown_user_threshold = defaultdict(int)
stop_flags = defaultdict(bool)  # Default value for each user is False

# Lock to synchronize WebSocket and video processing
lock = threading.Lock()


# Directory to store user data
BASE_DIR = "user_data"
if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

@app.websocket("/ws/identify/")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for client communication.
    """
    await websocket.accept()
    username = None

    try:
        # Identify the client
        data = await websocket.receive_text()
        message = json.loads(data)
        if message.get("type") == "identification" and "username" in message:
            username = message["username"]
            if username in connected_clients:
                await websocket.send_json({"type": "used", "message": "Username already in use."})
                await websocket.close()
                return
            connected_clients[username] = websocket
            start_video_identification(username)
            await websocket.send_json({"type": "confirmation", "message": f"Welcome, {username}!"})
        else:
            await websocket.send_json({"type": "error", "message": "Invalid identiication message."})
            await websocket.close()
            return

        # Keep the connection alive
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if username and username in connected_clients:
            del connected_clients[username]
        print(f"User '{username}' disconnected.")


def start_video_identification(username):
    """
    Starts the video identification process.
    Runs in a separate thread to avoid blocking WebSocket communication.
    """
    unknown_user_count[username] = 0
    unknown_user_threshold[username] = 2
    stop_flags[username] = False  # Reset stop flag for the user
    identification_thread = threading.Thread(target=video_identification,args=(username,), daemon=True)
    identification_thread.start()

def video_identification(username):
    """
    Simulates video identification.
    Detects users via a camera feed, and sends alerts if unknown users are detected.
    """
    global unknown_user_count

    while not stop_flags[username]:
        # Simulated result from identification (replace with real identification logic)
        identified_user, identified, cam = perform_identification(username)  # Replace with actual logic
        with lock:
            # if identified_user == "unknown":
            if identified == False:
                unknown_user_count[username] += 1 
                print(f"Unknown user detected. Count: {unknown_user_count[username]}")
                if unknown_user_count[username] >= unknown_user_threshold[username]:
                    asyncio.run(alert_unknown_user())
                    unknown_user_count[username] = 0
                    stop_video_identification(username, cam)
                    break
                
            elif identified == None:
                print(f"Identified user: {identified_user}")
                asyncio.run(send_identification_to_client(identified))
            else:
                unknown_user_count[username] = 0
                print(f"Identified user: {identified_user}")
                asyncio.run(send_identification_to_client(identified))


def stop_video_identification(username,cam):
    """
    Stops the video identification process.
    """
    # Implementation depends on how video capture is handled.
    print("Video identification stopped.")
    cam.release()
    stop_flags[username] = True  # Set the stop flag for the user
    # global running
    # running = False

@app.post('/stop_identification/')
async def stop_identification(username: str = Form(...)):
    """
    API endpoint to stop the video identification process for a specific user.
    """
    stop_video_identification(username)
    return {"success": True, "message": f"Video identification stopped for {username}."}


def perform_identification(username):
    """
    Placeholder function for user identification.
    Replace with the actual logic using a pre-trained model.
    """
    # import random
    # users = ["user1", "user2", "unknown"]  # Example result set
    # return random.choice(users)
    
    detector = MTCNN()
    embedder = FaceNet()

    # Load all user embeddings
    embeddings = {}
    for user_name in os.listdir("user_data"):
        user_folder = os.path.join("user_data", user_name)
        embedding_path = os.path.join(user_folder, "embedding.npy")
        if os.path.exists(embedding_path):
            embeddings[user_name] = np.load(embedding_path)


    cam = cv2.VideoCapture(0)  # Open the camera
    running = True

    while running:
        ret, frame = cam.read()
        if not ret:
            print("Error capturing video frame.")
            break

        # Detect face
        faces = detector.detect_faces(frame)
        if faces:
            # Get the largest face
            faces.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
            x, y, w, h = faces[0]['box']
            face = frame[y:y + h, x:x + w]

            # Preprocess the face
            face_resized = cv2.resize(face, (160, 160))
            face_array = np.expand_dims(face_resized, axis=0)

            # Compute embedding for the input
            input_embedding = embedder.embeddings(face_array)[0]
            normalized_input = input_embedding / np.linalg.norm(input_embedding)

            # Compare with stored embeddings
            similarities = {
                user: np.linalg.norm(normalized_input - stored_emb)
                for user, stored_emb in embeddings.items()
            }

            # Identify the user with the lowest similarity score
            identified_user = min(similarities, key=similarities.get)
            confidence = similarities[identified_user]

            # Threshold for identification (tune this value)
            threshold = 0.8
            if confidence < threshold:
                print(f"Identified: {identified_user} (Confidence: {confidence:.2f})")
                if str(identified_user).lower() == str(username).lower():
                    # unknown_count = 0  # Reset the counter if a known user is detected
                    return identified_user, True, cam
            else:
                print("Unknown user")
                # unknown_count += 1
                return "Unknown user",False, cam
        else:
            print("No face detected.")
            # unknown_count += 1
            return None, None, cam

    # cam.release()


async def alert_unknown_user():
    """
    Sends an alert to all connected clients when unknown users are detected multiple times.
    """
    if connected_clients:
        alert_message = {"type": "alert", "message": "Unknown user detected multiple times!"}
        for websocket in connected_clients.values():
            try:
                await websocket.send_json(alert_message)
            except Exception as e:
                print(f"Error sending alert: {e}")


async def send_identification_to_client(username):
    """
    Sends the identified user's name to the connected clients.
    """
    if connected_clients:
        identification_message = {"type": "identified", "message": f"Identified user: {username}"}
        for websocket in connected_clients.values():
            try:
                await websocket.send_json(identification_message)
            except Exception as e:
                print(f"Error sending identification: {e}")


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str, close_after=False):
        for connection in self.active_connections[:]:  # Iterate over a copy since we may modify the list
            try:
                await connection.send_text(message)
                if close_after:
                    await connection.close()
                    self.disconnect(connection)
            except WebSocketDisconnect:
                self.disconnect(connection)


manager = ConnectionManager()

@app.websocket("/ws/store_snapshots/")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for sending real-time frame updates to the frontend.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # Optional: Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)



@app.post('/store_snapshots/')
async def setup_user(username: str= Form(...)):
    """
    Register a new user by capturing their face and computing an embedding.
    """
    print(f"Registering user: {username}")
    cam = cv2.VideoCapture(0)
    detector = MTCNN()
    embedder = FaceNet()  # Initialize FaceNet from keras-facenet

    # Create user folder
    user_folder = os.path.join("user_data", username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    embeddings = []
    count = 0
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error capturing image.")
                break

            # Detect face
            faces = detector.detect_faces(frame)
            if faces:
                # Get the largest face
                faces.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
                x, y, w, h = faces[0]['box']
                face = frame[y:y + h, x:x + w]

                # Preprocess face
                face_resized = cv2.resize(face, (160, 160))
                face_array = np.expand_dims(face_resized, axis=0)

                # Compute embedding
                embedding = embedder.embeddings(face_array)[0]
                normalized_embedding = embedding / np.linalg.norm(embedding)

                # Save the embedding
                embeddings.append(normalized_embedding)
                count += 1
                
                # Send frame to frontend
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                await manager.broadcast(frame_base64)


            # Stop after 10 captures
            if count >= 10:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.release()

        # Notify the frontend that the task is complete and close the socket
        await manager.broadcast("TASK_COMPLETE", close_after=True)

    # Average the embeddings
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        np.save(os.path.join(user_folder, "embedding.npy"), avg_embedding)
        print(f"User {username} registered successfully.")
        return {"success": True, "message": "Snapshot saved successfully."}
    else:
        print("No faces captured. Registration failed.")
        return {"success": False, "message": "No faces captured. Registration failed."}
    


"""
    Fetch all usernames.
"""
@app.get("/users/")
async def get_users():
    print('users')
    if not os.path.exists(BASE_DIR):
        # os.mkdir(BASE_DIR)
        return {"users": []}

    all_usernames = os.listdir(BASE_DIR)
    return {"users": all_usernames}



@app.post('/old/register/')
async def register_user(username: str= Form(...), profile_picture: UploadFile = File(...)):
    print('register')
    user_dir = os.path.join(BASE_DIR, username)
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    
    if not os.path.exists(user_dir):
        os.mkdir(user_dir)
    try:
        profile_path = os.path.join(user_dir, "profile.jpg")
        with open(profile_path, 'wb') as f:
            shutil.copyfileobj(profile_picture.file, f)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving file.")
    
    return {"message": f"User {username} registered successfully!"}

@app.post('/register/')
async def register_user(username: str= Form(...)):
    print('register')
    user_dir = os.path.join(BASE_DIR, username)
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    
    if not os.path.exists(user_dir):
        os.mkdir(user_dir)
    
    return {"message": f"User {username} registered successfully!"}


@app.post('/old/register/')
async def register_user(username: str= Form(...), profile_picture: UploadFile = File(...)):
    print('register')
    user_dir = os.path.join(BASE_DIR, username)
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    
    if not os.path.exists(user_dir):
        os.mkdir(user_dir)
    try:
        profile_path = os.path.join(user_dir, "profile.jpg")
        with open(profile_path, 'wb') as f:
            shutil.copyfileobj(profile_picture.file, f)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving file.")
    
    return {"message": f"User {username} registered successfully!"}

