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
undetected_face = defaultdict(int)
stop_flags = defaultdict(bool)  # Default value for each user is False
embeddings = {}


# Directory to store user data
BASE_DIR = "user_data"
if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)


@app.post('/identify/')
async def perform_identification(username: str= Form(...), image: UploadFile = File(...)):
    """
    Identify the user based on an uploaded image.
    """
    # unknown_user_count[username] = 0
    # unknown_user_threshold[username] = 2
    
    detector = MTCNN()
    embedder = FaceNet()

    # Load all user embeddings
    # embeddings = {}
    for user_name in os.listdir("user_data"):
        user_folder = os.path.join("user_data", user_name)
        embedding_path = os.path.join(user_folder, "embedding.npy")
        if os.path.exists(embedding_path):
            embeddings[user_name] = np.load(embedding_path)

    # Read the uploaded image
    img_data = await image.read()
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Error processing image.")

    # Detect face in the uploaded image
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
            user: float(np.linalg.norm(normalized_input - stored_emb))
            for user, stored_emb in embeddings.items()
        }

        # Identify the user with the lowest similarity score
        identified_user = min(similarities, key=similarities.get)
        confidence = similarities[identified_user]

        # Threshold for identification (tune this value)
        threshold = 0.8
        if (confidence < threshold) and identified_user ==  username:
            print(f"Identified: {identified_user}\nConfidence: {confidence}, Identified: {True}")
            return {"message": f"Identified: {identified_user}", "confidence": confidence, "identified": True}
        else:
            print(f"Unknown user\nConfidence: {confidence}, Identified: {False}")
            return {"message": "Unknown user", "confidence": confidence, "identified": False}
    else:
        print(f"[{username}]: No face detected in the uploaded image for this user.")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

@app.post('/store_snapshots/')
async def setup_user(username: str = Form(...), images: list[UploadFile] = File(...)):
    """
    Register a new user by capturing their face and computing a single embedding
    from multiple uploaded images by averaging their embeddings.
    """
    print(f"Registering user: {username}")

    # Create user folder
    user_folder = os.path.join("user_data", username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    detector = MTCNN()
    embedder = FaceNet()  # Initialize FaceNet from keras-facenet

    # Initialize a list to store embeddings for aggregation
    embeddings_list = []

    # Iterate over each uploaded image
    for image in images:
        print(f"Processing image: {image.filename}")

        # Save the uploaded image
        img_data = await image.read()
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is None:
            undetected_face[username] +=1
            if undetected_face[username] > 3:
                raise HTTPException(status_code=400, detail="Error processing image.")
            return
        
        # Detect face in the image
        faces = detector.detect_faces(img)
        if not faces:
            print(img)
            undetected_face[username] += 1
            if undetected_face[username] > 3:
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": "No face detected in the image.",
                        "image_info": f"Size: {img.size}, Format: {img.format}",
                    }
                )
            return

        # Get the largest face
        faces.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
        x, y, w, h = faces[0]['box']
        face = img[y:y + h, x:x + w]

        # Preprocess the face
        face_resized = cv2.resize(face, (160, 160))
        face_array = np.expand_dims(face_resized, axis=0)

        # Compute embedding
        embedding = embedder.embeddings(face_array)[0]
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Append the embedding to the list
        embeddings_list.append(normalized_embedding)

    # Average the embeddings to get a single representation
    if embeddings_list:
        final_embedding = np.mean(embeddings_list, axis=0)
    else:
        raise HTTPException(status_code=400, detail="No valid faces found in the images.")

    # Save the final aggregated embedding
    np.save(os.path.join(user_folder, "embedding.npy"), final_embedding)
    print(f"User {username} registered successfully with an aggregated embedding.")
    
    return {"success": True, "message": "Snapshots saved successfully."}


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

