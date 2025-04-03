import streamlit as st
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize the YOLO model
model = YOLO("yolov8n.pt")
api_key=os.getenv('HF'),
# Initialize the Hugging Face Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv('HF'),
)

def detect_faces(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load image.")

    results = model(img)
    detections = results[0].boxes
    highest_confidence = 0
    best_face = None

    for box in detections:
        x1, y1, x2, y2, confidence = box.xyxy[0].tolist() + [box.conf[0].item()]
        if confidence > 0.5:  # Confidence threshold
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_face = img[int(y1):int(y2), int(x1):int(x2)]

    if best_face is not None:
        return best_face, highest_confidence
    else:
        return None, 0

def classify_image(image):
    # Convert the image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    # Use the Hugging Face Inference API to classify the image
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/resnet-50",
            headers=headers,
            files={"file": img_bytes}
        )
        
        if response.status_code == 200:
            return response.json()  # Return the classification results
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def display_image(image, caption="Image"):
    st.image(image, caption=caption, use_container_width=True)

# Streamlit App
st.title("Face Detection and Classification App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image_data = uploaded_file.read()
    
    # Detect faces
    best_face, highest_confidence = detect_faces(image_data)
    if best_face is not None:
        st.success(f"Face detected with confidence: {highest_confidence:.2f}")
        # Classify the best detected face
        classification_result = classify_image(best_face)
        st.write("Classification Results:")
        st.json(classification_result)  # Display results in JSON format
        # Display the best detected face
        display_image(best_face, caption="Detected Face")
    else:
        st.warning("No faces detected with sufficient confidence.")