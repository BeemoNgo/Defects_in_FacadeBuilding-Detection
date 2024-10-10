import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLO model function
def load_yolo_model(model_choice):
    if model_choice == 'YOLOv8':
        model = YOLO(r'C:\Users\280314\OneDrive - Swinburne University\AI in Enginerring\user_inteface\Model\yolov8m.pt')
    # elif model_choice == 'YOLOv9':
    #     model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_yolov9_weights')
    elif model_choice == 'YOLOv10':
        model = YOLO(r'C:\Users\280314\OneDrive - Swinburne University\AI in Enginerring\user_inteface\Model\yolov10s.pt')
    else:
        st.error('Unknown model choice')
        return None
    return model

# Image detection function
def detect_objects(model, image):
    results = model(image)
    return results

# Streamlit UI
st.title("YOLO Object Detection App")

# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Model selection
model_choice = st.selectbox("Choose YOLO Model", ["YOLOv8", "YOLOv9", "YOLOv10"])

# If an image is uploaded and a model is chosen
if uploaded_image and model_choice:
    # Open the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load the YOLO model based on user choice
    model = load_yolo_model(model_choice)
    
    if model is not None:
        # Convert the image to a format YOLO can process
        img_array = np.array(image)
        
        # Detect objects
        with st.spinner('Running object detection...'):
            results = detect_objects(model, img_array)
        
        # Display detection results
        st.success("Detection complete!")
        
        # Show the detected image with bounding boxes
        detected_img = results[0].plot()  # YOLOv8 renders the image with bounding boxes using `.plot()`
        st.image(detected_img, caption="Detection Results", use_column_width=True)
