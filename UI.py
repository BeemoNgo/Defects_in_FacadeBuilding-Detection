import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Function to load the Faster R-CNN model
def load_faster_rcnn_model():
    # Load the Faster R-CNN model architecture
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)  # No pre-trained weights

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Update the number of classes to 5
    num_classes = 5  

    # Reinitialize the box predictor head with the correct number of classes
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load the saved weights for your model
    state_dict = torch.load("./Model/faster_rcnn.pth", map_location=torch.device('cpu'))
    
    # Filter out the layers that do not match the current model architecture
    state_dict = {k: v for k, v in state_dict.items() if not (
        k.startswith('roi_heads.box_predictor.cls_score') or k.startswith('roi_heads.box_predictor.bbox_pred')
    )}

    # Load the state dict (ignoring the predictor layers)
    model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    model.eval()

    return model

# Load YOLO model function
def load_yolo_model(model_choice):
    if model_choice == 'YOLOv10S':
        model = YOLO("./Model/Yolov10s.pt")
    elif model_choice == 'YOLOv10M':
        model = YOLO("./Model/Yolov10m.pt")
    elif model_choice == 'YOLOv10L':
        model = YOLO("./Model/Yolov10l.pt")
    elif model_choice == 'YOLOv11S':
        model = YOLO("./Model/Yolov11s.pt")
    elif model_choice == 'YOLOv11M':
        model = YOLO("./Model/Yolov11m.pt")
    elif model_choice == 'YOLOv11L':
        model = YOLO("./Model/Yolov11l.pt")
    else:
        st.error('Unknown model choice')
        return None
    return model

# Faster R-CNN detection function
def detect_faster_rcnn(model, image):
    # Apply transformations: Resize to 600x600, Normalize, and convert to Tensor
    transform = A.Compose([
        A.Resize(600, 600),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard normalization
        ToTensorV2()
    ])

    # Transform the image and add a batch dimension
    transformed = transform(image=np.array(image))  # Convert PIL image to numpy array
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension for inference

    # Perform inference using Faster R-CNN
    with torch.no_grad():
        predictions = model(image_tensor)[0]  # Get the predictions

    return predictions

# YOLO detection function
def detect_yolo(model, image):
    results = model(image)
    return results

# Streamlit UI
st.title("Object Detection App")

# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Model selection
model_choice = st.selectbox("Choose Model", ["YOLOv10S", "YOLOv10M", "YOLOv10L", "YOLOv11S", "YOLOv11M", "YOLOv11L", "Faster_RCNN"])

# If an image is uploaded and a model is chosen
if uploaded_image and model_choice:
    # Open the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model based on user choice
    if model_choice == "Faster_RCNN":
        model = load_faster_rcnn_model()
    else:
        model = load_yolo_model(model_choice)

    if model is not None:
        # Convert the image to a format the model can process
        img_array = np.array(image)

        # Detect objects
        with st.spinner('Running object detection...'):
            if model_choice == "Faster_RCNN":
                predictions = detect_faster_rcnn(model, image)
                # Post-process predictions for Faster R-CNN
                boxes = predictions['boxes'].cpu().numpy().astype(int)
                labels = predictions['labels'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()

                # Draw bounding boxes on the image
                draw_image = img_array.copy()
                for box, label, score in zip(boxes, labels, scores):
                    if score > 0.5:  # Confidence threshold
                        cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(draw_image, f"Label {label}: {score:.2f}", (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Display the results with bounding boxes
                st.image(draw_image, caption="Faster R-CNN Detection Results", use_column_width=True)

            else:
                results = detect_yolo(model, img_array)
                # YOLO renders the image with bounding boxes using `.plot()`
                detected_img = results[0].plot()
                st.image(detected_img, caption="YOLO Detection Results", use_column_width=True)
