import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes

# Function to load the Faster R-CNN model
def load_faster_rcnn_model():
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)  # No pre-trained weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 5  # Update the number of classes to 5
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load("./Model/faster_rcnn.pth", map_location=torch.device('cpu'))
    
    # Load the state dict (ignoring predictor layers if needed)
    state_dict = {k: v for k, v in state_dict.items() if not (
        k.startswith('roi_heads.box_predictor.cls_score') or k.startswith('roi_heads.box_predictor.bbox_pred')
    )}
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
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
def detect_faster_rcnn(model, image, classes, device):
    transform = transforms.Compose([
        transforms.Resize((600, 600)),  # Resize but no normalization
        transforms.ToTensor(),  # Convert image to tensor
    ])
    
    # Transform the image to a tensor and move to the correct device
    image_tensor = transform(image).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]  # Get predictions

    # Filter boxes based on score
    scores = prediction['scores']
    boxes = prediction['boxes'][scores > 0.3]  # Set your threshold here
    labels = prediction['labels'][scores > 0.3].tolist()
    filtered_scores = scores[scores > 0.3].tolist()
    labels_with_scores = [f"{classes[i]} {score:.2f}" for i, score in zip(labels, filtered_scores)]

    # Convert the image back to its original scale for displaying
    img_int = (image_tensor[0] * 255).byte()

    # Draw bounding boxes with the corresponding labels
    result_image = draw_bounding_boxes(
        img_int, 
        boxes=boxes, 
        labels=labels_with_scores, 
        width=4
    )

    # Convert the result image to a format that Streamlit can display
    result_image_np = result_image.permute(1, 2, 0).cpu().numpy()  # HWC format
    return result_image_np

def detect_yolo(model, image):
    results = model(image)
    return results

# List of class names for dataset
classes = ["corrosion", "paint_defect", "crack", "dirt_mold", "delamination"]
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    else:
        model = load_yolo_model(model_choice)

    if model is not None:
        # Convert the image to a format the model can process
        img_array = np.array(image)

        # Detect objects
        with st.spinner('Running object detection...'):
            if model_choice == "Faster_RCNN":
                result_image = detect_faster_rcnn(model, image, classes, device)
                # Display the processed image with bounding boxes
                st.image(result_image, caption="Faster R-CNN Detection Results", use_column_width=True)
            else:
                results = detect_yolo(model, np.array(image))
                detected_img = results[0].plot()  # YOLO results
                st.image(detected_img, caption="YOLO Detection Results", use_column_width=True)