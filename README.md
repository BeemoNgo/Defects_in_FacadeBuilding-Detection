# Object Detection App

This project is an object detection application built using **Streamlit**, **PyTorch**, **Ultralytics YOLO**, and **OpenCV**. It allows users to upload an image and select from different object detection models such as **YOLOv10** or **Faster R-CNN**. The app will display the uploaded image and apply object detection, outputting the result with bounding boxes and confidence scores.

## Features

- **Faster R-CNN** model using MobileNet V3 backbone for object detection.
- **YOLOv10** and **YOLOv11** models in small, medium, and large configurations for real-time object detection.
- Supports image uploads in JPG, PNG, and JPEG formats.
- Displays detected objects with bounding boxes and labels, along with confidence scores.
- Responsive and easy-to-use web interface powered by Streamlit.

## Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Streamlit
- PyTorch
- Ultralytics YOLO
- OpenCV
- Torchvision

You can install all dependencies using the following command:

```bash
pip install -r requirements.txt

## Model Selection

The app provides an option to choose between several object detection models:

- **Faster R-CNN**: A powerful, region-based convolutional neural network for object detection.
- **YOLOv10/YOLOv11**: YOLO models are fast, real-time object detection models with different size configurations (Small, Medium, Large).

## How to Use

- **Upload Image**: You can upload an image in JPG, PNG, or JPEG format.
- **Select a Model**: Choose a model for object detection (YOLOv10, YOLOv11, or Faster R-CNN).
- **View Results**: The app will display the uploaded image with bounding boxes drawn around detected objects.

## Example Workflow

1. **Upload Image**: You can upload an image in JPG, PNG, or JPEG format.
2. **Select a Model**: Choose a model for object detection (YOLOv10, YOLOv11, or Faster R-CNN).
3. **View Results**: The app will process and display the results with bounding boxes and confidence scores.