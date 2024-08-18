# License Plate Detection and Recognition System

## Overview

This project aims to develop an automated system for detecting and recognizing license plates from images, videos, and live video feeds. The system leverages state-of-the-art computer vision technologies, specifically YOLOv10n (Nano Architecture used for faster inferencing) for object detection and EasyOCR for optical character recognition (OCR). 

## Features

- **License Plate Detection**: Detects license plates using YOLOv10.
- **Optical Character Recognition (OCR)**: Extracts text from detected license plates using EasyOCR.
- **Image Processing**: Corrects image orientation and extracts license plate regions.
- **Video Processing**: Processes video files to detect and recognize license plates frame by frame.
- **Live Feed Processing**: Captures and processes real-time video feeds from a webcam.

## Technologies Used

- **YOLOv10 (You Only Look Once)**: A deep learning model for real-time object detection.
- **EasyOCR**: An OCR library that enables text extraction from images.
- **OpenCV**: A library for image processing and computer vision tasks.
- **Streamlit**: A framework for building interactive web applications.

## Evaluation Metrics

The YOLOv10 model used in this project is the nano architecture. The evaluation metrics for this model are as follows:

- **Precision**: 0.911
- **Recall**: 0.731
- **mAP50** (Mean Average Precision at IoU threshold 0.5): 0.852
- **mAP50-95** (Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95): 0.635

## Installation

To set up the project, clone the repository, you'll need to install the required dependencies. You can use `pip` to install them:

```bash
pip install requirements.txt
```

## Usage

### Running the Application

To run the Streamlit application, navigate to the project directory and use the following command:

```bash
streamlit run app.py
```

### Input Sources

1. **Image**: Upload an image file (`.jpg`, `.jpeg`, `.png`) to perform license plate detection and recognition.
2. **Video**: Upload a video file (`.mp4`, `.avi`, `.mov`) to process each frame for license plate detection and recognition.
3. **Live Feed**: Stream video from a webcam to detect and recognize license plates in real time.

### Example

Here is an example of how the application works with an uploaded image:

1. **Upload**: Choose an image file from your local machine.
2. **Processing**: The application detects and recognizes the license plate.
3. **Results**: The processed image with detected license plate(s) and recognized text is displayed.

### Output Images

![LP-output](https://github.com/user-attachments/assets/10e746a0-6fb6-4fad-b5c3-a6bbd11823d9)

## Code

The core functionality of the application is implemented in the `app.py` file. Key components include:

- **Image Rotation Correction**: Ensures images are correctly oriented before processing.
- **Object Detection**: Utilizes YOLOv10 for detecting license plates in images and video frames.
- **OCR Processing**: Uses EasyOCR to extract text from detected license plates.
- **Streamlit Integration**: Provides an interactive web interface for users to upload images/videos and view results.
