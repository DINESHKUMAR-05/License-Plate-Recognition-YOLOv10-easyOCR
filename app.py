import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image

model = YOLO(r"detect/train/weights/best.pt")  
reader = easyocr.Reader(['en'])

def correct_image_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        median_angle = 90
        
        center = tuple(np.array(image.shape[1::-1]) / 2)  # Calculate the center of the image
        
        M = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
        
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        
        bound_w = int(image.shape[1] * abs_cos + image.shape[0] * abs_sin)
        bound_h = int(image.shape[1] * abs_sin + image.shape[0] * abs_cos)
        
        M[0, 2] += bound_w / 2 - center[0]
        M[1, 2] += bound_h / 2 - center[1]
        
        rotated_image = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC)
        
        return rotated_image
    
    return image

def process_image(image, apply_rotation=True):
    if apply_rotation:
        image = correct_image_rotation(image)
    results = model(image)

    license_plate_label = "license-plate"  
    confidence_threshold = 0.5  

    if license_plate_label in model.names.values():
        label_index = list(model.names.values()).index(license_plate_label)
    else:
        st.error(f"Label '{license_plate_label}' not found in the model's classes.")
        label_index = None

    if label_index is not None:
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidences = result.boxes.conf
            for i, cls in enumerate(classes):
                if int(cls) == label_index and confidences[i] > confidence_threshold:
                    x1, y1, x2, y2 = map(int, boxes[i])
                    license_plate_image = image[y1:y2, x1:x2]
                    ocr_result = reader.readtext(license_plate_image)
                    
                    if ocr_result:
                        license_plate_number = ocr_result[0][-2]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
                        cv2.putText(image, f"{license_plate_number.upper()}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 
                                    (255, 0, 0), 3)

    return image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_image(frame, apply_rotation=False)
        frame_placeholder.image(processed_frame, channels="BGR")
    
    cap.release()

def process_live_feed():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop = False

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image.")
            break
        
        processed_frame = process_image(frame, apply_rotation=False)
        frame_placeholder.image(processed_frame, channels="BGR")
        

    cap.release()

st.title("License Plate Detection and Recognition")


option = st.sidebar.selectbox("Choose Input Source", ("Image", "Video", "Live Feed"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        processed_image = process_image(image)
        st.image(processed_image, caption='Processed Image with License Plate Detection')

elif option == "Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        video_bytes = video_file.read()
        video_path = f"temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        process_video(video_path)

elif option == "Live Feed":
    process_live_feed()
