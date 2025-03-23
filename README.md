import sys
import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import time
import pickle
import os
import numpy as np
import csv
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import easyocr
import geopy.distance
import geopy.geocoders
from geopy.geocoders import Nominatim
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer

# Initialize text-to-speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load object detection model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
COCO_LABELS = {i: label for i, label in enumerate(
    ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "watch", "pothole", "manhole"])}

transform = transforms.Compose([transforms.ToTensor()])

# Currency recognition setup
reader = easyocr.Reader(['en'])

def recognize_currency(image):
    results = reader.readtext(image)
    total_amount = 0
    for (_, text, _) in results:
        text = text.replace("Rs", "").strip()
        if text.isdigit():
            total_amount += int(text)
    return total_amount

# Navigation Assistance
def get_distance(coord1, coord2):
    return geopy.distance.geodesic(coord1, coord2).meters

def get_location():
    response = requests.get("http://ip-api.com/json/")
    data = response.json()
    return (data['lat'], data['lon'])

class VisionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.camera_on = False
        self.cap = None

    def initUI(self):
        self.setWindowTitle("AI Vision Assistant")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()
        self.btn_toggle = QPushButton("Turn On Camera", self)
        self.btn_toggle.clicked.connect(self.toggle_camera)
        self.layout.addWidget(self.btn_toggle)
        self.setLayout(self.layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.btn_toggle.setText("Turn On Camera")
        else:
            self.camera_on = True
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.btn_toggle.setText("Turn Off Camera")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        image = Image.fromarray(rgb_frame)
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            predictions = model(image_tensor)
        detected_objects = [COCO_LABELS.get(label.item(), "Unknown Object") for label, score in zip(predictions[0]['labels'], predictions[0]['scores']) if score > 0.3]
        
        if not detected_objects and not face_encodings:
            speak("No objects detected. Please adjust the camera.")
        elif "pothole" in detected_objects or "manhole" in detected_objects:
            speak("Warning! Pothole or manhole detected ahead. Please be cautious.")
        elif detected_objects:
            speak(f"Objects detected: {', '.join(detected_objects)}")
        
        amount = recognize_currency(frame)
        if amount > 0:
            speak(f"Total detected amount is {amount} rupees")
        
        for face_encoding in face_encodings:
            name = "Unknown"
            speak(f"Recognized person: {name}")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionApp()
    window.show()
    sys.exit(app.exec_())
