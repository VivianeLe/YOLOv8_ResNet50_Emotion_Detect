# import time
import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from utils.params import emotion_labels, resnet_path, yolo_path, device
import streamlit as st

class InferencePipeline:

    def __init__(self):
        """
        Initialize the pipeline with required models and transformations.
        """
        # Load YOLO face detection model
        self.face_detector = YOLO(yolo_path)

        # Load ResNet50 emotion classification model
        self.resnet50 = self._load_classifier(resnet_path)

        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_classifier(self, model_path):
        """
        Load the pretrained ResNet50 model for emotion detection.
        """
        from torchvision import models
        import torch.nn as nn

        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(2048, len(emotion_labels))
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        return model

    def detect_faces(self, image):
        """
        Detect faces in the input image.
        
        :param image: Input image (numpy array)
        :return: List of tuples (face_image, bounding_box)
        """
        results = self.face_detector.predict(source=image, conf=0.5)
        faces = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]
                faces.append((face, (x1, y1, x2, y2)))
        return faces

    def predict_emotion(self, face):
        """
        Predict the emotion from a face image.
        
        :param face: Cropped face image (numpy array)
        :return: Emotion label
        """
        try:
            if face is not None and face.size > 0 and len(face.shape) == 3:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = self.transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = self.resnet50(face_tensor)
                    _, predicted = outputs.max(1)
                    return emotion_labels[predicted.item()]
            else:
                return "Invalid Face"
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "Error"

    def detect_predict_pipeline(self, photo):
        faces = self.detect_faces(photo)

        if not faces:
            return photo

        for face, (x1, y1, x2, y2) in faces:
            emotion_label = self.predict_emotion(face)
            cv2.rectangle(photo, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(photo, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        return photo

    def process_image(self, image_path):
        """
        Process an image to detect faces and emotions.
        
        :param image_path: Path to the input image
        :return: Image with bounding boxes and emotion labels
        """
        photo = cv2.imread(image_path)
        if photo is None:
            raise ValueError(f"Cannot read image from path: {image_path}")

        photo = self.detect_predict_pipeline(photo)
        return photo

    def process_video(self, source=0):
        """
        Process a video to detect faces and emotions in real-time.
        
        :param source: Video source (file path or camera index)
        """
        stframe = st.empty()
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Cannot open video source.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = self.detect_predict_pipeline(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        stframe.empty()
        st.write("Video processing complete.")
        # cv2.destroyAllWindows()

    def run(self, input_type, input_path=None, camera=0):
        """
        Run the pipeline for the given input type.
        
        :param input_type: 'image', 'video', or 'realtime'
        :param input_path: Path to the input image or video (if applicable)
        """
        if input_type == 'image' and input_path:
            return self.process_image(input_path)
        elif input_type == 'video' and input_path:
            self.process_video(input_path)
        elif input_type == 'realtime':
            self.process_video(camera)
        else:
            raise ValueError("Invalid input type or missing input path.")
