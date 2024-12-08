from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

batch_size = 64
image_size = 224
target_size=(image_size, image_size)
num_classes = 7 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
yolo_path = 'pretrained/yolov8n-face-lindevs.pt'
resnet_path = 'pretrained/best_resnet50_model_statedict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 3 Channels
    transforms.Resize(target_size),               # Resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

face_detector = YOLO(yolo_path) 

def load_classifier(model_path):
    resnet50 = models.resnet50(pretrained=False)  # Don't load ImageNet weight
    resnet50.fc = nn.Sequential(
        nn.Linear(resnet50.fc.in_features, 2048),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(2048, num_classes) 
    )

    # Load trained weight
    resnet50.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    resnet50.eval()
    resnet50 = resnet50.to(device)
    return resnet50

# Load pretrained resnet50 for FER2013
resnet50 = load_classifier(resnet_path)

from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

batch_size = 64
image_size = 224
target_size=(image_size, image_size)
num_classes = 7 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
yolo_path = 'pretrained/yolov8n-face-lindevs.pt'
resnet_path = 'pretrained/best_resnet50_model_statedict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 3 Channels
    transforms.Resize(target_size),               # Resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_detector = YOLO(yolo_path) 

def load_classifier(model_path):
    resnet50 = models.resnet50(pretrained=False)  # Don't load ImageNet weight
    resnet50.fc = nn.Sequential(
        nn.Linear(resnet50.fc.in_features, 2048),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(2048, num_classes) 
    )

    # Load trained weight
    resnet50.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    resnet50.eval()
    resnet50 = resnet50.to(device)
    return resnet50

# Load pretrained resnet50 for FER2013
resnet50 = load_classifier(resnet_path)