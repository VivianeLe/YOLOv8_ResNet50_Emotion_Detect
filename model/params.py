from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

batch_size = 64
image_size = 224
target_size=(image_size, image_size)
num_classes = 7  # Số class trong bộ dữ liệu FER2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
yolo_path = 'pretrained/yolov8n-face-lindevs.pt'
resnet_path = 'pretrained/best_resnet50_model_statedict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình YOLOv8 đã huấn luyện trước
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Đảm bảo ảnh có 3 kênh
    transforms.Resize(target_size),               # Resize ảnh
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa
])

face_detector = YOLO(yolo_path)  # YOLOv8 dành riêng cho phát hiện khuôn mặt

def load_classifier(model_path):
    resnet50 = models.resnet50(pretrained=False)  # Không tải trọng số ImageNet
    resnet50.fc = nn.Sequential(
        nn.Linear(resnet50.fc.in_features, 2048),
        nn.ReLU(),
        nn.Dropout(0.7),
        nn.Linear(2048, num_classes) 
    )

    # Load trọng số đã train
    resnet50.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Đưa model về chế độ đánh giá
    resnet50.eval()
    resnet50 = resnet50.to(device)
    return resnet50

# Load mô hình ResNet50 đã huấn luyện trước
resnet50 = load_classifier(resnet_path)