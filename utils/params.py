import torch


# batch_size = 64
# image_size = 224
# target_size=(image_size, image_size)
# num_classes = 7 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
yolo_path = 'model/yolov8n-face-lindevs.pt'
resnet_path = 'model/best_resnet50_model_statedict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")