# EMOTION DETECTION WITH YOLOv8 AND RESNET50 
This framework can detect face emotion with validation accuracy at 70% 
### Requirements
- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- YOLOv8
- RESNET50
- Emotion Detection Dataset

### Installation
```bash
pip install -r requirements.txt
```
### Usage
1. Without ngrok 
```bash
streamlit run main.py --server.port 8501 --server.address 0.0.0
```
2. With ngrok
```bash
python app.py
```
