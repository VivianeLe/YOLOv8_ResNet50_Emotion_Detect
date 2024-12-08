import cv2
from utils.params import *
from PIL import Image
import torch
import streamlit as st

# Hàm detect khuôn mặt
# def detect_faces(frame):
#     results = face_detector.predict(source=frame, conf=0.5)
#     faces = []
#     for result in results:
#         for box in result.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box)
#             face = frame[y1:y2, x1:x2]
#             faces.append((face, (x1, y1, x2, y2)))
#     return faces

# # Hàm phân loại cảm xúc
# def predict_emotion(face):
#     resized_face = cv2.resize(face, target_size)
#     resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
#     normalized_face = resized_face / 255.0
#     reshaped_face = np.expand_dims(normalized_face, axis=0)
#     emotion_prediction = emotion_model.predict(reshaped_face)
#     return emotion_labels[np.argmax(emotion_prediction)]

# Use pytorch
def detect_faces(photo):
    results = face_detector.predict(source=photo, conf=0.5)  # Phát hiện khuôn mặt
    faces = []
    for result in results:
        for box in result.boxes.xyxy:  # Lấy bounding box
            x1, y1, x2, y2 = map(int, box)
            face = photo[y1:y2, x1:x2]  # Cắt khuôn mặt từ ảnh
            faces.append((face, (x1, y1, x2, y2)))  # Lưu khuôn mặt và tọa độ
    return faces

def predict_emotion(face):
    try:
        if face is not None and face.size > 0 and len(face.shape) == 3:
            # Chuyển numpy.ndarray sang PIL Image
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Transform ảnh
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Dự đoán cảm xúc
            with torch.no_grad():
                outputs = resnet50(face_tensor)
                _, predicted = outputs.max(1)
                emotion_label = emotion_labels[predicted.item()]
                return emotion_label
        else:
            return "Invalid Face"
    except Exception as e:
        print(f"Lỗi khi xử lý khuôn mặt: {e}")
        return "Error"

def process_image(image_path):
    # Đọc ảnh
    photo = cv2.imread(image_path)
    if photo is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")

    # Phát hiện khuôn mặt
    faces = detect_faces(photo)

    for face, (x1, y1, x2, y2) in faces:
        if face is not None and face.size > 0:
            # Dự đoán cảm xúc
            emotion_label = predict_emotion(face)

            # Vẽ bounding box và nhãn cảm xúc lên ảnh
            cv2.rectangle(photo, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(photo, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            print("Khuôn mặt không hợp lệ hoặc không thể xử lý.")

    return photo

def process_video(source=0):
    stframe = st.empty()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened() and isinstance(source, int):
        print("Không thể mở video/camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện khuôn mặt
        faces = detect_faces(frame)

        for face, (x1, y1, x2, y2) in faces:
            if face is not None and face.size > 0:
                # Dự đoán cảm xúc
                emotion_label = predict_emotion(face)

                # Vẽ bounding box và nhãn cảm xúc lên frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Hiển thị frame
        # cv2.imshow("Emotion Detection", frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Nhấn 'q' để thoát
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
