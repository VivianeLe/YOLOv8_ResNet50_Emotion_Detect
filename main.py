import streamlit as st
import cv2
import tempfile
from model.support import process_video, process_image

# Streamlit app
st.title("Emotion Detection with YOLOv8 and ResNet50")

option = st.sidebar.selectbox(
    "Choose Input Type",
    ("Image Upload", "Video Upload", "Realtime Camera")
)

if option == "Image Upload":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read image
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_image.read())
            temp_path = temp_file.name

        output_image = process_image(temp_path)

        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Detected Faces and Emotions", use_container_width=True)

elif option == "Video Upload":
    # Upload video
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        process_video(temp_video_path)


elif option == "Realtime Camera":
    st.header("Realtime Camera Detection")
    st.write("Click 'Start' to begin capturing.")
    start_button = st.button("Start Camera")

    if start_button:
        process_video(0)

