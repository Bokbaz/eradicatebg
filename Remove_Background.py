import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import yt_dlp
import streamlit as st
import imageio_ffmpeg as ffmpeg_lib
import uuid
import time

def download_youtube_video(youtube_url, download_path):
    """Download a YouTube video using yt_dlp."""
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s')
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        output_file = ydl.prepare_filename(info_dict)
    return output_file

def process_video(input_video_path, output_video_path, temp_path):
    # Initialize Mediapipe solutions for selfie segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open the video file '{input_video_path}'. Ensure it is valid.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))

    GREEN = (0, 255, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = selfie_segmentation.process(frame_rgb)
        mask = result.segmentation_mask > 0.5

        green_background = np.zeros(frame.shape, dtype=np.uint8)
        green_background[:] = GREEN
        output_frame = np.where(mask[:, :, None], frame, green_background)
        out.write(output_frame)

    cap.release()
    out.release()

    ffmpeg_path = ffmpeg_lib.get_ffmpeg_exe()
    ffmpeg_command = [
        ffmpeg_path, "-i", temp_path, "-i", input_video_path,
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_video_path
    ]
    subprocess.run(ffmpeg_command)

    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == "__main__":
    st.set_page_config(page_title="Green Screen Video Processor", layout="wide", initial_sidebar_state="expanded")

    # Apply custom styles for deep jade green theme
    st.markdown(
        """
        <style>
        body {
            background-color: #004d40;
            color: #e0f2f1;
        }
        .stButton>button {
            background-color: #00796b;
            color: white;
            border: None;
        }
        .stButton>button:hover {
            background-color: #004d40;
            color: #e0f2f1;
        }
        .stTextInput>div>input {
            background-color: #004d40;
            color: #e0f2f1;
            border: 1px solid #00796b;
        }
        .stTextInput>div>input:focus {
            outline: none;
            border: 1px solid #004d40;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Green Screen Video Processor")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    youtube_url = st.text_input("Or provide a YouTube video URL")

    if st.button("Process Video"):
        try:
            if uploaded_file:
                input_video_path = os.path.join("temp", uploaded_file.name)
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                del uploaded_file

            elif youtube_url:
                input_video_path = download_youtube_video(youtube_url, "temp")

            else:
                st.error("Please upload a video or provide a YouTube URL.")
                st.stop()

            unique_id = str(uuid.uuid4())
            temp_path = os.path.join("temp", f"temp_video_{unique_id}.mp4")
            output_video_path = os.path.join("temp", f"output_{unique_id}.mp4")

            st.write("Processing video...")
            process_video(input_video_path, output_video_path, temp_path)

            st.write("Processing complete. Download your green screen video below:")
            st.video(output_video_path)
            with open(output_video_path, "rb") as f:
                st.download_button("Download Video", f, file_name=f"output_{unique_id}.mp4")

            time.sleep(1)
            os.remove(input_video_path)
            os.remove(output_video_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")
