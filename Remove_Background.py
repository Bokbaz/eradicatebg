import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import yt_dlp
import streamlit as st
import imageio_ffmpeg as ffmpeg_lib

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

    # Debugging: Print absolute paths
    print(f"Absolute input path: {input_video_path}")
    print(f"Absolute output path: {output_video_path}")

    # Check if the input file exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"The file '{input_video_path}' does not exist.")

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception(f"Could not open the video file '{input_video_path}'. Ensure it is a valid video file and not corrupted.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))

    # Green color for background replacement
    GREEN = (0, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB as required by Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for segmentation
        result = selfie_segmentation.process(frame_rgb)

        # Create a mask where the background is identified
        mask = result.segmentation_mask > 0.5

        # Create the green screen background
        green_background = np.zeros(frame.shape, dtype=np.uint8)
        green_background[:] = GREEN

        # Combine the original frame with the green background
        output_frame = np.where(mask[:, :, None], frame, green_background)

        # Write the processed frame to the temporary output video
        out.write(output_frame)

    # Release resources
    cap.release()
    out.release()

    # Use ffmpeg to retain audio using imageio-ffmpeg's binary
    ffmpeg_path = ffmpeg_lib.get_ffmpeg_exe()
    ffmpeg_command = [
        ffmpeg_path,
        "-i", temp_path,
        "-i", input_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_video_path
    ]
    subprocess.run(ffmpeg_command)

    # Remove temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == "__main__":
    st.title("Green Screen Video Processor")

    # User inputs
    st.write("### Upload a video or provide a YouTube URL")

    # Option 1: Upload a video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    # Option 2: Provide a YouTube URL
    youtube_url = st.text_input("Or provide a YouTube video URL")

    # Process button
    if st.button("Process Video"):
        try:
            if uploaded_file is not None:
                # Save the uploaded file locally
                input_video_path = os.path.join("temp", uploaded_file.name)
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_file.read())

            elif youtube_url:
                # Download the YouTube video
                st.write("Downloading YouTube video...")
                input_video_path = download_youtube_video(youtube_url, "temp")
                st.write("Download complete.")

            else:
                st.error("Please upload a video file or provide a YouTube URL.")
                st.stop()

            # Define output paths
            output_video_name = "output.mp4"
            output_video_path = os.path.join("temp", output_video_name)
            temp_path = os.path.join("temp", "temp_video.mp4")

            # Process the video
            st.write("Processing video...")
            process_video(input_video_path, output_video_path, temp_path)

            # Provide download link for the output video
            st.write("Processing complete. Download your green screen video below:")
            st.video(output_video_path)
            with open(output_video_path, "rb") as f:
                st.download_button("Download Video", f, file_name=output_video_name)

            # Clean up temporary files
            if os.path.exists(input_video_path):
                os.remove(input_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")
