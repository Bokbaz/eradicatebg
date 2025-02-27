import numpy as np
import os
import subprocess
import yt_dlp
import streamlit as st
import imageio_ffmpeg as ffmpeg_lib
import uuid
import time
import stripe
import cv2  # Ensure opencv-python-headless is used

# Retrieve Stripe secret key from Streamlit's secrets management (TOML format)
stripe.api_key = st.secrets["stripe_secret_key"]

if not stripe.api_key:
    raise ValueError("Stripe secret key not found! Set it in Streamlit secrets.")

def ensure_directory_exists(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_youtube_video(youtube_url, download_path):
    """Download a YouTube video using yt_dlp."""
    ensure_directory_exists(download_path)
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s')
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        output_file = ydl.prepare_filename(info_dict)
    return output_file

def process_video(input_video_path, output_video_path, temp_path):
    ensure_directory_exists("temp")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open the video file '{input_video_path}'. Ensure it is valid.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))

    GREEN = (0, 255, 0)
    BACKGROUND_COLOR = [255, 255, 255]  # Example: White background
    COLOR_THRESHOLD = 60  # Adjust as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space for easier color segmentation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        background_color_hsv = cv2.cvtColor(np.uint8([[BACKGROUND_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]

        # Create a mask for the background color
        lower_bound = np.array([
            max(0, background_color_hsv[0] - COLOR_THRESHOLD),
            max(0, background_color_hsv[1] - COLOR_THRESHOLD),
            max(0, background_color_hsv[2] - COLOR_THRESHOLD)
        ])
        upper_bound = np.array([
            min(179, background_color_hsv[0] + COLOR_THRESHOLD),
            min(255, background_color_hsv[1] + COLOR_THRESHOLD),
            min(255, background_color_hsv[2] + COLOR_THRESHOLD)
        ])

        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)

        # Apply the green screen background
        green_background = np.zeros(frame.shape, dtype=np.uint8)
        green_background[:] = GREEN

        # Combine the original frame with the green background
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        bg = cv2.bitwise_and(green_background, green_background, mask=mask)
        output_frame = cv2.add(fg, bg)

        out.write(output_frame)

    cap.release()
    out.release()

    ffmpeg_path = ffmpeg_lib.get_ffmpeg_exe()
    result = subprocess.run([
        ffmpeg_path, "-i", temp_path, "-i", input_video_path,
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_video_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"FFmpeg failed with error: {result.stderr}")

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

    # Stripe Checkout integration
    st.write("### Step 1: Payment")

    payment_successful = st.session_state.get("success", False)

    if not payment_successful:
        if st.button("Pay $1 to Process Video"):
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': 'Green Screen Video Processing',
                        },
                        'unit_amount': 100,
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url="https://greenscreen.streamlit.app?success=true",
                cancel_url="https://greenscreen.streamlit.app?canceled=true",
            )
            st.markdown(f"[Click here to pay]({session.url})", unsafe_allow_html=True)

    query_params = st.query_params
    if "success" in query_params and not payment_successful:
        st.session_state["success"] = True
        st.success("Payment successful! Proceed to upload your video.")

    if st.session_state.get("success", False):
        st.write("### Step 2: Upload or Provide Video URL")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
        youtube_url = st.text_input("Or provide a YouTube video URL")

        if st.button("Process Video"):
            try:
                if uploaded_file:
                    input_video_path = os.path.join("temp", uploaded_file.name)
                    ensure_directory_exists("temp")
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

                st.session_state["success"] = False

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif "canceled" in query_params:
        st.warning("Payment canceled. Please try again.")
