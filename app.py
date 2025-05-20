# 🔒 Avoid Streamlit + torch.classes crash
import os
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

import streamlit as st
import tempfile
import time
from PIL import Image
from ultralytics import YOLO
import shutil
import numpy as np
import cv2
import shutil
import google.generativeai as genai
import glob
import subprocess
import imageio_ffmpeg

# ========== Load Gemini API Key ==========
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# ========== Streamlit Config ==========
st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🪖 Helmet Detection App")
st.markdown("Upload an image or video to detect **helmets** and get feedback.")

# ========== Load YOLOv8 Safely ==========
if "yolo_model" not in st.session_state:
    try:
        st.session_state.yolo_model = YOLO("slefMadeModel.pt")
    except RuntimeError:
        st.error("❌ YOLO load failed. Restart the app.")
        st.stop()

model = st.session_state.yolo_model

# ========== Gemini Description ==========
def explain_with_gemini(image_path):
    try:
        model_g = genai.GenerativeModel("gemini-1.5-flash")
        with open(image_path, "rb") as f:
            img_data = f.read()
        response = model_g.generate_content([
            "well this image is passed to my model for training.so you have to explain two things about it.wether a person is wearing helmet or not.also explain the details in pic.like red bike and a ccar in the background and two persons on thebike etc",
            {"mime_type": "image/jpeg", "data": img_data}
        ])
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# ========== Video Conversion (if needed) ==========
def convert_to_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.call([
        ffmpeg_path, "-y", "-i", input_path,
        "-vcodec", "libx264", "-acodec", "aac", output_path
    ])

# ========== Sidebar ==========
task = st.sidebar.selectbox("Choose Input Type", ["Image", "Video"])

# ========== Image Processing ==========
if task == "Image":
    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Detecting..."):
            results = model(img)
            result_img = results[0].plot()
            result_img_pil = Image.fromarray(result_img)
            st.image(result_img_pil, caption="Detection Result", use_container_width=True)

            temp_img_path = os.path.join(tempfile.gettempdir(), "helmet_result.jpg")
            result_img_pil.save(temp_img_path)

            explanation = explain_with_gemini(temp_img_path)
            st.subheader("AI Explanation")
            st.write(explanation)

# ========== Video Processing ==========
elif task == "Video":
    uploaded_vid = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_vid:
        temp_dir = tempfile.mkdtemp()
        extension = os.path.splitext(uploaded_vid.name)[-1]
        input_path = os.path.join(temp_dir, f"input{extension}")

        with open(input_path, "wb") as f:
            f.write(uploaded_vid.read())

        st.info("📦 Video uploaded. Running full YOLOv8 processing...")

        timestamp = str(int(time.time()))
        output_folder = f"runs/streamlit/output_{timestamp}"

        with st.spinner("Processing video..."):
            model.predict(
                source=input_path,
                save=True,
                save_txt=False,
                save_conf=False,
                project="runs/streamlit",
                name=f"output_{timestamp}",
                exist_ok=True,
                vid_stride=1,
                show=False
            )

        # Wait for processed video to appear
        st.write("🔍 Scanning for processed video...")
        output_path = None
        for _ in range(30):
            video_files = glob.glob(os.path.join(output_folder, "*.*"))
            for v in video_files:
                if v.endswith((".mp4", ".avi", ".mov")):
                    output_path = v
                    break
            if output_path:
                break
            time.sleep(0.5)

        if output_path and os.path.exists(output_path):
            # Convert to mp4 if needed
            if not output_path.endswith(".mp4"):
                converted_path = output_path.rsplit(".", 1)[0] + "_converted.mp4"
                convert_to_mp4(output_path, converted_path)
                output_path = converted_path

            st.success("✅ Video processed. Preview below:")
            st.video(output_path)

        else:
            st.error("❌ Processed video not found. Try again.")

        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            st.warning(f"⚠️ Cleanup failed: {e}")

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # delete file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # delete subfolder
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}: {e}")

clear_folder("runs/streamlit")