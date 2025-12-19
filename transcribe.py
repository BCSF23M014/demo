import streamlit as st
from moviepy import VideoFileClip
import whisper
import os
import imageio_ffmpeg

# 🔥 Fix ffmpeg path for Streamlit Cloud
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

st.title("Video Transcription App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded")

    clip = VideoFileClip("temp_video.mp4")
    st.write("Duration:", clip.duration)

    audio_path = "audio.wav"
    clip.audio.write_audiofile(audio_path, logger=None)

    @st.cache_resource
    def load_model():
        return whisper.load_model("tiny")

    model = load_model()

    st.info("Transcribing…")
    result = model.transcribe(audio_path)

    st.subheader("Transcription")
    st.write(result["text"])
