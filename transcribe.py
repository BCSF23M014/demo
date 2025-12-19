import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import os

st.title("Video Transcription App")

# Step 1: Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save temporary video file
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Video uploaded successfully!")

    # Step 2: Extract audio
    st.info("Extracting audio from video...")
    clip = VideoFileClip(temp_video_path)
    temp_audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(temp_audio_path)
    st.success("Audio extracted!")

    # Step 3: Load Whisper model
    @st.cache_resource
    def load_whisper_model():
        return whisper.load_model("tiny")  # You can change to "base", "small", etc.
    
    model = load_whisper_model()
    st.info("Whisper model loaded!")

    # Step 4: Transcribe audio
    st.info("Transcribing audio...")
    result = model.transcribe(temp_audio_path)
    st.success("Transcription done!")

    # Step 5: Show transcription
    st.subheader("Transcribed Text")
    st.text(result["text"])
