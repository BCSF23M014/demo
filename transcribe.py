import streamlit as st
from moviepy.editor import VideoFileClip
import whisper

st.title("Video Transcription App")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save video temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Video uploaded successfully!")

    # Extract audio
    st.info("Extracting audio...")
    clip = VideoFileClip(temp_video_path)
    temp_audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(temp_audio_path)
    st.success("Audio extracted!")

    # Load Whisper model
    @st.cache_resource
    def load_whisper_model():
        return whisper.load_model("tiny")
    
    model = load_whisper_model()
    st.info("Whisper model loaded!")

    # Transcribe audio
    st.info("Transcribing audio...")
    result = model.transcribe(temp_audio_path)
    st.success("Transcription done!")

    # Show transcription
    st.subheader("Transcribed Text")
    st.text(result["text"])
