import streamlit as st
from moviepy import VideoFileClip
import whisper
import subprocess
import os
from datetime import timedelta

st.title("🔥 Video Transcription & Subtitle Burner")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mpeg4"]
)

if uploaded_file:
    # Save video
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded")

    clip = VideoFileClip("input.mp4")
    st.write("Duration:", clip.duration)

    # Extract audio
    audio_path = "audio.wav"
    clip.audio.write_audiofile(audio_path, logger=None)

    @st.cache_resource
    def load_model():
        return whisper.load_model("tiny")

    model = load_model()

    st.info("Transcribing...")
    result = model.transcribe(audio_path)

    # ---------- CREATE SRT ----------
    def format_time(seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        ms = int((seconds - total_seconds) * 1000)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_path = "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(seg["text"].strip() + "\n\n")

    st.success("Subtitles generated")

    # ---------- BURN SUBTITLES ----------
    output_video = "output_with_subs.mp4"

    st.info("Burning subtitles into video...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", "input.mp4",
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_video
    ]

    subprocess.run(cmd, check=True)

    st.success("✅ Subtitles burned successfully!")

    # ---------- SHOW OUTPUT -------
