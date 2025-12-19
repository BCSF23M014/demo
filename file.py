import streamlit as st
import os

st.title("Load File Example")

# Step 1: File uploader
uploaded_file = st.file_uploader("Choose a file", type=["mp4", "mov", "srt", "txt"])

if uploaded_file:
    # Step 2: Show a button to load/save the file
    if st.button("Load File"):
        os.makedirs("uploaded_files", exist_ok=True)
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File loaded successfully: {uploaded_file.name}")
        
        # Optional: preview video or text
        if uploaded_file.type.startswith("video"):
            st.video(file_path)
        elif uploaded_file.type == "text/plain" or uploaded_file.type == "application/x-subrip":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.text_area("File Content", content, height=300)
