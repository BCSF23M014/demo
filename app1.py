import streamlit as st

# Example SRT content
srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello, this is the first subtitle.

2
00:00:05,000 --> 00:00:08,000
And this is the second subtitle.
"""

# Streamlit app
st.title("Download SRT Example")

# Button to download SRT
st.download_button(
    label="Download SRT",
    data=srt_content,
    file_name="example.srt",
    mime="text/plain"
)
