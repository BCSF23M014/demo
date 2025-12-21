import streamlit as st
import os
from moviepy import VideoFileClip
import whisper
import subprocess
from datetime import timedelta, datetime
import srt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import sqlite3
import spacy

from pydantic import BaseModel, Field
from langchain_classic.docstore.document import Document
from langchain_classic.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from argostranslate import translate, package

# -----------------------------
# API Key
# -----------------------------
api_key = st.secrets.get("GOOGLE_API_KEY")

# -----------------------------
# Load Whisper model
# -----------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

# -----------------------------
# Load Spacy
# -----------------------------
try:
    nlp = spacy.load("xx_sent_ud_sm")
except OSError:
    nlp = None

# -----------------------------
# Conversational Memory
# -----------------------------
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# -----------------------------
# Language Map
# -----------------------------
LANG_MAP = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Russian": "ru",
    "Turkish": "tr",
    "Korean": "ko"
}

# -----------------------------
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            task_type TEXT NOT NULL,
            user_query TEXT,
            agent_response TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(task_type, user_query, agent_response, confidence):
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions (
            timestamp, task_type, user_query, agent_response, confidence
        ) VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), task_type, user_query, agent_response, confidence))
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# History Builder
# -----------------------------
def build_history():
    history_text = ""
    for turn in st.session_state.conversation_memory[-5:]:
        history_text += f"User: {turn['user']}\nAI: {turn['ai']}\n"
    return history_text

# -----------------------------
# Pydantic Models
# -----------------------------
class QAResponse(BaseModel):
    topic_in_video: str
    video_content: str
    general_answer: str
    confidence: float

class SummaryResponse(BaseModel):
    summary: str
    confidence: float

# -----------------------------
# Audio Extraction
# -----------------------------
def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    if audio:
        audio.write_audiofile(output_audio_path)
        return True
    return False

# -----------------------------
# Translation
# -----------------------------
@st.cache_resource
def load_argos(from_code, to_code):
    package.update_package_index()
    available = package.get_available_packages()
    pkg = next(p for p in available if p.from_code == from_code and p.to_code == to_code)
    package.install_from_path(pkg.download())

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    available = package.get_available_packages()
    direct = [p for p in available if p.from_code == src_lang and p.to_code == tgt_lang]
    if direct:
        package.install_from_path(direct[0].download())
        return translate.translate(text, src_lang, tgt_lang)
    if src_lang != "en" and tgt_lang != "en":
        mid = translate_text(text, src_lang, "en")
        return translate_text(mid, "en", tgt_lang)
    return text

# -----------------------------
# Whisper Transcription
# -----------------------------
def transcribe(audio_path):
    model = load_whisper_model()
    st.info("🎧 Transcribing audio...")
    return model.transcribe(audio_path, task="transcribe", verbose=True)

# -----------------------------
# SRT
# -----------------------------
def convert_to_srt(transcription_result):
    segments = transcription_result.get("segments", [])
    subtitles = []
    for i, seg in enumerate(segments):
        subtitles.append(srt.Subtitle(
            index=i+1,
            start=timedelta(seconds=seg["start"]),
            end=timedelta(seconds=seg["end"]),
            content=seg["text"].strip()
        ))
    return srt.compose(subtitles)

import shlex

def burn_subtitles(video_path, srt_path, output_path):
    """
    Burn subtitles into a video using ffmpeg. Works with multilingual SRTs.
    """
    # Ensure paths are absolute
    video_path = os.path.abspath(video_path)
    srt_path = os.path.abspath(srt_path)
    output_path = os.path.abspath(output_path)

    # Build ffmpeg command
    # -sub_charenc UTF-8 ensures non-ASCII characters are handled
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"subtitles={shlex.quote(srt_path)}:charenc=UTF-8",
        "-c:a", "copy",
        output_path
    ]

    print("Running command:", " ".join(command))  # DEBUG: check command
    subprocess.run(command, check=True)



# -----------------------------
# Chatbot Prompt Template
# -----------------------------
video_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a strict, professional AI assistant that answers questions ONLY using the provided video transcript.

Follow these rules strictly:

Line 1: Either "This topic is discussed in the video." OR "This topic is NOT discussed in the video."
Line 2: Transcript-based answer or "N/A"
Line 3: General explanation/definition

Video Transcript:
{context}

Question: {question}
"""
)

category_prompt_text = """
Classify the main topic of this video using ONLY the transcript.

Rules:
- Do NOT guess.
- Do NOT use external knowledge.
- Choose ONE best category.

Allowed Categories:
Education, Technology, Programming, Artificial Intelligence, Data Science,
Business, Entrepreneurship, Marketing, Finance, Health, Science,
Religion, Motivation, Tutorial / How-To, News, Entertainment, Other

Return EXACTLY in this format:

Category: <category>
Reason: <one-line reason from transcript>
"""

# -----------------------------
# Video Chatbot
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_video_chatbot(transcript_text, api_key):
    docs = [Document(page_content=transcript_text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = splitter.split_documents(docs)
    embeddings = load_embeddings()
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="video_transcript")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": video_prompt},
        return_source_documents=True
    )
    return qa_chain

def calculate_confidence(question, answer, source_docs, transcript_text):
    # Simple scoring
    relevance_score = 0.6 if not source_docs else 0.8
    coverage_score = min(len(answer.split()) / max(len(transcript_text.split()), 1), 1.0)
    grounding_score = 0.3 if "NOT discussed" in answer.upper() else 1.0
    length_score = min(len(answer.split()) / 80, 1.0)
    confidence = 0.4*relevance_score + 0.3*coverage_score + 0.2*grounding_score + 0.1*length_score
    return round(max(0.0, min(confidence,1.0)), 2)

# -----------------------------
# Video Analyzer
# -----------------------------
def analyze_transcript(text, top_n=10):
    words = re.findall(r'\b\w+\b', text.lower())
    counter = Counter(words)
    most_common = counter.most_common(top_n)
    return most_common, counter

def plot_wordcloud(text):
    if not text.strip():
        st.warning("No text available for word cloud.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc.to_array(), interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Video Assistant", layout="wide")
st.title("📹 AI-powered Video Assistant")

# Upload Video
uploaded_video = st.file_uploader("Upload Video", type=["mp4","mov"], key="upload_video")
if uploaded_video:
    os.makedirs("videos", exist_ok=True)
    video_path = "videos/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.video(video_path)

    # Extract Audio
    if "transcript_text" not in st.session_state:
        st.session_state["transcript_text"] = ""
        st.session_state["chat_history"] = []
        os.makedirs("audio", exist_ok=True)
        audio_path = "audio/uploaded_audio.wav"
        st.info("Extracting audio and transcribing video...")

        if extract_audio(video_path, audio_path):
            with st.spinner("🧠 Whisper is transcribing..."):
                transcription_result = transcribe(audio_path)
                st.session_state["transcription_result"] = transcription_result
                st.session_state["transcript_text"] = transcription_result.get("text","")
                st.success("✅ Transcription completed!")
        else:
            st.error("❌ No audio found in video.")

# Caption Language
st.subheader("Caption Language Selection")
target_language = st.selectbox("Select language for captions", list(LANG_MAP.keys()), key="lang_select")

# -----------------------------
# Generate Transcript
# -----------------------------
if st.button("Generate Transcript", key="generate_transcript_btn") and st.session_state.get("transcript_text"):
    src_lang = st.session_state["transcription_result"]["language"]
    target_code = LANG_MAP[target_language]

    # Translate full text
    translated_text = translate_text(st.session_state["transcription_result"]["text"], src_lang, target_code)

    # Translate segments
    for seg in st.session_state["transcription_result"]["segments"]:
        seg["text"] = translate_text(seg["text"], src_lang, target_code)

    translated_full_text = "\n".join(seg["text"] for seg in st.session_state["transcription_result"]["segments"])

    st.subheader("📜 Full Translated Transcript")
    st.text_area("Translated Transcript", translated_full_text, height=300, key="translated_transcript_area")

    # Generate SRT
    srt_content = convert_to_srt(st.session_state["transcription_result"])
    os.makedirs("captions", exist_ok=True)
    srt_path = "captions/output.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    st.info("Burning subtitles...")
    burn_subtitles(video_path, srt_path, "videos/output_video.mp4")
    st.success("✅ Video with captions ready!")
    st.video("videos/output_video.mp4")

# -----------------------------
# 💬 Video Chatbot
# -----------------------------
st.subheader("💬 Video Chatbot")
user_input = st.text_input("You:", key="chat_input")

if user_input:
    if st.session_state.get("transcript_text"):
        try:
            # Create chatbot chain
            chatbot = create_video_chatbot(
                st.session_state["transcript_text"],
                api_key
            )

            MEMORY_TRIGGERS = [
                "previous question",
                "last question",
                "what did i ask",
                "earlier question",
                "before this"
            ]

            def is_memory_question(text):
                return any(t in text.lower() for t in MEMORY_TRIGGERS)

            # Handle memory questions
            if is_memory_question(user_input):
                if st.session_state.conversation_memory:
                    last = st.session_state.conversation_memory[-1]["user"]
                    st.success(f"Your previous question was:\n\n**{last}**")

            # Invoke chatbot
            result = chatbot.invoke({"query": user_input})
            answer_text = result["result"]

            confidence = calculate_confidence(
                question=user_input,
                answer=answer_text,
                source_docs=result.get("source_documents", []),
                transcript_text=st.session_state["transcript_text"]
            )

            qa_validated = QAResponse(
                topic_in_video=answer_text.splitlines()[0],
                video_content=answer_text.splitlines()[1],
                general_answer=answer_text.splitlines()[2],
                confidence=confidence
            )

            # -----------------------------
            # Format response depending on N/A
            # -----------------------------
            if qa_validated.video_content.strip().upper() == "N/A":
                response_text = f"""
Present in Video: {qa_validated.topic_in_video}

General answer: {qa_validated.general_answer}
"""
            else:
                response_text = f"""
Present in Video: {qa_validated.topic_in_video}

Answer from Video: {qa_validated.video_content}

General answer: {qa_validated.general_answer}
"""

            # -----------------------------
            # Append conversation to memory
            # -----------------------------
            if "conversation_memory" not in st.session_state:
                st.session_state.conversation_memory = []

            st.session_state.conversation_memory.append({
                "user": user_input,
                "ai": response_text
            })

            # Log interaction
            log_interaction(
                task_type="QA",
                user_query=user_input,
                agent_response=response_text,
                confidence=confidence
            )

            # -----------------------------
            # Display all conversation history
            # -----------------------------
            st.markdown("### 🕒 Chat History")
            for turn in st.session_state.conversation_memory:
                st.markdown(f"**You:** {turn['user']}")
                st.markdown(f"**AI:** {turn['ai']}")
                st.markdown("---")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Transcript not ready yet!")


# -----------------------------
# Summarization
# -----------------------------
st.subheader("📝 Video Summarization")
if st.button("Summarize Video", key="summarize_btn"):
    if st.session_state.get("transcript_text"):
        try:
            chatbot = create_video_chatbot(st.session_state["transcript_text"], api_key)
            summary_result = chatbot.invoke("Summarize this video transcript in a few sentences.")
            summary_validated = SummaryResponse(summary=summary_result["result"], confidence=1.0)
            st.write(summary_validated.summary)
            st.write("Confidence:", summary_validated.confidence)
            log_interaction("SUMMARY","Summarize video",summary_validated.summary,summary_validated.confidence)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Transcript not ready yet!")

# -----------------------------
# Categorization
# -----------------------------
st.subheader("🏷️ Video Categorization")
if st.button("Categorize Video", key="categorize_btn"):
    if st.session_state.get("transcript_text"):
        try:
            chatbot = create_video_chatbot(st.session_state["transcript_text"], api_key)
            category_result = chatbot.invoke(category_prompt_text)
            st.write(category_result["result"])
            log_interaction("CATEGORY","Categorize video",category_result["result"],1.0)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Transcript not ready yet!")

# -----------------------------
# Video Analysis
# -----------------------------
st.subheader("📊 Video Transcript Analysis")
if st.session_state.get("transcript_text"):
    most_common, counter = analyze_transcript(st.session_state["transcript_text"], top_n=15)
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("### 🔑 Top Keywords")
        for word, count in most_common:
            st.write(f"{word} ({count})")
    with col2:
        st.markdown("### ☁️ Word Cloud")
        plot_wordcloud(st.session_state["transcript_text"])

# -----------------------------
# Agent Logs
# -----------------------------
st.subheader("🗄️ Agent Interaction Logs")
if st.checkbox("Show Database Logs", key="show_logs"):
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    for row in rows:
        st.markdown(f"""
**ID:** {row[0]}  
**Time:** {row[1]}  
**Task:** {row[2]}  
**Query:** {row[3]}  
**Response:** {row[4][:300]}  
**Confidence:** {row[5]}
---
""")
