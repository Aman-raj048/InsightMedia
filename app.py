import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load .env before importing project modules
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from main import run_pipeline
from core.rag_engine import ask_question


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="InsightMedia",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Header */

.header {
    padding: 20px 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #123d70, #1769aa);
    color: white;
    margin-bottom: 25px;
}

.header-title {
    font-size: 32px;
    font-weight: 700;
}

.header-subtitle {
    font-size: 15px;
    opacity: 0.9;
}


/* Cards */

.card {
    background: #ffffff;
    color: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}

.card * {
    color: #111827 !important;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
}


/* Result cards */

.result-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    min-height: 150px;
}

.result-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}


/* Chat */

.chat-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
}


/* Sidebar Footer */
.sidebar-footer {
    
    padding: 15px 5px 5px 5px;
    color: #9ca0aa;
    font-size: 14px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🎙️ InsightMedia")

    st.caption("AI Video & Meeting Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🆕 New Analysis",
            "💬 Chat (RAG)"
        ]
    )

    st.divider()

    st.markdown("### 🚀 Features")

    st.markdown("""
    - 🎬 YouTube / Video Input
    - 🎙️ Whisper Transcription
    - 🤖 AI Summary
    - ✅ Action Items
    - 🔑 Key Decisions
    - ❓ Open Questions
    - 🧠 RAG Q&A
    """)

    st.divider()

    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-footer">
        <div>InsightMedia • AI Video Assistant</div>
        <div>Built by Aman</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="header">
    <div class="header-title">🎙️ InsightMedia</div>
    <div class="header-subtitle">
        Get smart insights from YouTube videos or your local files
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.markdown("## Welcome to InsightMedia 👋")

    st.write(
        "Convert videos and meetings into transcripts, summaries, "
        "action items, decisions and searchable AI insights."
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🎬\n\n**Video Processing**\n\nYouTube URL or local video")

    with col2:
        st.success("🤖\n\n**AI Insights**\n\nSummary and structured information")

    with col3:
        st.warning("💬\n\n**RAG Chat**\n\nAsk questions about your video")

    st.markdown("### 🔄 How it works")

    st.markdown("""
    <div style="
        background: #162b3d;
        padding: 28px 30px;
        border-radius: 15px;
        border: 1px solid #29465c;
        text-align: center;
        font-size: 17px;
        font-weight: 600;
        color: #f1f5f9;
    ">
        🎬 Video
        <span style="color:#6b7280;"> → </span>
        🎧 Audio
        <span style="color:#6b7280;"> → </span>
        📝 Transcript
        <span style="color:#6b7280;"> → </span>
        🤖  Groq AI
        <span style="color:#6b7280;"> → </span>
        🧠 ChromaDB
        <span style="color:#6b7280;"> → </span>
        💬 RAG
        <span style="color:#6b7280;"> → </span>
        📊 Insights
    </div>

    <div style="
        text-align: center;
        margin-top: 12px;
        color: #6b7280;
        font-size: 14px;
    ">
        Upload → Process → Analyze → Store → Ask → Get Insights
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# NEW ANALYSIS
# ==================================================

elif page == "🆕 New Analysis":

    st.markdown("## 🆕 New Analysis")

    st.write("Upload a video/audio file or provide a YouTube URL.")

    input_type = st.radio(
        "Choose input type",
        ["YouTube URL", "Local File"],
        horizontal=True
    )

    source = None

    if input_type == "YouTube URL":

        source = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload video/audio",
            type=[
                "mp4",
                "mkv",
                "mov",
                "avi",
                "mp3",
                "wav",
                "m4a"
            ]
        )

        if uploaded_file:

            temp_dir = tempfile.gettempdir()

            file_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            source = file_path

    language = st.selectbox(
        "Language",
        ["english", "hinglish"]
    )

    st.markdown("")

    analyze = st.button(
        "🚀 Analyze Video",
        type="primary",
        use_container_width=True
    )

    if analyze:

        if not source:

            st.warning(
                "Please provide a YouTube URL or upload a file."
            )

        else:

            st.session_state.messages = []

            with st.status(
                "Processing your video...",
                expanded=True
            ):

                try:

                    st.write("🎬 Processing video/audio...")
                    st.write("🎙️ Transcribing with Whisper...")
                    st.write("🤖 Generating AI insights...")
                    st.write("🧠 Building RAG vector store...")

                    result = run_pipeline(
                        source,
                        language
                    )

                    st.session_state.result = result

                    if input_type == "YouTube URL":
                        st.session_state.source_name = source
                    else:
                        st.session_state.source_name = uploaded_file.name

                    st.success(
                        "Analysis completed successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )


# ==================================================
# DISPLAY RESULTS
# ==================================================

if st.session_state.result is not None:

    result = st.session_state.result

    st.markdown("---")

    st.markdown("## 📊 Analysis Results")

    # ----------------------------------------------
    # Title
    # ----------------------------------------------

    with st.container(border=True):

        st.markdown("### 📌 Title")

        st.markdown(
            f"## {result['title']}"
        )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    with st.container(border=True):

        st.markdown("### 📋 Summary")

        st.markdown(
            result["summary"]
        )

    # ----------------------------------------------
    # Action Items + Key Decisions
    # ----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### ✅ Action Items")

            st.markdown(
                result["action_items"]
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 🔑 Key Decisions")

            st.markdown(
                result["key_decisions"]
            )

    # ----------------------------------------------
    # Open Questions
    # ----------------------------------------------

    with st.container(border=True):

        st.markdown("### ❓ Open Questions")

        st.markdown(
            result["open_questions"]
        )

    # ----------------------------------------------
    # Transcript
    # ----------------------------------------------

    with st.expander("📄 View Full Transcript"):

        st.write(
            result["transcript"]
        )

# ==================================================
# CHAT
# ==================================================

if (
    page == "💬 Chat (RAG)"
    and st.session_state.result is not None
):

    st.markdown("## 💬 Chat with your video")

    st.caption(
        "Ask questions based only on the processed video transcript."
    )

    rag_chain = st.session_state.result["rag_chain"]

    # Previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about your video..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = ask_question(
                    rag_chain,
                    question
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


elif page == "💬 Chat (RAG)":

    st.info(
        "Please analyze a video first, then you can chat with it."
    )