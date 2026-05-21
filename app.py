import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()


# ── API Key ───────────────────────────────────────────────────────────────────
def get_groq_api_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


# ── Session State ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube RAG", page_icon="🎬", layout="wide")
st.title("🎬 YouTube Video Q&A with RAG")
st.caption("Paste a YouTube URL, get a summary, then ask anything about the video.")


# ── URL Parsing ───────────────────────────────────────────────────────────────
def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/shorts/([^?&]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ── Transcript Helpers ────────────────────────────────────────────────────────
def snippets_to_text(fetched_transcript) -> str:
    """Handle all transcript formats returned by any fetch method."""
    if isinstance(fetched_transcript, list):
        if not fetched_transcript:
            return ""
        first = fetched_transcript[0]
        if isinstance(first, dict):
            return " ".join(t.get("text", "") for t in fetched_transcript)
    if hasattr(fetched_transcript, "snippets"):
        return " ".join(s.text for s in fetched_transcript.snippets)
    return " ".join(t["text"] for t in fetched_transcript)


def fetch_via_ytdlp(video_id: str) -> list:
    """
    Fetch transcript using yt-dlp.
    Bypasses YouTube IP blocks entirely — most reliable method.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US"],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # Try manual subtitles first, then auto-generated captions
        for caption_type in ["subtitles", "automatic_captions"]:
            captions = info.get(caption_type, {})
            for lang in ["en", "en-US", "en-GB", "en-orig"]:
                if lang not in captions:
                    continue
                entries = captions[lang]
                for fmt in entries:
                    # --- JSON3: richest format ---
                    if fmt.get("ext") == "json3":
                        try:
                            resp = requests.get(fmt["url"], timeout=15)
                            data = resp.json()
                            events = data.get("events", [])
                            transcript = []
                            for event in events:
                                segs = event.get("segs", [])
                                text = "".join(
                                    s.get("utf8", "") for s in segs
                                ).strip()
                                if text and text != "\n":
                                    transcript.append(
                                        {
                                            "text": text,
                                            "start": event.get("tStartMs", 0) / 1000,
                                        }
                                    )
                            if transcript:
                                return transcript
                        except Exception:
                            continue

                    # --- VTT / SRV fallback ---
                    elif fmt.get("ext") in ["vtt", "srv3", "srv2", "srv1"]:
                        try:
                            resp = requests.get(fmt["url"], timeout=15)
                            lines = resp.text.splitlines()
                            text_lines = [
                                l.strip()
                                for l in lines
                                if l.strip()
                                and not l.startswith("WEBVTT")
                                and not l.startswith("NOTE")
                                and "-->" not in l
                                and not l.strip().isdigit()
                            ]
                            combined = " ".join(text_lines)
                            if combined:
                                return [{"text": combined}]
                        except Exception:
                            continue

    raise RuntimeError(
        "yt-dlp could not find any English captions. "
        "The video may not have subtitles enabled."
    )


def fetch_transcript(video_id: str):
    """
    3-layer fallback for maximum reliability:
      1. youtube-transcript-api  (fast, direct)
      2. youtube-transcript-api with cookies.txt  (if IP blocked)
      3. yt-dlp  (bypasses all IP restrictions — interview-safe)
    """

    # --- Layer 1: Direct API ---
    try:
        api = YouTubeTranscriptApi()
        return api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except (TranscriptsDisabled, VideoUnavailable) as e:
        raise RuntimeError(str(e))
    except Exception:
        pass  # IP blocked or not found → try next layer

    # --- Layer 2: With cookies (if cookies.txt present) ---
    cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
    if os.path.exists(cookie_path):
        try:
            api = YouTubeTranscriptApi(cookie_path=cookie_path)
            return api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            pass  # cookies expired → try next layer

    # --- Layer 3: yt-dlp (most reliable) ---
    try:
        return fetch_via_ytdlp(video_id)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"All transcript methods failed: {e}")


# ── Indexing ──────────────────────────────────────────────────────────────────
def load_and_index(url: str):
    with st.spinner("⏳ Fetching transcript..."):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check the link and try again.")
            return None, None

        try:
            transcript = fetch_transcript(video_id)
        except RuntimeError as e:
            st.error(f"❌ {e}")
            return None, None
        except Exception as e:
            st.error(f"❌ Unexpected error fetching transcript: {e}")
            return None, None

        text = snippets_to_text(transcript)
        if not text.strip():
            st.error("❌ Transcript is empty.")
            return None, None

        docs = [Document(page_content=text)]

    with st.spinner("✂️ Chunking text..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)

    with st.spinner("🔢 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)

    return db, chunks


# ── Summarisation ─────────────────────────────────────────────────────────────
def summarize(chunks, llm):
    sample_text = " ".join([c.page_content for c in chunks[:5]])
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Given the following transcript excerpt from a YouTube video,
write a concise 5-bullet summary covering the main topics discussed.

Transcript:
{text}

Summary (5 bullets):""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": sample_text})


# ── QA Chain ──────────────────────────────────────────────────────────────────
def build_qa_chain(db, model_name):
    llm = ChatGroq(model=model_name, temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are an expert assistant helping users understand a YouTube video.

IMPORTANT:
- Always answer in English.
- If the context is in another language, translate it to English before answering.

Context:
{context}

Question: {question}

Answer (in English):""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, llm


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    default_key = get_groq_api_key()
    api_key = st.text_input(
        "GROQ API Key",
        value=default_key,
        type="password",
        help="Loaded automatically from .env or Streamlit secrets if set.",
    )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    model_name = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
    )

    st.divider()
    url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if st.button("🚀 Load and Index", use_container_width=True):
        if not api_key:
            st.error("Please enter your GROQ API key.")
        elif not url:
            st.error("Please enter the URL of the video.")
        else:
            try:
                db, chunks = load_and_index(url=url)
                if db is not None:
                    chain, llm = build_qa_chain(db, model_name=model_name)
                    st.session_state.vectorstore = db
                    st.session_state.chain = chain
                    st.session_state.summary = summarize(chunks=chunks, llm=llm)
                    st.session_state.chat_history = []
                    st.success(f"✅ Indexed {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.vectorstore:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.rerun()

    st.divider()
    st.caption("💡 Tip: Add a `cookies.txt` file to your project folder for extra reliability.")


# ── Summary ───────────────────────────────────────────────────────────────────
if st.session_state.summary:
    with st.expander("📋 Auto-generated Summary", expanded=True):
        st.markdown(st.session_state.summary)


# ── Chat ──────────────────────────────────────────────────────────────────────
if st.session_state.vectorstore:
    st.subheader("💬 Ask anything about the video")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("e.g. What are the main takeaways?"):
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
            st.write(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

else:
    st.info("👈 Enter a YouTube URL in the sidebar and click **Load and Index** to get started.")