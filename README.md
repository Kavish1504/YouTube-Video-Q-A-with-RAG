# 🎬 YouTube Video Q&A with RAG

An interactive Streamlit application that allows users to:
- Load any YouTube video
- Automatically generate a summary
- Ask questions about the video using Retrieval-Augmented Generation (RAG)

---

## 🚀 Features

- 📺 Load YouTube videos via URL  
- 🧠 Automatic transcript extraction  
- ✂️ Smart text chunking  
- 🔍 Semantic search using FAISS  
- 🤖 Q&A powered by LLM (Groq)  
- 📝 Auto-generated summary  
- 🌐 Multi-language support (auto-translation to English)  
- 💬 Chat-based interface  

---

## 🛠️ Tech Stack

- Frontend: Streamlit  
- LLM: Groq (LLaMA / Mixtral / Gemma)  
- Embeddings: HuggingFace (all-MiniLM-L6-v2)  
- Vector DB: FAISS  
- Framework: LangChain  
- Transcript API: youtube-transcript-api (v1.0+)  

---

## 📂 Project Structure

```
YouTube-RAG/
│── app.py
│── .env
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Kavish1504/YouTube-Video-Q-A-with-RAG
cd YouTube-RAG
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file and add:
```bash
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the App
```bash
streamlit run app.py
```

---

## 📌 How It Works

1. User inputs a YouTube URL  
2. Transcript is extracted (with language fallback + translation)  
3. Text is split into chunks  
4. Embeddings are created using HuggingFace  
5. Stored in FAISS vector database  
6. RetrievalQA fetches relevant chunks  
7. LLM generates answers based on context  

---

## ⚠️ Known Issues

- YouTube may block transcript requests (IP-based restriction)  
- Some videos may not have transcripts available  
- Auto-generated transcripts may be noisy  
- `youtube-transcript-api` v1.0+ removed the old static methods (`get_transcript`, `list_transcripts`) — ensure you are on the latest version: `pip install -U youtube-transcript-api`  

---

## 🧠 Future Improvements

- Whisper-based transcription (avoid API blocking)  
- Timestamp-based answers  
- Source chunk highlighting  
- Full multilingual support  
- Caching for faster performance  

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.


---

## 👨‍💻 Author

Kavish Gupta

---

## ⭐ Support

If you like this project, consider giving it a star!