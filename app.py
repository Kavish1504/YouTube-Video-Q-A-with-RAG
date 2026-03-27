import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="YouTube RAG", page_icon="🎬", layout="wide")
st.title("🎬 YouTube Video Q&A with RAG")
st.caption("Paste a YouTube URL, get a summary, then ask anything about the video.")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore=None

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

if "summary" not in st.session_state:
    st.session_state.summary=None

def load_and_index(url:str):
    with st.spinner("⏳ Loading transcript..."):
        loader=YoutubeLoader.from_youtube_url(url,add_video_info=False,language=["hi","en"])
        docs=loader.load()

    with st.spinner("Chunking text"):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=text_splitter.split_documents(docs)

    with st.spinner("🔢 Creating embeddings locally (first run downloads ~90MB model)..."):
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db=FAISS.from_documents(chunks,embeddings)

    return db,chunks

def summarize(chunks,llm):
    sample_text=[c.page_content for c in chunks[:5]]
    prompt= f"""
            You are a helpful assistant. Given the following transcript excerpt from a YouTube video,
            write a concise 5-bullet summary covering the main topics discussed.
            
            Transcript:
            {sample_text}
            
            Summary (5 bullets):"""
    return llm.invoke(prompt).content

def qa_chain(db,model_name):
    llm=ChatGroq(model=model_name,temperature=0)
    retriever=db.as_retriever(search_kwargs={"k":4})

    prompt_template=PromptTemplate(
        input_variables=["context","question"],
        template="""You are an expert assistant helping users understand a YouTube video.

            IMPORTANT:
            - Always answer in English.
            - If the context is in another language, translate it to English before answering.


            Context:
            {context}

            Question: {question}

            Answer (in English):""",
    )
    chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff", 
        chain_type_kwargs={"prompt":prompt_template}
    )
    return chain,llm

with st.sidebar:
    st.header("Setup")
    api_key=st.text_input("GROQ-API_KEY",value=os.getenv("GROQ_API_KEY"),type="password")

    if api_key:
        os.environ["GROQ_API_KEY"]=api_key

    model_name=st.selectbox(
         "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
    )

    st.divider()
    url=st.text_input("Youtube Video URL",placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Load and Index",use_container_width=True):
        if not api_key:
            st.error("Please enter your GROQ Api key")
        elif not url:
            st.error("Please enter the URL of the video")
        else:
            try:
                db,chunks=load_and_index(url=url)
                chain,llm=qa_chain(db,model_name=model_name)
                st.session_state.vectorstore=db
                st.session_state.chain=chain
                st.session_state.summary=summarize(chunks=chunks,llm=llm)
                st.session_state.chat_history=[]
                st.success(f"✅ Indexed {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.vectorstore:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.rerun()

if st.session_state.summary:
    with st.expander("Auto generated summary",expanded=True):
        st.markdown(st.session_state.summary)

if st.session_state.vectorstore:
    st.subheader("Ask anything about video")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question:=st.chat_input("e.g. What are the main takeaways?"):
        st.session_state.chat_history.append({"role":"user","content":question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result=st.session_state.chain.invoke({"query":question})
                answer=result["result"]

            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Enter a YouTube URL in the sidebar and click **Load & Index** to get started.")









