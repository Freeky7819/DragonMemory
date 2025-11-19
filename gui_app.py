import streamlit as st
import time
import pandas as pd
import os
import tempfile
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.resonant_agent import ResonantAgent

# --- IMPORTS ---
try: from pypdf import PdfReader
except: pass
try: from docx import Document
except: pass
try: import whisper
except: pass

# --- CONFIGURATION ---
st.set_page_config(page_title="Dragon Brain v3.5", page_icon="🐉", layout="wide")

# --- STYLES ---
st.markdown("""
<style>
    .stTextArea textarea { font-size: 14px; font-family: monospace; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- INIT ---
if "agent" not in st.session_state:
    try:
        st.session_state.agent = ResonantAgent()
        st.session_state.messages = []
        # Whisper
        if 'whisper_model' not in st.session_state:
            try:
                import whisper
                st.session_state.whisper_model = whisper.load_model("base", device="cuda" if st.session_state.agent.rag.device.type == "cuda" else "cpu")
            except: st.session_state.whisper_model = None
    except Exception as e: st.error(f"Init Error: {e}")

agent = st.session_state.agent

# ==========================================
# SIDEBAR: CONTROL CENTER
# ==========================================
with st.sidebar:
    st.title("Control Panel")
    
    # --- 1. MODEL & API ---
    with st.expander("Model & API Keys", expanded=True):
        model_option = st.selectbox("Model", ["llama3", "mistral", "gemma", "gpt-4o"], index=0)
        if st.button("Apply Model"):
            agent.set_model(model_option)
            st.toast(f"Activated: {model_option}")
        
        st.markdown("---")
        # API key input (Secure storage)
        api_key_input = st.text_input("New API Key (OpenAI/Anthropic)", type="password", placeholder="sk-...")
        key_type = st.selectbox("Key Type", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
        
        if st.button("Save to .env"):
            if api_key_input:
                agent.save_api_key(key_type, api_key_input)
                st.success(f"Key {key_type} saved! Field will be cleared.")
                time.sleep(1)
                st.rerun()  # This clears the input field

    # --- 2. FILTER (PERSONA) ---
    with st.expander("Filter & Persona", expanded=True):
        st.caption("How should the model behave?")
        system_persona = st.text_area(
            "System Instructions",
            value=st.session_state.get("system_persona", "You are a helpful, precise, and slightly witty assistant. You love using metaphors."),
            height=150,
            help="Define personality and behavior rules here."
        )
        st.session_state["system_persona"] = system_persona
    
    # --- 3. CONVERSATION MEMORY ---
    with st.expander("Contextual Memory", expanded=True):
        use_history = st.toggle("Remember conversation", value=st.session_state.get("use_history", True))
        st.session_state["use_history"] = use_history
        
        if st.button("Clear Conversation"):
            agent.clear_chat_history()
            st.session_state.messages = []  # Clear GUI display as well
            st.toast("Tabula Rasa! (Memory cleared)")
            time.sleep(0.5)
            st.rerun()

    # --- 4. RAG DATABASE ---
    st.divider()
    st.caption(f"RAG Database: {len(agent.rag.memory_vectors)} units")
    c1, c2 = st.columns(2)
    if c1.button("Save DB"):
        ok, msg = agent.rag.save_knowledge_base()
        if ok: st.toast(msg)
        else: st.error(msg)
    if c2.button("Load DB"):
        ok, msg = agent.rag.load_knowledge_base()
        if ok: st.toast(msg)
        else: st.error(msg)

# ==========================================
# MAIN FRAME
# ==========================================
tab_chat, tab_docs, tab_audio = st.tabs(["Chat", "Documents", "Audio"])

# --- TAB 1: CHAT ---
with tab_chat:
    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]: st.info(s[:300]+"...")

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. RAG Search
            sources = agent.rag.search(prompt, k=3)
            
            # 2. LLM Call (with all new parameters)
            with st.spinner("Thinking..."):
                response = agent.query_llm(
                    prompt, 
                    context_data=sources if sources else None,
                    system_persona=st.session_state.get("system_persona", "You are a helpful assistant."),  # Your filter
                    use_history=st.session_state.get("use_history", True)  # Your memory
                )
            
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "sources": sources if sources else None
            })

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def extract_text_from_file(file):
    ftype = file.name.split('.')[-1].lower()
    if ftype == 'pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(file)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        except: return "Error reading PDF"
    elif ftype == 'docx':
        try:
            from docx import Document
            doc = Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        except: return "Error reading DOCX"
    else:
        return file.read().decode("utf-8")

# ---------------------------------------------------------
# TAB 2: DOCUMENTS (Text section)
# ---------------------------------------------------------
with tab_docs:
    st.header("Knowledge Base (Text)")
    st.write("Upload PDF, DOCX or TXT files.")
    
    uploaded_files = st.file_uploader("Select documents", accept_multiple_files=True, type=['txt', 'md', 'pdf', 'docx'])
    
    if uploaded_files:
        if st.button("Process Documents"):
            bar = st.progress(0)
            count = 0
            for i, file in enumerate(uploaded_files):
                raw_text = extract_text_from_file(file)
                if raw_text and "Error" not in raw_text:
                    chunks = chunk_text(raw_text)
                    for chunk in chunks:
                        agent.rag.add_memory(f"Source: {file.name}\n{chunk}")
                    count += 1
                bar.progress((i + 1) / len(uploaded_files))
            
            if count > 0:
                st.success(f"Successfully loaded {count} files!")
                time.sleep(1)
                st.rerun()

# ---------------------------------------------------------
# TAB 3: AUDIO STUDIO
# ---------------------------------------------------------
with tab_audio:
    st.header("Audio Studio")
    st.write("Audio transcription and speech search.")
    
    audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'm4a'])
    
    if audio_file:
        # Display audio player
        st.audio(audio_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Start Transcription", type="primary"):
                if not st.session_state.get('whisper_model'):
                    st.error("Whisper model not loaded (missing ffmpeg or initialization error).")
                else:
                    with st.spinner("Listening and transcribing (Whisper)..."):
                        # Temp file trick for Windows
                        suffix = f".{audio_file.name.split('.')[-1]}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(audio_file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            result = st.session_state.whisper_model.transcribe(tmp_path)
                            # Save result
                            st.session_state['last_transcript'] = result["text"]
                            st.session_state['last_audio_name'] = audio_file.name
                        except Exception as e:
                            st.error(f"Transcription error: {e}")
                        finally:
                            try: os.remove(tmp_path)
                            except: pass
                        
                        st.success("Complete!")
                        st.rerun()

    st.divider()

    # Display transcript (if exists in session memory)
    if 'last_transcript' in st.session_state:
        st.subheader(f"Transcript: {st.session_state.get('last_audio_name', 'Unknown')}")
        
        # Large window for editing/review
        transcript_text = st.text_area(
            "Content (you can edit before saving)", 
            value=st.session_state['last_transcript'], 
            height=300
        )
        
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Save to RAG Memory"):
                with st.spinner("Compressing with Dragon..."):
                    chunks = chunk_text(transcript_text, size=800, overlap=100)
                    for chunk in chunks:
                        agent.rag.add_memory(f"Audio Source: {st.session_state['last_audio_name']}\n{chunk}")
                st.success("Saved! You can now ask about this in Chat.")
                time.sleep(1.5)
                st.rerun()

