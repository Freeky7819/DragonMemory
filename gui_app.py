import streamlit as st
import time
import pandas as pd
import os
import tempfile
import sys
import json
import uuid  # ADDED: For reliable temporary IDs

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
st.set_page_config(page_title="DragonMemory v.2", page_icon="🐉", layout="wide")

# --- STYLES ---
st.markdown("""
<style>
    .stTextArea textarea { font-size: 14px; font-family: monospace; }
    .block-container { padding-top: 2rem; }
    .stChatInput { padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# --- INIT ---
# Global settings
if "compression_ratio" not in st.session_state: st.session_state["compression_ratio"] = 16
if "system_persona" not in st.session_state: st.session_state["system_persona"] = "You are a helpful, precise, and slightly witty assistant. You love using metaphors."
if "use_history" not in st.session_state: st.session_state["use_history"] = True
if "show_thoughts" not in st.session_state: st.session_state["show_thoughts"] = False
if "save_int8_default" not in st.session_state: st.session_state["save_int8_default"] = False
if "save_float16_default" not in st.session_state: st.session_state["save_float16_default"] = False

# Session Management
if "all_sessions" not in st.session_state: 
    st.session_state.all_sessions = {"Default Session": []}
if "current_session_id" not in st.session_state: 
    st.session_state.current_session_id = "Default Session"
# NEW: State for naming session
if "naming_new_session" not in st.session_state: st.session_state["naming_new_session"] = False
    

# Functions for access/synchronization
def get_current_messages():
    return st.session_state.all_sessions[st.session_state.current_session_id]

def set_current_messages(messages):
    st.session_state.all_sessions[st.session_state.current_session_id] = messages

# Function to synchronize agent's internal memory
def sync_agent_history(current_agent):
    current_messages = get_current_messages()
    # Agent stores only role/content, not sources/thoughts from GUI
    current_agent.chat_history = [(m['role'], m['content']) for m in current_messages if m['role'] in ('user', 'assistant')]

if "agent" not in st.session_state:
    try:
        ratio = st.session_state.get("compression_ratio", 16)
        # Agent initialization
        st.session_state.agent = ResonantAgent(ratio=ratio) 
        
        # Whisper model
        if 'whisper_model' not in st.session_state:
            try:
                import whisper
                st.session_state.whisper_model = whisper.load_model("base", device="cuda" if st.session_state.agent.rag.device.type == "cuda" else "cpu")
            except: st.session_state.whisper_model = None
        
        sync_agent_history(st.session_state.agent)
        
    except Exception as e: 
        st.error(f"Init Error: {e}") 
        st.session_state.agent = None  # On error, set to None

# ERROR FIX: Add agent and stop execution if initialization failed
agent = st.session_state.agent

if agent is None:
    st.stop()



# ==========================================
# SIDEBAR: QUICK ACCESS & SESSION MANAGEMENT
# ==========================================
with st.sidebar:
    st.title("Dragon Brain v3.9")
    
    # --- 1. SESSION MANAGEMENT ---
    st.subheader("Chat Sessions")
    
    # Logic for naming session
    if st.session_state.naming_new_session:
        # Display input field for naming
        new_name_input = st.text_input("Enter new session name:", key="new_session_name_input_key")
        
        col_ok, col_cancel = st.columns(2)
        with col_ok:
            if st.button("Create Session"):
                if new_name_input and new_name_input not in st.session_state.all_sessions:
                    st.session_state.all_sessions[new_name_input] = []
                    st.session_state.current_session_id = new_name_input
                    agent.clear_chat_history()
                    st.session_state.naming_new_session = False
                    st.rerun()
                elif new_name_input in st.session_state.all_sessions:
                    st.warning("Session name already exists.")
                else:
                    st.warning("Please enter a valid name.")
        with col_cancel:
            if st.button("Cancel"):
                st.session_state.naming_new_session = False
                st.rerun()
    
    # Button for new conversation
    if st.button("➕ New Chat", use_container_width=True, disabled=st.session_state.naming_new_session):
        st.session_state.naming_new_session = True
        st.rerun()

    # Session Selector
    session_id = st.selectbox(
        "Select Session", 
        list(st.session_state.all_sessions.keys()), 
        index=list(st.session_state.all_sessions.keys()).index(st.session_state.current_session_id),
        key='session_selector'
    )
    
    # Logic for switching conversation
    if session_id != st.session_state.current_session_id:
        st.session_state.current_session_id = session_id
        sync_agent_history(agent)  # Load selected history into LLM agent
        st.toast(f"Switched to: {session_id}")
        st.rerun()
        
    # --- 2. MODEL SELECT ---
    st.markdown("---")
    with st.expander("Active Model", expanded=True):
        model_option = st.selectbox("LLM Backend", ["llama3", "mistral", "gemma", "gpt-4o"], index=0)
        if st.button("Apply Model", key="apply_model_btn"):
            agent.set_model(model_option)
            st.toast(f"Activated: {model_option}")
    
    # --- 3. RAG DATABASE (Quick Save/Load) ---
    st.divider()
    st.caption(f"RAG Database: {len(agent.rag.memory_vectors)} units (1:{st.session_state['compression_ratio']})")
    c1, c2 = st.columns(2)
    if c1.button("Save DB"):
        ok, msg = agent.rag.save_knowledge_base(
            use_int8=st.session_state.get("save_int8_default", False),
            use_float16=st.session_state.get("save_float16_default", False)
        )
        if ok: st.toast(msg)
        else: st.error(msg)
    if c2.button("Load DB"):
        ok, msg = agent.rag.load_knowledge_base()
        if ok: st.toast(msg)
        else: st.error(msg)




# ==========================================
# MAIN FRAME
# ==========================================
tab_chat, tab_docs, tab_audio, tab_settings = st.tabs(["Chat", "Documents", "Audio", "Settings"])




# --- TAB 1: CHAT ---
with tab_chat:
    
    # Control elements for transparency and memory (Top of Chat window)
    col_history, col_thoughts, col_save = st.columns([1.5, 1.5, 1])
    
    with col_history:
        st.session_state["use_history"] = st.toggle(
            "LLM Short-Term History", 
            value=st.session_state.get("use_history", True),
            help="If ON, LLM remembers the last few messages in the current session."
        )
    
    with col_thoughts:
        st.session_state["show_thoughts"] = st.toggle(
            "Show RAG/AI Thoughts", 
            value=st.session_state.get("show_thoughts", False),
            help="Displays RAG sources and LLM's internal reasoning (if model supports it)."
        )
        
    with col_save:
        if st.button("💾 Save Chat to RAG", use_container_width=True):
            current_messages = get_current_messages()
            if current_messages:
                full_context = ""
                for i, msg in enumerate(current_messages):
                    full_context += f"Source: Chat Session {st.session_state.current_session_id} Turn {i+1}\n{msg['role']}: {msg['content']}\n---\n"
                
                agent.rag.add_memory(full_context)
                st.toast(f"Saved {len(current_messages)} turns from {st.session_state.current_session_id} to RAG memory!")
            else:
                st.warning("Chat is empty, nothing to save.")
                
    st.divider() 

    # Display conversation history
    for msg in get_current_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if st.session_state["show_thoughts"]:
                if msg.get("sources"):
                    with st.expander("RAG Sources"):
                        for s in msg["sources"]: st.info(s[:300]+"...")
                
                if msg.get("thoughts"):
                    with st.expander("AI Thoughts"):
                        st.code(msg["thoughts"], language='markdown')




    if prompt := st.chat_input(f"Ask in {st.session_state.current_session_id}..."):
        # 1. Add user input to current session
        current_messages = get_current_messages()
        current_messages.append({"role": "user", "content": prompt})
        set_current_messages(current_messages)
        
        # 2. Synchronize agent and GUI
        sync_agent_history(agent)
        
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. RAG Search
            sources = agent.rag.search(prompt, k=3)
            
            # 2. LLM Call
            with st.spinner("Thinking..."):
                response, thoughts = agent.query_llm(
                    prompt, 
                    context_data=sources if sources else None,
                    system_persona=st.session_state["system_persona"], 
                    use_history=st.session_state["use_history"],
                    show_thoughts=st.session_state["show_thoughts"]
                )
            
            # 3. Display
            if st.session_state["show_thoughts"]:
                if thoughts:
                    with st.expander("AI Thoughts"): st.code(thoughts, language='markdown')
                if sources:
                    with st.expander("RAG Sources"): 
                        for s in sources: st.info(s[:300]+"...")

            st.markdown(response)
            
            # 4. Save response, sources and thoughts
            current_messages.append({
                "role": "assistant", 
                "content": response, 
                "sources": sources if sources else None,
                "thoughts": thoughts if thoughts else None
            })
            set_current_messages(current_messages)
            st.rerun()  # Rerun to refresh window





# ==========================================
# HELPER FUNCTIONS (other functions remain the same)
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

# --- TAB 2: DOCUMENTS ---
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

# --- TAB 3: AUDIO STUDIO ---
with tab_audio:
    st.header("Audio Studio")
    st.write("Audio transcription and speech search.")
    
    audio_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'm4a'])
    
    if audio_file:
        st.audio(audio_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Start Transcription", type="primary"):
                if not st.session_state.get('whisper_model'):
                    st.error("Whisper model not loaded (missing ffmpeg or initialization error).")
                else:
                    with st.spinner("Listening and transcribing (Whisper)..."):
                        suffix = f".{audio_file.name.split('.')[-1]}"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(audio_file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            result = st.session_state.whisper_model.transcribe(tmp_path)
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

    if 'last_transcript' in st.session_state:
        st.subheader(f"Transcript: {st.session_state.get('last_audio_name', 'Unknown')}")
        
        transcript_text = st.text_area(
            "Content (you can edit before saving)", 
            value=st.session_state['last_transcript'], 
            height=300
        )
        
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Save to RAG Memory", key="audio_save_btn"):
                with st.spinner("Compressing with Dragon..."):
                    chunks = chunk_text(transcript_text, size=800, overlap=100)
                    for chunk in chunks:
                        agent.rag.add_memory(f"Audio Source: {st.session_state['last_audio_name']}\n{chunk}")
                st.success("Saved! You can now ask about this in Chat.")
                time.sleep(1.5)
                st.rerun()


# ==========================================
# TAB 4: SETTINGS
# ==========================================
with tab_settings:
    st.header("System Configuration")
    st.write("Adjust core model parameters and API settings.")

    # --- 1. LLM / API Configuration ---
    st.subheader("1. LLM & API Configuration")
    
    col_api, col_type = st.columns([2, 1])
    
    with col_api:
        api_key_input = st.text_input(
            "API Key (OpenAI/Anthropic)", 
            type="password", 
            placeholder="sk-...",
            key="new_api_key_input"
        )
    with col_type:
        key_type = st.selectbox("Key Type", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    
    if st.button("Save API Key to .env"):
        if api_key_input:
            agent.save_api_key(key_type, api_key_input)
            st.success(f"Key {key_type} saved! (Requires App Restart to fully load)")

    # --- 2. RAG & Compression Settings ---
    st.subheader("2. RAG & Compression Settings")
    
    col_ratio, col_mem = st.columns([1, 2])
    
    with col_ratio:
        new_ratio = st.number_input(
            "Compression Ratio (1:N)", 
            min_value=4, 
            max_value=64, 
            value=st.session_state["compression_ratio"], 
            step=4,
            help=f"Sequence length 128 tokens compressed to {128 // st.session_state['compression_ratio']} vectors."
        )
        if new_ratio != st.session_state["compression_ratio"]:
            st.session_state["compression_ratio"] = new_ratio
            st.warning("Ratio changed! Restart application to rebuild RAG backend.")


    with col_mem:
        st.caption("Default Save Format (Applies to 'Save DB' button)")
        
        st.session_state["save_int8_default"] = st.checkbox(
            "Use INT8 Quantization (Smallest Index)", 
            value=st.session_state["save_int8_default"],
            help="Saves memory vectors as 8-bit integers, reducing size by 75% compared to Float32."
        )
        st.session_state["save_float16_default"] = st.checkbox(
            "Use Float16 (Balanced Index)", 
            value=st.session_state["save_float16_default"],
            help="Saves memory vectors as 16-bit floats, reducing size by 50% compared to Float32 with minimal quality loss."
        )
        if st.session_state["save_int8_default"] and st.session_state["save_float16_default"]:
            st.warning("Only one format can be active. INT8 will be prioritized.")


    # --- 3. Persona / Filtering ---
    st.subheader("3. Assistant Persona")
    
    system_persona = st.text_area(
        "System Instructions",
        value=st.session_state["system_persona"],
        height=200,
        help="Define personality and behavior rules here. This acts as the System Prompt."
    )
    st.session_state["system_persona"] = system_persona
    
    # --- 4. Memory Deletion (Keyword/Source) - MOVED FROM CHAT
    st.subheader("4. Memory Deletion (Advanced)")
    
    delete_keyword = st.text_input(
        "Keyword/Source to Delete",
        placeholder="e.g. document_name.pdf or Chat Session 2",
        help="Deletes ALL memory units containing this keyword (case-insensitive)."
    )
    
    if st.button("Delete Memories by Keyword", type="secondary"):
        if delete_keyword:
            # Assumption: rag.delete_by_keyword is implemented
            deleted_count, msg = agent.rag.delete_by_keyword(delete_keyword)
            if deleted_count > 0:
                st.success(msg)
                time.sleep(1)
                st.rerun()
            else:
                st.info(msg)
        else:
            st.warning("Please enter a keyword to confirm deletion.")
