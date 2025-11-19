import requests
import json
import os
from .resonant_rag import ResonantRAG
from dotenv import load_dotenv, set_key

# --- ADDED FOR OPENAI ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("OpenAI library not found. Install with: pip install openai")

# Load existing keys
load_dotenv()

class ResonantAgent:
    def __init__(self):
        print("Initializing Agent v3.6 (Multi-Backend)...")
        self.rag = ResonantRAG()
        
        # Connection settings
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
        self.active_model = "llama3"
        
        # Init OpenAI Client
        self.openai_client = None
        self._init_openai()
        
        # INTERNAL CONVERSATION MEMORY
        self.chat_history = [] 
        self.max_history_len = 10

    def _init_openai(self):
        """Helper to initialize/refresh OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI and api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def set_model(self, model_name):
        self.active_model = model_name

    def save_api_key(self, key_name, key_value):
        """Securely save key to .env file and refresh client"""
        env_path = ".env"
        if not os.path.exists(env_path):
            open(env_path, 'a').close()
        
        set_key(env_path, key_name, key_value)
        os.environ[key_name] = key_value
        
        # If we saved OpenAI key, refresh client
        if key_name == "OPENAI_API_KEY":
            self._init_openai()
            
        return True

    def clear_chat_history(self):
        self.chat_history = []

    def query_llm(self, user_input, context_data=None, system_persona="", use_history=True):
        """
        Main brain. Supports both Ollama and OpenAI.
        """
        
        # 1. PREPARE SYSTEM INSTRUCTION (System Prompt)
        base_system = system_persona if system_persona.strip() else "You are a helpful AI assistant."
        
        if context_data:
            base_system += f"\n\nCONTEXT FROM KNOWLEDGE BASE:\n{chr(10).join(context_data)}"
            base_system += "\n\nInstructions: Answer based primarily on the context above."

        # 2. CHECK: Are we using OpenAI?
        # If model name starts with "gpt" (e.g. gpt-4o, gpt-3.5-turbo), use OpenAI
        if self.active_model.lower().startswith("gpt"):
            return self._query_openai(user_input, base_system, use_history)
        else:
            return self._query_ollama(user_input, base_system, use_history)

    def _query_openai(self, user_input, base_system, use_history):
        if not self.openai_client:
            return "Error: OpenAI API Key missing. Please enter it in the sidebar."

        # Prepare messages in OpenAI format
        messages = [{"role": "system", "content": base_system}]
        
        if use_history and self.chat_history:
            # Convert history ("User", "text") to format {"role": "user", "content": "..."}
            for role, text in self.chat_history[-self.max_history_len:]:
                role_lower = "user" if role == "User" else "assistant"
                messages.append({"role": role_lower, "content": text})
        
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.openai_client.chat.completions.create(
                model=self.active_model,
                messages=messages,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            
            if use_history:
                self.chat_history.append(("User", user_input))
                self.chat_history.append(("Assistant", reply))
            
            return reply
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    def _query_ollama(self, user_input, base_system, use_history):
        # Existing logic for Ollama (unchanged, just moved to its own function)
        history_str = ""
        if use_history and self.chat_history:
            history_str = "\n\nCONVERSATION HISTORY:\n"
            for role, text in self.chat_history[-self.max_history_len:]:
                history_str += f"{role}: {text}\n"

        full_prompt = f"System: {base_system}{history_str}\nUser: {user_input}"

        payload = {
            "model": self.active_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            if response.status_code == 200:
                reply = response.json().get('response', '')
                if use_history:
                    self.chat_history.append(("User", user_input))
                    self.chat_history.append(("Assistant", reply))
                return reply
            else:
                return f"Ollama Error: {response.text}"
        except Exception as e:
            return f"Connection Error: {e}"