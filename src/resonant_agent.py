import requests
import json
import os
from .resonant_rag import ResonantRAG
from dotenv import load_dotenv, set_key

# Load existing keys
load_dotenv()

class ResonantAgent:
    def __init__(self):
        print("Initializing Agent v3.5...")
        self.rag = ResonantRAG()
        
        # Connection settings
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
        self.active_model = "llama3"
        
        # INTERNAL CONVERSATION MEMORY (Context)
        # This is the "separate database" you wanted
        self.chat_history = [] 
        self.max_history_len = 10  # How many previous messages to remember (to not overload model)

    def set_model(self, model_name):
        self.active_model = model_name

    def save_api_key(self, key_name, key_value):
        """Securely save key to .env file"""
        env_path = ".env"
        if not os.path.exists(env_path):
            open(env_path, 'a').close()  # Create if doesn't exist
        
        # Write to file
        set_key(env_path, key_name, key_value)
        # Refresh current environment (so no restart needed)
        os.environ[key_name] = key_value
        return True

    def clear_chat_history(self):
        self.chat_history = []

    def query_llm(self, user_input, context_data=None, system_persona="", use_history=True):
        """
        Main brain. Constructs prompt from:
        1. Persona (Filter)
        2. RAG Data (Memory)
        3. Conversation History (Context)
        4. Current Question
        """
        
        # A. CONSTRUCTING SYSTEM INSTRUCTION (FILTER)
        # If user didn't enter anything, use default
        base_system = system_persona if system_persona.strip() else "You are a helpful AI assistant."
        
        # If we have RAG data, add it to system part
        if context_data:
            base_system += f"\n\nCONTEXT FROM KNOWLEDGE BASE:\n{chr(10).join(context_data)}"
            base_system += "\n\nInstructions: Answer based primarily on the context above."

        # B. CONSTRUCTING HISTORY (CHAT MEMORY)
        history_str = ""
        if use_history and self.chat_history:
            history_str = "\n\nCONVERSATION HISTORY:\n"
            for role, text in self.chat_history[-self.max_history_len:]:
                history_str += f"{role}: {text}\n"

        # C. FINAL PROMPT
        # Ollama format (System + History + User)
        full_prompt = f"System: {base_system}{history_str}\nUser: {user_input}"

        # D. MODEL CALL
        payload = {
            "model": self.active_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.7}  # Can add to settings
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            if response.status_code == 200:
                reply = response.json().get('response', '')
                
                # Save to conversation memory
                if use_history:
                    self.chat_history.append(("User", user_input))
                    self.chat_history.append(("Assistant", reply))
                
                return reply
            else:
                return f"Ollama Error: {response.text}"
        except Exception as e:
            return f"Connection Error: {e}"

