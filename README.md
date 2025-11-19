# 🐉 DragonMemory

**Neural Memory Compression System for RAG Applications**

DragonMemory is a production-ready system that compresses embeddings using a neural architecture (Dragon v7) to achieve 16:1 compression ratios while maintaining semantic accuracy above 90%.

## ✨ Features

- **16:1 Compression Ratio**: Compress 384-dimensional embeddings to 24 dimensions
- **90%+ Semantic Accuracy**: Maintains high retrieval quality after compression
- **RAG Integration**: Seamless integration with retrieval-augmented generation
- **Streamlit GUI**: User-friendly interface for document processing and chat
- **Multi-format Support**: PDF, DOCX, TXT, MD document processing
- **Audio Transcription**: Whisper integration for audio-to-text processing
- **Persistent Memory**: Save and load knowledge bases

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DragonMemory

# Install dependencies
pip install -r requirements.txt

# Download the pre-trained model (if not included)
# Place dragon_pro_1_16.pth in models/ directory
```

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```env
OLLAMA_BASE_URL=http://localhost:11434
```

3. Make sure Ollama is running locally with your preferred model (llama3, mistral, etc.)

### Running the Application

```bash
streamlit run gui_app.py
```

## 📁 Project Structure

```
DragonMemory/
├── assets/                  # Images for README (logo, GUI screenshots)
│   └── dragon_gui_preview.png
├── src/                     # Main code
│   ├── __init__.py
│   ├── memory_v3_model.py   # Architecture (Dragon v7)
│   ├── resonant_rag.py      # RAG logic + Persistence
│   └── resonant_agent.py    # Agent logic + Chat History
├── models/                  # Model weights location
│   └── dragon_pro_1_16.pth  # Pre-trained model
├── gui_app.py               # Main Streamlit application
├── requirements.txt         # Dependencies
├── .gitignore               # Git ignore rules
├── .env.example             # Environment template
├── LICENSE                  # AGPLv3
├── README.md                # This file
└── WHITE_PAPER.md           # Technical documentation
```

## 🧠 Architecture

### Dragon v7: Light Resonant Architecture

- **Compression Ratio**: 16:1 (128 tokens → 8 tokens)
- **Architecture**: Multi-phase resonant pointer with neighbor mixing
- **Normalization**: LayerNorm for stable training
- **Performance**: 90.4% semantic reconstruction accuracy

See `WHITE_PAPER.md` for detailed technical documentation.

## 📖 Usage

### Adding Documents

1. Navigate to the **Documents** tab
2. Upload PDF, DOCX, TXT, or MD files
3. Click **Process Documents**
4. Documents are automatically chunked and compressed

### Chat Interface

1. Navigate to the **Chat** tab
2. Ask questions about your documents
3. The system retrieves relevant context using compressed embeddings
4. Responses are generated using your local Ollama model

### Audio Processing

1. Navigate to the **Audio** tab
2. Upload MP3, WAV, or M4A files
3. Click **Start Transcription**
4. Review and save transcriptions to memory

## 🔧 Configuration

### Model Selection

In the sidebar, select your preferred Ollama model:
- llama3
- mistral
- gemma
- gpt-4o (requires API key)

### System Persona

Customize the assistant's behavior in the **Filter & Persona** section.

### Memory Management

- **Save DB**: Persist knowledge base to disk
- **Load DB**: Restore knowledge base from disk
- **Clear Conversation**: Reset chat history

## 📊 Performance

- **Compression**: 16:1 ratio (384D → 24D)
- **Semantic Accuracy**: 90.4% cosine similarity
- **Retrieval Recall**: 85%+ @k=3
- **Inference Speed**: <10ms per query

On internal tests, DragonMemory achieves self-recall@1 = 1.0 and partial-recall@3 
between 0.6–1.0 depending on the corpus (technical reports vs. long-form literature)

## 🤝 Contributing

Contributions are welcome! Please read the LICENSE file for details on our code of conduct.

## 📄 License

DragonMemory (code in this repository) is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

In practice, this means:

- ✅ You can freely use, study, modify and run this software for personal, research, and commercial purposes.
- ✅ You can deploy it on your own machines or internal infrastructure.
- 🔁 If you modify this software and make it available to users over a network (e.g. as a SaaS/API/web service), you must provide the corresponding source code of your modified version to those users under the same AGPL-3.0 license.
- ❌ You cannot take this code, build a closed-source network service around it, and deny users access to the modified source.

If you are a company and would like to use DragonMemory inside a proprietary, closed-source product or SaaS without the obligations of AGPL-3.0, please contact the author to discuss a separate commercial license or support agreement.

## 🙏 Acknowledgments

- Sentence Transformers for embedding models
- Ollama for local LLM inference
- Streamlit for the GUI framework
- PyTorch for the neural architecture

## 📧 Contact

For questions or support, please open an issue on the repository.

