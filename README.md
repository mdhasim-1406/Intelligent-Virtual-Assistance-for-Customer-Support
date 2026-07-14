<div align="center">
  <h1>🤖 Project Kural</h1>
  <p><strong>An Adaptive, Multilingual AI Customer Service Agent</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python" alt="Python 3.9+"/>
    <img src="https://img.shields.io/badge/LLM-OpenRouter-FF6F00?style=flat" alt="OpenRouter LLM"/>
    <img src="https://img.shields.io/badge/Vector_Store-FAISS-00C853?style=flat" alt="FAISS"/>
    <img src="https://img.shields.io/badge/UI-Gradio-FF6B6B?style=flat" alt="Gradio UI"/>
    <img src="https://img.shields.io/badge/TTS-gTTS-4285F4?style=flat" alt="gTTS"/>
    <img src="https://img.shields.io/badge/STT-Whisper-FFD700?style=flat" alt="Whisper STT"/>
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License MIT"/>
  </p>
</div>

---

## 📋 Overview

**Project Kural** (குறள்) is a cognitive AI customer service agent that goes beyond conventional chatbots. Named after the ancient Tamil literary work *Thirukkural* — emphasizing virtue, ethics, and compassionate communication — this system delivers **emotionally intelligent, context-aware, and multilingual customer support**.

At its core, Kural dynamically adapts its communication persona based on detected user sentiment, maintains persistent memory across sessions, and provides voice-based interaction in English, Tamil, and Hindi — making it a truly inclusive solution for global customer service operations.

---

## ✨ Key Capabilities

| Capability | Description |
|---|---|
| **🎭 Adaptive Persona Engine** | Dynamically switches between three personas — empathetic de-escalation, efficient-friendly, and professional-direct — based on real-time sentiment analysis of user input |
| **🧠 Long-Term Memory** | Persistent JSON-backed conversation history with per-user context retention across sessions, including summarization and recall |
| **🔍 Vector Knowledge Base** | FAISS-powered semantic search over **26,800+ real customer service interactions** for accurate, contextually grounded responses |
| **🌐 Multilingual Voice I/O** | Speech-to-text via OpenAI Whisper and text-to-speech via gTTS, with automatic language detection for English, Tamil, and Hindi |
| **🛠️ Tool Integration** | Extensible tool-calling framework for real-time data access — billing lookups, network status checks, and custom API integrations |
| **⚡ LLM-Powered Reasoning** | Backed by OpenRouter's LLM infrastructure (Gemini Flash 1.5 default) for fast, coherent, and context-rich responses |

---

## 🧱 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                     │
│          Chat UI · Audio I/O · Persona Indicator             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Perception Module                         │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │  Whisper STT     │  │  Sentiment Analysis (OpenRouter)  │  │
│  │  (speech→text)   │  │  (frustrated/positive/neutral)    │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Kural Agent (Orchestrator)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │  Persona  │ │  Memory  │ │  Tools   │ │  Vector Store  │ │
│  │  Manager  │ │  Module  │ │  Layer   │ │  (FAISS)       │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
│                                                              │
│  OpenRouter API ────► LLM (Gemini Flash 1.5)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Data Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Training CSV    │  │  Telecom FAQ    │                   │
│  │  (26,800+ ints)  │  │  Knowledge Base │                   │
│  └─────────────────┘  └─────────────────┘                   │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  User DB (JSON)  │  │  Persona Config │                   │
│  │  (per-user mem)  │  │  (3 personas)   │                   │
│  └─────────────────┘  └─────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
project-kural/
├── app.py                          # Main Gradio application entry point
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys)
│
├── core/                          # Core backend modules
│   ├── __init__.py
│   ├── agent.py                  # Main AI agent orchestrator (KuralAgent)
│   ├── memory.py                 # Conversation memory management
│   ├── perception.py             # Speech-to-text & sentiment analysis
│   ├── tools.py                  # External API integration tools
│   └── vector_store.py           # FAISS knowledge base & semantic search
│
├── personas/                      # Adaptive personality prompts
│   ├── empathetic_deescalation.txt   # For frustrated / upset customers
│   ├── efficient_friendly.txt        # For positive / satisfied customers
│   └── professional_direct.txt       # For neutral / businesslike interactions
│
├── knowledge_base/               # Domain-specific knowledge
│   └── telecom_faq.txt           # Telecom FAQ reference
│
├── training_data/                # Vector search corpus
│   └── Intelligent Virtual Assistants for Customer Support (1).csv
│                                 # 26,800+ real customer service interactions
│
├── user_database/                # Persistent user memory
│   └── users.json
│
├── tests/                        # Test suite
│   ├── test_core_logic.py
│   └── test_app_logic.py
│
└── TROUBLESHOOTING.md            # Setup & runtime troubleshooting guide
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** — [python.org](https://python.org)
- **FFmpeg** — Required by Whisper for audio processing
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  # macOS
  brew install ffmpeg
  ```
- **OpenRouter API Key** — [openrouter.ai](https://openrouter.ai)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/mdhasim-1406/Intelligent-Virtual-Assistance-for-Customer-Support.git
cd Intelligent-Virtual-Assistance-for-Customer-Support/project-kural

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 5. (Optional) Download embeddings model for local vector search
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# 6. Launch the application
python app.py
```

### 🐳 Git LFS Note

The embeddings model (`all-MiniLM-L6-v2`) uses Git LFS for large model files. Ensure Git LFS is set up:

```bash
sudo apt-get install git-lfs   # Linux
brew install git-lfs           # macOS
git lfs install
```

After cloning the model repo, verify file sizes are correct (~91 MB each):
```bash
ls -lh all-MiniLM-L6-v2/model.safetensors
```

---

## 🎭 Adaptive Persona System

Kural intelligently detects user sentiment and selects the appropriate communication persona:

| Sentiment | Persona | Behavior |
|---|---|---|
| **Frustrated / Upset** | 🟠 Empathetic De-escalation | Validates emotions, apologizes sincerely, speaks calmly, takes ownership |
| **Satisfied / Positive** | 🟢 Efficient & Friendly | Matches enthusiasm, proactive suggestions, quick solutions, cheerful tone |
| **Neutral / Businesslike** | 🔵 Professional & Direct | Clear, structured instructions; concise; task-focused; formal etiquette |

---

## 🧪 Running Tests

```bash
cd project-kural
pytest tests/ -v                    # Run all tests
pytest tests/test_core_logic.py     # Run core logic tests
pytest tests/ -k "test_name"        # Run specific test
```

---

## 📊 Dataset

The system is trained on a comprehensive dataset of **26,800+ real customer service interactions** covering intents such as:
- Order cancellation and modification
- Billing inquiries and disputes
- Technical support and troubleshooting
- Account management
- General inquiries

The data is ingested into a **FAISS vector index** for efficient semantic similarity search at query time.

---

## 🛣️ Roadmap

- [ ] **Streaming responses** for reduced latency
- [ ] **Multi-turn tool execution** with state tracking
- [ ] **Database-backed memory** (PostgreSQL / SQLite) replacing JSON storage
- [ ] **Admin dashboard** for monitoring conversations and persona performance
- [ ] **Custom persona builder** for enterprise branding
- [ ] **WhatsApp / Telegram bot integration**

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please start a discussion first.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ for intelligent, compassionate customer service</sub>
</div>
