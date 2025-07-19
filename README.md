# Project Kural: An Adaptive, Multilingual AI Customer Service Agent

## Vision

Project Kural is not just another chatbot—it's a cognitive entity designed to revolutionize customer service interactions. By combining adaptive personas, long-term memory, and multilingual voice capabilities, Kural provides personalized, context-aware support that evolves with each conversation. Built on a foundation of 26,800+ real customer service interactions, it delivers accurate, empathetic, and professionally consistent responses across languages and cultures.

## Core Features

- **🎭 Adaptive Persona**: Dynamically changes its communication tone and style based on detected user sentiment (empathetic for frustrated customers, efficient for positive interactions, professional for neutral inquiries)
- **📊 Data-Driven Responses**: Leverages a comprehensive dataset of 26,800+ customer service interactions for accurate, high-quality answers using FAISS vector similarity search
- **🌍 Multilingual Voice I/O**: Supports seamless text and voice interactions in English, Tamil, and Hindi with automatic language detection
- **🧠 Long-Term Memory**: Maintains persistent conversation history and user context across sessions for personalized service
- **🔧 Tool Usage**: Integrates with external APIs and tools for real-time data access (billing information, network status, etc.)
- **⚡ Real-time Processing**: Powered by OpenRouter's LLM infrastructure for fast, reliable responses

## Tech Stack

- **Backend**: Python 3.9+, LangChain, OpenRouter API, FAISS Vector Database
- **Data Processing**: Pandas, NumPy, OpenAI Whisper (speech-to-text)
- **Frontend**: Streamlit, gTTS (text-to-speech)
- **Memory**: JSON-based persistent storage, conversation summarization
- **Voice**: OpenAI Whisper for transcription, gTTS for synthesis

## Project Structure

```
project-kural/
├── README.md
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys)
├── core/                          # Core backend modules
│   ├── __init__.py
│   ├── agent.py                   # Main AI agent orchestrator
│   ├── memory.py                  # Conversation memory management
│   ├── perception.py              # Speech & sentiment analysis
│   ├── tools.py                   # External API integration tools
│   └── vector_store.py            # Knowledge base vector database
├── knowledge_base/                # Additional knowledge files
│   └── telecom_faq.txt
├── personas/                      # Adaptive personality prompts
│   ├── efficient_friendly.txt
│   ├── empathetic_deescalation.txt
│   └── professional_direct.txt
├── training_data/                 # Core knowledge dataset
│   └── Intelligent Virtual Assistants for Customer Support (1).csv
├── user_database/                 # User conversation histories
│   └── users.json
└── tests/                         # Test suite
    └── test_core_logic.py
```

## Setup and Installation

### 🚨 CRITICAL PREREQUISITES

**Git LFS is NOT OPTIONAL** - The application depends on large model files that require Git Large File Storage (git-lfs). **Without proper git-lfs setup, the application will fail to start with cryptic errors.**

#### System Requirements
- **Python 3.9+** - Download from [python.org](https://python.org)
- **Git** - Download from [git-scm.com](https://git-scm.com)
- **OpenRouter API Key** - Get from [openrouter.ai](https://openrouter.ai)
- **FFmpeg** - Required for audio processing

#### MANDATORY: Git LFS Installation & Configuration

**⚠️ YOU MUST COMPLETE ALL STEPS BELOW BEFORE PROCEEDING**

1. **Install Git LFS System Package**
   ```bash
   # Linux (Ubuntu/Debian)
   sudo apt-get update && sudo apt-get install git-lfs
   
   # Linux (CentOS/RHEL)
   sudo yum install git-lfs
   
   # macOS
   brew install git-lfs
   
   # Windows
   # Download installer from: https://git-lfs.github.io/
   ```

2. **Initialize Git LFS for Your User Account**
   ```bash
   git lfs install
   ```
   
   **Important**: This command only needs to be run **once per user account**. It configures Git to use LFS globally.

3. **Verify Git LFS Installation**
   ```bash
   git lfs version
   # Should display version information like: git-lfs/3.4.0 (GitHub; linux amd64; go 1.20.3)
   ```

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/project-kural.git
   cd project-kural
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **🔧 CRITICAL: Clean Up Any Previous Failed Downloads**
   
   If you've previously attempted to download the model and encountered errors, you **must** clean up the corrupted files:
   
   ```bash
   # Remove any existing model directory
   rm -rf all-MiniLM-L6-v2
   
   # Clear any Git LFS cache (optional but recommended)
   git lfs prune
   ```

5. **Download the Embeddings Model** 
   ```bash
   # IMPORTANT: Run this command from the project root directory
   git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   ```
   
   **Expected Behavior**: You should see progress messages like:
   ```
   Cloning into 'all-MiniLM-L6-v2'...
   remote: Enumerating objects: 20, done.
   remote: Counting objects: 100% (20/20), done.
   ...
   Downloading model.safetensors (91.0 MB)
   ```

6. **🔍 MANDATORY: Verify the Download Was Successful**
   
   **This is the most critical step** - verify that the large files were actually downloaded:
   
   ```bash
   ls -lh all-MiniLM-L6-v2/
   ```
   
   **✅ SUCCESS INDICATORS - You must see:**
   - `model.safetensors` with size approximately **91M** (not 133B or similar small size)
   - `pytorch_model.bin` with size approximately **91M**
   - Multiple files totaling ~200MB+
   
   **❌ FAILURE INDICATORS - If you see:**
   - `model.safetensors` with size of a few hundred bytes (e.g., 133B, 256B)
   - Files containing text starting with "version https://git-lfs.github.com/spec/v1"
   - Total directory size under 10MB
   
   **If you see failure indicators, the download failed. You must:**
   1. Re-check your git-lfs installation: `git lfs version`
   2. Delete the directory: `rm -rf all-MiniLM-L6-v2`
   3. Re-run the git clone command

7. **Configure API Key**
   
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add your OpenRouter API key to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

8. **Final System Verification**
   ```bash
   # Verify Python environment
   python -c "import streamlit; print('✅ Streamlit installed successfully')"
   
   # Verify model files
   python -c "import os; print('✅ Model size:', os.path.getsize('all-MiniLM-L6-v2/model.safetensors'), 'bytes')"
   ```

### 🛠️ Troubleshooting Installation Issues

#### Git LFS Issues

**Problem**: `HeaderTooLarge` error when starting the application

**Diagnosis**: Git LFS pointer files were downloaded instead of actual model files

**Solution**:
```bash
# Step 1: Verify git-lfs is installed and working
git lfs version

# Step 2: If git-lfs is not installed, install it
sudo apt-get update && sudo apt-get install git-lfs  # Linux
brew install git-lfs                                 # macOS

# Step 3: Initialize git-lfs for your user
git lfs install

# Step 4: Clean up corrupted download
rm -rf all-MiniLM-L6-v2

# Step 5: Re-download correctly
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Step 6: Verify success
ls -lh all-MiniLM-L6-v2/model.safetensors
# Should show ~91M, not a few hundred bytes
```

#### File Size Verification

**Check if you have the correct files**:
```bash
# These commands should show large file sizes
ls -lh all-MiniLM-L6-v2/model.safetensors      # Should be ~91M
ls -lh all-MiniLM-L6-v2/pytorch_model.bin      # Should be ~91M
du -sh all-MiniLM-L6-v2/                       # Should be ~200M total
```

**If files are tiny (< 1KB each)**:
```bash
# You have pointer files, not real files
cat all-MiniLM-L6-v2/model.safetensors
# If you see "version https://git-lfs.github.com/spec/v1", you have pointers

# Fix: Delete and re-download with proper git-lfs setup
rm -rf all-MiniLM-L6-v2
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

#### Network and Connectivity Issues

**Problem**: Download hangs or fails

**Solutions**:
```bash
# Option 1: Use git-lfs explicit pull
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
cd all-MiniLM-L6-v2
git lfs pull

# Option 2: Use alternative clone method
git lfs clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Option 3: Check LFS tracking
git lfs track
git lfs ls-files
```

#### Directory Structure Verification

After **successful** installation, your directory should look like:
```
project-kural/
├── all-MiniLM-L6-v2/              # ← Downloaded embeddings model
│   ├── config.json                # ~1KB
│   ├── model.safetensors          # ~91MB ← MUST BE LARGE
│   ├── pytorch_model.bin          # ~91MB ← MUST BE LARGE
│   ├── tokenizer.json             # ~2MB
│   ├── vocab.txt                  # ~232KB
│   └── ...
├── core/
├── personas/
├── training_data/
├── app.py
├── requirements.txt
└── README.md
```

**🚨 WARNING SIGNS** - If you see:
- `model.safetensors` showing 133 bytes instead of ~91MB
- Files containing "version https://git-lfs.github.com/spec/v1"
- Total directory size under 10MB

**You have pointer files, not actual model files. The application will fail to start.**