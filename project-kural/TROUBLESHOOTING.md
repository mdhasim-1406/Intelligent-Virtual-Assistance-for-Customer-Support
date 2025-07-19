# Project Kural - Troubleshooting Guide

## 🚨 Critical Setup Issues

### **Issue 1: "No Microphone Permission" Error**

**Symptoms**: Voice input doesn't work, no transcription happening, audio component appears inactive.

**Root Cause**: Browser has blocked microphone access or permission was accidentally denied.

**Solution**:
1. **Initial Setup**: When you first run the app, your browser (Chrome, Firefox, Safari, Edge) will show a pop-up asking for permission to use your microphone. You **MUST** click **"Allow"**.

2. **If You Accidentally Clicked "Block"**:
   - Look for the padlock icon (🔒) or "Not secure" warning in your browser's address bar
   - Click on the padlock/warning icon
   - Find the "Microphone" permission in the dropdown menu
   - Change it from "Blocked" to "Allow"
   - **IMPORTANT**: Reload the page (F5 or Ctrl+R) for the change to take effect

3. **Alternative Method**:
   - Go to your browser settings
   - Search for "Site permissions" or "Privacy and security"
   - Find "Camera and microphone" settings
   - Add `localhost:7860` to the allowed sites

### **Issue 2: "No Speech Detected" Error**

**Symptoms**: Microphone works but transcription is empty or fails.

**Solution**:
1. **Check Microphone Quality**: Ensure you're speaking clearly and close to the microphone
2. **Background Noise**: Try recording in a quieter environment
3. **Recording Duration**: Speak for at least 2-3 seconds before stopping
4. **Microphone Selection**: Check if the correct microphone is selected in your browser settings

### **Issue 3: Dependencies/Import Errors**

**Symptoms**: Application fails to start with ImportError or ModuleNotFoundError.

**Solution**:
```bash
# Ensure you're in the project directory
cd project-kural

# Install all dependencies
pip install -r requirements.txt --upgrade

# Install system dependencies (Linux/Mac)
sudo apt-get install ffmpeg  # Linux
brew install ffmpeg          # Mac

# For Windows, download ffmpeg from https://ffmpeg.org/download.html
```

### **Issue 4: API Key Missing**

**Symptoms**: "OPENROUTER_API_KEY not found" error on startup.

**Solution**:
1. Create a `.env` file in the project-kural directory
2. Add your API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
3. Get your API key from: https://openrouter.ai/
4. Restart the application

## 🔧 Debug Mode Features

The application now runs in **DEBUG MODE** by default, which provides:

### **Comprehensive Logging**
- All events are logged to the terminal
- Detailed log file: `project_kural_debug.log`
- Each step of the voice processing pipeline is tracked

### **Error Display**
- Backend errors are displayed directly in the chat interface
- System status is shown in real-time on the UI
- Component initialization status is visible

### **Testing Tools**
- Browser console logging (Press F12 → Console tab)
- Step-by-step process verification
- Audio format validation

## 🎯 Step-by-Step Testing Process

### **1. Startup Verification**
```bash
python app.py
```

Look for these success messages:
- ✅ API key found
- ✅ Perception module initialized successfully
- ✅ Knowledge base initialized successfully
- ✅ Agent initialized successfully
- 🎉 ALL COMPONENTS INITIALIZED SUCCESSFULLY

### **2. Browser Setup**
1. Open http://localhost:7860 in your browser
2. Look for the green "✅ System Status: READY" indicator
3. Grant microphone permission when prompted

### **3. Voice Test**
1. Click the 🎤 microphone button
2. Record a clear message: "Hello, can you help me?"
3. Check the terminal for these logs:
   - 🎯 AUDIO INPUT EVENT TRIGGERED
   - 📊 Audio data - Sample rate: XXXX, Shape: (XXXX,)
   - 📝 Transcription result: 'Hello, can you help me?' (Language: en)
   - 💭 Agent response generated

### **4. Expected Results**
- User message appears in chat history
- AI response appears in chat history
- Voice response plays automatically
- No error messages in red

## 🚨 Common Error Messages and Solutions

### **"🔴 System Error: Components not initialized"**
- **Cause**: Backend modules failed to load
- **Solution**: Check terminal for specific error details, ensure all dependencies are installed

### **"🔴 Audio Format Error: Invalid audio format"**
- **Cause**: Browser sent invalid audio data
- **Solution**: Try a different browser (Chrome recommended), check microphone permissions

### **"🔴 Transcription Error"**
- **Cause**: Whisper model failed to process audio
- **Solution**: Ensure ffmpeg is installed, check audio quality, try speaking more clearly

### **"🔴 Agent Error"**
- **Cause**: AI model API call failed
- **Solution**: Check API key, internet connection, and OpenRouter service status

## 🔍 Advanced Debugging

### **Check Log File**
```bash
tail -f project_kural_debug.log
```
This shows real-time logs as you use the application.

### **Browser Developer Tools**
1. Press F12 to open Developer Tools
2. Go to Console tab
3. Look for JavaScript errors or network issues
4. Go to Network tab to check API calls

### **Test Individual Components**
```python
# Test in Python shell
from core.perception import PerceptionModule
perception = PerceptionModule()
print(perception.health_check())
```

## 📞 Support Checklist

Before seeking help, please verify:
- [ ] Python 3.8+ is installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] ffmpeg is installed and accessible
- [ ] OPENROUTER_API_KEY is in .env file
- [ ] Browser allows microphone access
- [ ] No firewall blocking localhost:7860
- [ ] Check project_kural_debug.log for errors

## 🚀 Performance Tips

1. **Use Chrome or Firefox** for best microphone support
2. **Close other apps** using the microphone
3. **Speak clearly** and avoid background noise
4. **Wait for responses** - AI processing takes 2-5 seconds
5. **Check internet connection** for API calls

---

**Remember**: Debug mode provides extensive logging. Always check the terminal output and log file for detailed error information!