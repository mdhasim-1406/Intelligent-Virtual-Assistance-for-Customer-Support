"""
Project Kural: Multilingual AI Customer Service Agent with Voice Interface
FINAL PRODUCTION VERSION - Complete System Overhaul & Polish

FINAL DIRECTIVE IMPLEMENTATION:
- AI Output Refinement: Clean responses without internal reasoning exposure
- UI Stability: Perfect event handler return matching
- Minimalist Design: Professional, clean, accordion-based layout
"""

import gradio as gr
import os
import logging
import tempfile
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
import numpy as np
import time
from datetime import datetime
from langdetect import detect, LangDetectException

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('project_kural_production.log')
    ]
)
logger = logging.getLogger(__name__)

# Import core modules with error handling
try:
    from core.perception import PerceptionModule
    from core.agent import KuralAgent
    from core.memory import MemoryModule
    from core.tools import get_billing_info, check_network_status
    from core.vector_store import initialize_knowledge_base
    logger.info("✅ All core modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import core modules: {e}")
    raise

# Global components - initialized once for efficiency
perception_module = None
agent = None
memory_module = None
vector_store = None
initialization_status = {"success": False, "error": None, "timestamp": None}

def initialize_components():
    """Initialize all core components globally with comprehensive error handling."""
    global perception_module, agent, memory_module, vector_store, initialization_status
    
    try:
        logger.info("🚀 Starting Project Kural component initialization...")
        start_time = time.time()
        
        # Initialize perception module
        perception_module = PerceptionModule()
        logger.info("✅ Perception module initialized")
        
        # Initialize memory module
        memory_module = MemoryModule()
        logger.info("✅ Memory module initialized")
        
        # Initialize knowledge base
        vector_store = initialize_knowledge_base()
        logger.info("✅ Knowledge base initialized")
        
        # Initialize agent with tools
        tools = [get_billing_info, check_network_status]
        agent = KuralAgent(
            openrouter_api_key="",  # Will use environment variable
            tools=tools,
            vector_store=vector_store
        )
        logger.info("✅ Agent initialized")
        
        initialization_time = time.time() - start_time
        initialization_status = {
            "success": True, 
            "error": None, 
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "init_time": f"{initialization_time:.2f}s"
        }
        logger.info(f"🎉 ALL COMPONENTS INITIALIZED in {initialization_time:.2f}s")
        return True
        
    except Exception as e:
        error_msg = f"Component initialization failed: {e}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        initialization_status = {
            "success": False, 
            "error": error_msg, 
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "init_time": "Failed"
        }
        return False

def generate_voice_response(text: str, language: str) -> Optional[str]:
    """Generate voice response using gTTS and return file path."""
    try:
        from gtts import gTTS
        
        lang_map = {"en": "en", "ta": "ta", "hi": "hi", "es": "es", "fr": "fr"}
        tts_lang = lang_map.get(language, "en")
        
        # Generate TTS
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name
            tts.save(temp_path)
        
        logger.info(f"🎵 Voice response generated: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"❌ Voice generation failed: {e}")
        return None

def create_minimalist_theme():
    """
    Creates a professional minimalist theme using Gradio's built-in themes.
    Uses Monochrome base with clean, aesthetic customizations.
    """
    return gr.themes.Monochrome(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Lexend Deca"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

def start_session(user_id: str, session_state: Dict) -> Tuple[Dict, bool, bool, str]:
    """
    MANDATE 2: Multi-Stage User Flow - Session Start
    Transitions from Login UI to Chat UI with proper session management.
    """
    try:
        if not user_id or not user_id.strip():
            return session_state, True, False, "❌ Please enter a valid User ID or Order Number"
        
        clean_user_id = user_id.strip()
        
        # Create new session state
        new_session = {
            "user_id": clean_user_id,
            "chat_history": [],
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": True
        }
        
        # Load user's historical context if available
        try:
            if memory_module:
                long_term_summary = memory_module.get_long_term_summary(clean_user_id)
                new_session["has_history"] = bool(long_term_summary)
                logger.info(f"📚 Session started for: {clean_user_id}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load user history: {e}")
            new_session["has_history"] = False
        
        logger.info(f"🚀 Session started for user: {clean_user_id}")
        
        return (
            new_session,  # Updated session state
            False,        # Hide login UI
            True,         # Show chat UI
            f"✅ Welcome! Session started for: {clean_user_id}"
        )
        
    except Exception as e:
        logger.error(f"❌ Session start failed: {e}")
        return (
            session_state,
            True,
            False,
            f"❌ Session start failed: {str(e)}"
        )

def end_session(session_state: Dict) -> Tuple[Dict, bool, bool, str, List, str]:
    """End the current session and return to login UI."""
    try:
        if session_state.get("active"):
            user_id = session_state.get("user_id", "Unknown")
            logger.info(f"👋 Session ended for user: {user_id}")
        
        # Reset session state
        empty_session = {"user_id": None, "chat_history": [], "active": False}
        
        return (
            empty_session,  # Reset session state
            True,           # Show login UI
            False,          # Hide chat UI
            "",             # Clear status message
            [],             # Clear chat history
            ""              # Clear audio output
        )
        
    except Exception as e:
        logger.error(f"❌ Session end failed: {e}")
        return session_state, True, False, "❌ Error ending session", [], ""

def handle_interaction(text_input: str, audio_input: Tuple, session_state: Dict) -> Tuple[Dict, List, str, str]:
    """
    COMMAND 4: Unified Core Logic Handler
    MANDATE 3: Dual Voice Input Processing + Text Input
    MANDATE 4: Multilingual Voice I/O
    Core interaction processor handling text input, microphone recording, and file upload input.
    """
    try:
        # --- Start of handle_interaction function ---
        chat_history = session_state.get("chat_history", [])
        user_id = session_state.get("user_id")

        if not user_id:
            # Safety check
            return session_state, chat_history, "", "❌ Session not active. Please restart."

        # Determine input source
        if audio_input is not None:
            logger.info("🎤 Processing voice input...")
            
            # Check system status
            if not initialization_status["success"]:
                error_msg = f"🔴 System not ready: {initialization_status['error']}"
                return session_state, chat_history, "", error_msg
            
            # MANDATE 3: Handle both microphone recording and file upload
            try:
                if isinstance(audio_input, tuple):
                    # Microphone input: (sample_rate, audio_data)
                    sample_rate, audio_data = audio_input
                    logger.info(f"📊 Microphone input: {sample_rate}Hz, shape: {audio_data.shape}")
                    processed_audio = audio_input
                elif isinstance(audio_input, str):
                    # File upload: file path
                    logger.info(f"📁 File upload: {audio_input}")
                    # Use librosa to load the uploaded file
                    import librosa
                    audio_data, sample_rate = librosa.load(audio_input, sr=16000)
                    processed_audio = (sample_rate, audio_data)
                    logger.info(f"📁 File processed: {sample_rate}Hz, length: {len(audio_data)}")
                else:
                    return session_state, chat_history, "", "❌ Invalid audio format"
            except Exception as e:
                return session_state, chat_history, "", f"❌ Audio processing error: {e}"
            
            # MANDATE 4: Transcribe audio with language detection
            try:
                transcription_result = perception_module.transcribe_audio(processed_audio)
                user_text = transcription_result.get("text", "").strip()
                detected_language = transcription_result.get("language", "en")
                
                if not user_text:
                    return session_state, chat_history, "", "⚠️ No speech detected"
                
                logger.info(f"📝 Transcribed: '{user_text}' ({detected_language})")
                
            except Exception as e:
                logger.error(f"❌ Transcription failed: {e}")
                return session_state, chat_history, "", f"❌ Transcription error: {e}"
                
        elif text_input and text_input.strip():
            logger.info("⌨️ Processing text input...")
            user_text = text_input.strip()
            # CRITICAL INPUT LOGIC UPGRADE: Robust text-based language detection
            try:
                # Detect the language of the text input
                detected_language = detect(user_text)
                logger.info(f"Language detected for text input: {detected_language}")
            except LangDetectException:
                # If detection fails (e.g., for very short text), default to English
                logger.warning("Language detection failed for text, defaulting to 'en'.")
                detected_language = "en"
        else:
            # No input provided
            return session_state, chat_history, "", "" # Return silently

        # Analyze sentiment
        try:
            sentiment = perception_module.analyze_sentiment(user_text)
            logger.info(f"😊 Sentiment: {sentiment}")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analysis failed: {e}")
            sentiment = "Neutral"
        
        # Add user message
        chat_history.append({"role": "user", "content": user_text})
        
        # Get user context
        try:
            long_term_summary = memory_module.get_long_term_summary(user_id)
            short_term_memory = memory_module.get_short_term_memory()
        except Exception as e:
            logger.warning(f"⚠️ Memory retrieval failed: {e}")
            long_term_summary = ""
            short_term_memory = None
        
        # MANDATE 4: Generate AI response with detected language
        try:
            ai_response = agent.run(
                user_id=user_id,
                user_input=user_text,
                language=detected_language,
                sentiment=sentiment,
                short_term_memory=short_term_memory,
                long_term_summary=long_term_summary
            )
            
            logger.info(f"🤖 AI Response generated")
            
        except Exception as e:
            logger.error(f"❌ Agent error: {e}")
            ai_response = "I apologize, but I'm experiencing technical difficulties. Please try again."
        
        # Add AI response to chat history
        chat_history.append({"role": "assistant", "content": ai_response})
        
        # Update session state
        session_state["chat_history"] = chat_history
        
        # MANDATE 4: Generate voice response in detected language
        audio_response_path = ""
        try:
            audio_path = generate_voice_response(ai_response, detected_language)
            if audio_path:
                audio_response_path = audio_path
        except Exception as e:
            logger.error(f"❌ Voice generation error: {e}")
        
        # Save conversation if needed
        try:
            if len(chat_history) >= 4:
                from langchain_core.messages import HumanMessage, AIMessage
                messages = []
                for msg in chat_history[-4:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                if messages:
                    memory_module.save_conversation_summary(user_id, messages)
                    logger.info("💾 Conversation saved")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save conversation: {e}")
        
        # The function must end by returning a tuple of 4 items to match the outputs
        # return updated_session_state, updated_chat_history, audio_response_path, ""
        return session_state, chat_history, audio_response_path, ""
        
    except Exception as e:
        logger.error(f"💥 Critical error in interaction processing: {e}", exc_info=True)
        chat_history.append({"role": "assistant", "content": f"❌ System error: {e}"})
        return session_state, chat_history, "", ""

def clear_conversation(session_state: Dict) -> Tuple[List, str]:
    """Clear the current conversation while maintaining session."""
    try:
        if session_state.get("active"):
            session_state["chat_history"] = []
            user_id = session_state.get("user_id", "User")
            logger.info(f"🗑️ Conversation cleared for: {user_id}")
            return [], f"✅ Conversation cleared for {user_id}"
        else:
            return [], "❌ No active session"
    except Exception as e:
        logger.error(f"❌ Clear conversation failed: {e}")
        return [], f"❌ Error: {e}"

def get_system_status() -> str:
    """Get current system status for display."""
    try:
        if initialization_status["success"]:
            status_color = "#22c55e"
            status_icon = "✅"
            status_text = "OPERATIONAL"
            
            components_status = f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 12px;">
                <div style="font-size: 11px; color: #059669;">🎤 Voice Recognition: Active</div>
                <div style="font-size: 11px; color: #059669;">📁 File Upload: Ready</div>
                <div style="font-size: 11px; color: #059669;">🧠 Knowledge Base: Loaded</div>
                <div style="font-size: 11px; color: #059669;">🤖 AI Agent: Ready</div>
                <div style="font-size: 11px; color: #059669;">💾 Memory: Operational</div>
                <div style="font-size: 11px; color: #6b7280;">⏱️ Init: {initialization_status.get('init_time', 'N/A')}</div>
            </div>
            """
        else:
            status_color = "#ef4444"
            status_icon = "❌"
            status_text = "ERROR"
            
            components_status = f"""
            <div style="margin-top: 12px; padding: 12px; background-color: #fef2f2; border-radius: 8px; border-left: 4px solid #ef4444;">
                <div style="font-size: 12px; color: #dc2626; font-weight: 500;">System Error:</div>
                <div style="font-size: 11px; color: #7f1d1d; margin-top: 4px;">{initialization_status.get('error', 'Unknown error')}</div>
            </div>
            """
        
        return f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; padding: 16px; border: 1px solid #cbd5e0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 16px;">{status_icon}</span>
                <span style="font-weight: 600; color: {status_color}; font-size: 14px;">System Status: {status_text}</span>
                <span style="font-size: 10px; color: #6b7280; margin-left: auto;">Last Check: {initialization_status.get('timestamp', 'N/A')}</span>
            </div>
            {components_status}
        </div>
        """
        
    except Exception as e:
        return f"""
        <div style="background-color: #fef2f2; border-radius: 12px; padding: 16px; border: 1px solid #fca5a5;">
            <div style="color: #dc2626; font-weight: 600;">❌ Status Check Failed</div>
            <div style="font-size: 11px; color: #7f1d1d; margin-top: 4px;">Error: {e}</div>
        </div>
        """

def create_main_interface():
    """
    SIMPLIFIED LAYOUT: Single column with accordion-based Controls & Input section
    Create the main application interface with minimalist design.
    """
    
    # MINIMALIST THEME: Professional Monochrome with Lexend Deca
    custom_theme = create_minimalist_theme()
    
    # Clean, minimalist CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Lexend Deca', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-chatbot {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .controls-accordion {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        margin: 16px 0;
    }
    """
    
    with gr.Blocks(
        title="Project Kural - Professional AI Customer Service",
        theme=custom_theme,
        css=custom_css
    ) as demo:
        
        # Session state management
        session_state = gr.State({
            "user_id": None,
            "chat_history": [],
            "active": False
        })
        
        # Clean header
        gr.HTML("""
        <div style="text-align: center; padding: 24px; background: linear-gradient(135deg, #4299e1 0%, #667eea 100%); color: white; border-radius: 16px; margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 600;">🎭 Project Kural</h1>
            <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">Professional AI Customer Service Platform</p>
        </div>
        """)
        
        # Status message
        status_message = gr.HTML("", visible=False)
        
        # Login UI
        with gr.Group(visible=True) as login_ui:
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="text-align: center; margin-bottom: 24px;">
                    <h3 style="color: #2d3748; margin-bottom: 8px;">Start Your Session</h3>
                    <p style="color: #718096; font-size: 14px;">Enter your User ID or Order Number</p>
                </div>
                """)
                
                user_id_input = gr.Textbox(
                    label="👤 User ID / Order Number",
                    placeholder="e.g., USER123, ORD-456789",
                    lines=1,
                    container=True
                )
                
                with gr.Row():
                    start_session_btn = gr.Button("🚀 Start Session", variant="primary", size="lg")
                    demo_btn = gr.Button("🎯 Demo", variant="secondary", size="lg")
        
        # Main Chat UI - SIMPLIFIED LAYOUT
        with gr.Group(visible=False) as chat_ui:
            session_info = gr.HTML("", visible=False)
            
            # SINGLE MAIN COLUMN
            with gr.Column():
                # Chatbot at the top
                chatbot = gr.Chatbot(
                    value=[],
                    label="💬 Conversation",
                    height=400,
                    type="messages",
                    avatar_images=("👤", "🎭"),
                    elem_classes=["main-chatbot"]
                )
                
                # ACCORDION FOR CONTROLS & INPUT
                with gr.Accordion("🎛️ Controls & Input", open=True, elem_classes=["controls-accordion"]):
                    # Dual voice input
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="🎤📁 Voice Input (Microphone or File Upload)",
                        interactive=True
                    )
                    
                    # Text input
                    with gr.Row():
                        text_input = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    # Control buttons
                    with gr.Row():
                        clear_chat_btn = gr.Button("🗑️ Clear", variant="secondary")
                        end_session_btn = gr.Button("👋 End Session", variant="stop")
                
                # Audio output
                audio_output = gr.Audio(
                    label="🔊 AI Voice Response",
                    autoplay=True,
                    visible=True
                )
                
                # SMALLER ACCORDION FOR SYSTEM STATUS
                with gr.Accordion("📊 System Status", open=False):
                    system_status_display = gr.HTML(get_system_status())
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                
                # SMALLER ACCORDION FOR QUICK GUIDE
                with gr.Accordion("📖 Quick Guide", open=False):
                    gr.HTML("""
                    <div style="font-size: 12px; line-height: 1.4;">
                        <p><strong>🎤 Voice:</strong> Click mic → speak → stop recording</p>
                        <p><strong>📁 Upload:</strong> Click upload tab → select audio file</p>
                        <p><strong>⌨️ Text:</strong> Type message → press Enter or Send</p>
                        <p><strong>🌍 Languages:</strong> English, Tamil, Hindi, Spanish, French</p>
                    </div>
                    """)
        
        # CRITICAL BUG FIX: Event handlers with perfect return matching
        
        def handle_start_session(user_id_value, session_data):
            """FIXED: Returns exactly 6 values to match outputs."""
            result = start_session(user_id_value, session_data)
            new_session, hide_login, show_chat, message = result
            
            if new_session.get("active"):
                session_info_html = f"""
                <div style="background: #48bb78; color: white; padding: 8px 16px; border-radius: 8px; margin-bottom: 16px;">
                    Active Session: {new_session['user_id']} | Started: {new_session['session_start']}
                </div>
                """
                session_info_visible = True
            else:
                session_info_html = ""
                session_info_visible = False
            
            status_html = f"""
            <div style="background: {'#f0fff4' if '✅' in message else '#fff5f5'}; padding: 8px; border-radius: 8px; margin: 8px 0;">
                {message}
            </div>
            """ if message else ""
            
            return (
                new_session,                                          # session_state
                gr.update(visible=not hide_login),                   # login_ui visibility
                gr.update(visible=show_chat),                        # chat_ui visibility
                gr.update(value=status_html, visible=bool(message)), # status_message
                gr.update(value=session_info_html, visible=session_info_visible), # session_info
                ""                                                   # clear user_id_input
            )
        
        def handle_demo_mode(session_data):
            """FIXED: Returns exactly 6 values to match outputs."""
            return handle_start_session("DEMO_USER_001", session_data)
        
        def handle_clear_chat(session_data):
            """FIXED: Returns exactly 3 values to match outputs."""
            cleared_history, status = clear_conversation(session_data)
            status_html = f"""
            <div style="background: #f0fff4; padding: 8px; border-radius: 8px; margin: 8px 0;">
                {status}
            </div>
            """ if status else ""
            return (
                session_data,                                    # session_state
                cleared_history,                                 # chatbot (cleared)
                gr.update(value=status_html, visible=bool(status)) # status_message
            )
        
        def handle_end_session(session_data):
            """FIXED: Returns exactly 7 values to match outputs."""
            result = end_session(session_data)
            new_session, show_login, hide_chat, status, chat_history, audio = result
            
            status_html = """
            <div style="background: #f0fff4; padding: 8px; border-radius: 8px; margin: 8px 0;">
                👋 Session ended successfully. Thank you for using Project Kural!
            </div>
            """
            
            return (
                new_session,                                    # session_state (reset)
                gr.update(visible=show_login),                 # login_ui (show)
                gr.update(visible=not hide_chat),              # chat_ui (hide)
                gr.update(value=status_html, visible=True),    # status_message
                gr.update(value="", visible=False),            # session_info (hide)
                chat_history,                                   # chatbot (clear)
                audio                                          # audio_output (clear)
            )
        
        def handle_refresh():
            """FIXED: Returns exactly 1 value to match outputs."""
            return get_system_status()
        
        # FLAWLESS EVENT WIRING - Perfect input/output matching
        
        start_session_btn.click(
            fn=handle_start_session,
            inputs=[user_id_input, session_state],
            outputs=[session_state, login_ui, chat_ui, status_message, session_info, user_id_input]
        )
        
        user_id_input.submit(
            fn=handle_start_session,
            inputs=[user_id_input, session_state],
            outputs=[session_state, login_ui, chat_ui, status_message, session_info, user_id_input]
        )
        
        demo_btn.click(
            fn=handle_demo_mode,
            inputs=[session_state],
            outputs=[session_state, login_ui, chat_ui, status_message, session_info, user_id_input]
        )
        
        text_input.submit(
            fn=handle_interaction,
            inputs=[text_input, audio_input, session_state],
            outputs=[session_state, chatbot, audio_output, text_input]
        )
        
        send_btn.click(
            fn=handle_interaction,
            inputs=[text_input, audio_input, session_state],
            outputs=[session_state, chatbot, audio_output, text_input]
        )
        
        audio_input.stop_recording(
            fn=handle_interaction,
            inputs=[text_input, audio_input, session_state],
            outputs=[session_state, chatbot, audio_output, text_input]
        )
        
        audio_input.upload(
            fn=handle_interaction,
            inputs=[text_input, audio_input, session_state],
            outputs=[session_state, chatbot, audio_output, text_input]
        )
        
        clear_chat_btn.click(
            fn=handle_clear_chat,
            inputs=[session_state],
            outputs=[session_state, chatbot, status_message]
        )
        
        end_session_btn.click(
            fn=handle_end_session,
            inputs=[session_state],
            outputs=[session_state, login_ui, chat_ui, status_message, session_info, chatbot, audio_output]
        )
        
        refresh_btn.click(
            fn=handle_refresh,
            outputs=[system_status_display]
        )
    
    return demo

def main():
    """Launch the complete Project Kural application with FINAL POLISH."""
    print("\n" + "="*80)
    print("🎭 PROJECT KURAL - FINAL PRODUCTION DEPLOYMENT")
    print("="*80)
    print("FINAL DIRECTIVE IMPLEMENTATION COMPLETE:")
    print("✅ AI OUTPUT REFINEMENT: Clean responses without step-by-step reasoning")
    print("✅ UI STABILITY: Perfect event handler return matching - ALL BUGS FIXED")
    print("✅ MINIMALIST DESIGN: Professional Monochrome theme with Lexend Deca font")
    print("✅ SIMPLIFIED LAYOUT: Accordion-based Controls & Input section")
    print("✅ FLAWLESS EVENT WIRING: Every handler perfectly aligned with outputs")
    print("="*80)
    
    # Validate environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ CRITICAL: OpenRouter API Key Missing!")
        print("Create .env file with: OPENROUTER_API_KEY=your_key")
        return False
    
    print("✅ Environment validated")
    
    # Initialize components
    print("🔧 Initializing system components...")
    if not initialize_components():
        print("❌ Component initialization failed!")
        return False
    
    print("✅ All components initialized successfully")
    
    # Launch interface
    try:
        print("🎨 Creating POLISHED production interface...")
        demo = create_main_interface()
        
        print("\n🚀 PROJECT KURAL FINAL POLISH COMPLETE")
        print("📋 ALL FINAL DIRECTIVE MANDATES FULFILLED:")
        print("   ✅ AI Brain Refined: STEP 4 formatting + native language generation")
        print("   ✅ UI Bugs Fixed: Perfect event handler return matching")
        print("   ✅ Minimalist Theme: Monochrome + Lexend Deca professional design")
        print("   ✅ Simplified Layout: Accordion-based Controls & Input section")
        print("   ✅ Production Ready: Stable, beautiful, intelligent")
        print("\n🌐 Launching FINAL application...")
        
        demo.launch(
            server_name="0.0.0.0",
            share=False,
            show_error=True,
            debug=False,
            max_threads=40
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        exit(1)
