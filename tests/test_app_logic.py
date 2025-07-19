"""
FINAL VERIFICATION PROTOCOL - Project Kural App Logic Test Suite
Comprehensive pytest suite for headless testing of app.py functions
Tests the application logic in isolation with complete backend mocking

CRITICAL FIX: Absolute imports from project root as required
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple
import gradio as gr

# CRITICAL: Correct absolute imports from project root
from project_kural.app import (
    start_session, 
    end_session, 
    process_interaction, 
    clear_conversation, 
    get_system_status, 
    generate_voice_response, 
    initialize_components, 
    create_custom_theme
)


class TestProjectKuralApp:
    """Comprehensive test suite for Project Kural app.py functions."""

    @pytest.fixture
    def mock_backend_services(self, monkeypatch):
        """Master fixture that mocks all global core modules used by app.py."""
        
        # Mock perception module
        mock_perception = MagicMock()
        mock_perception.transcribe_audio.return_value = {
            'text': 'test transcription', 
            'language': 'en'
        }
        mock_perception.analyze_sentiment.return_value = "Positive"
        monkeypatch.setattr('project_kural.app.perception_module', mock_perception)
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = "This is a mock AI response."
        monkeypatch.setattr('project_kural.app.agent', mock_agent)
        
        # Mock memory module
        mock_memory = MagicMock()
        mock_memory.get_long_term_summary.return_value = "User has a previous history."
        mock_memory.get_short_term_memory.return_value = None
        mock_memory.save_conversation_summary.return_value = None
        monkeypatch.setattr('project_kural.app.memory_module', mock_memory)
        
        # Mock vector store
        mock_vector_store = MagicMock()
        monkeypatch.setattr('project_kural.app.vector_store', mock_vector_store)
        
        # Mock generate_voice_response function
        def mock_voice_response(text: str, language: str) -> str:
            return "/tmp/mock_audio.mp3"
        
        monkeypatch.setattr('project_kural.app.generate_voice_response', mock_voice_response)
        
        # Mock initialization status as successful
        mock_init_status = {
            "success": True,
            "error": None,
            "timestamp": "12:00:00",
            "init_time": "2.5s"
        }
        monkeypatch.setattr('project_kural.app.initialization_status', mock_init_status)
        
        return {
            'perception': mock_perception,
            'agent': mock_agent,
            'memory': mock_memory,
            'vector_store': mock_vector_store
        }

    @pytest.fixture
    def initial_session_state(self):
        """Fixture that returns a clean, default session state dictionary."""
        return {
            "user_id": None,
            "chat_history": [],
            "active": False
        }

    @pytest.fixture
    def active_session_state(self):
        """Fixture that returns an active session state."""
        return {
            "user_id": "TEST_USER",
            "chat_history": [],
            "session_start": "2025-07-19 12:00:00",
            "active": True,
            "has_history": False
        }

    @pytest.fixture
    def sample_audio_input(self):
        """Fixture that returns sample audio input."""
        sample_rate = 16000
        audio_data = np.zeros(16000, dtype=np.int16)  # 1 second of silence
        return (sample_rate, audio_data)

    def test_theme_creation(self):
        """Test that create_custom_theme returns a valid Gradio theme."""
        theme = create_custom_theme()
        
        # Assert that the return value is an instance of gr.themes.Theme
        assert isinstance(theme, gr.themes.Theme), "Theme should be a Gradio Theme instance"

    def test_start_session_happy_path(self, initial_session_state):
        """Test successful session start with valid User ID."""
        # Test the happy path
        new_session, login_visible, chat_visible, status_msg = start_session(
            "TEST_USER", initial_session_state
        )
        
        # Assert session state is updated correctly
        assert new_session["active"] is True, "Session should be active"
        assert new_session["user_id"] == "TEST_USER", "User ID should be set correctly"
        assert isinstance(new_session["chat_history"], list), "Chat history should be a list"
        assert "session_start" in new_session, "Session should have start timestamp"
        
        # Assert UI updates are correct
        assert login_visible is False, "Login UI should be hidden"
        assert chat_visible is True, "Chat UI should be visible"
        assert "Welcome" in status_msg, "Status message should indicate success"

    def test_start_session_sad_path(self, initial_session_state):
        """Test session start failure with empty User ID."""
        # Test with empty string
        new_session, login_visible, chat_visible, status_msg = start_session(
            "", initial_session_state
        )
        
        # Assert session state does not change
        assert new_session == initial_session_state, "Session state should not change"
        assert login_visible is True, "Login UI should remain visible"
        assert chat_visible is False, "Chat UI should remain hidden"
        assert "❌" in status_msg, "Status message should indicate error"

        # Test with None
        new_session, login_visible, chat_visible, status_msg = start_session(
            None, initial_session_state
        )
        
        assert new_session == initial_session_state, "Session state should not change with None"

        # Test with whitespace only
        new_session, login_visible, chat_visible, status_msg = start_session(
            "   ", initial_session_state
        )
        
        assert new_session == initial_session_state, "Session state should not change with whitespace"

    def test_process_interaction_success(self, mock_backend_services, active_session_state, sample_audio_input):
        """Test successful voice input processing - the most critical test."""
        # Call process_interaction with valid inputs
        chat_history, audio_path, status_msg = process_interaction(
            sample_audio_input, active_session_state
        )
        
        # Assert that the returned chat_history contains two entries
        assert len(chat_history) == 2, "Chat history should contain user and AI messages"
        
        # Check user message
        user_msg = chat_history[0]
        assert user_msg["role"] == "user", "First message should be from user"
        assert user_msg["content"] == "test transcription", "User message should contain transcribed text"
        
        # Check AI response
        ai_msg = chat_history[1]
        assert ai_msg["role"] == "assistant", "Second message should be from assistant"
        assert ai_msg["content"] == "This is a mock AI response.", "AI message should contain mock response"
        
        # Assert audio response path is correct
        assert audio_path == "/tmp/mock_audio.mp3", "Audio path should be the mocked path"
        
        # Assert status indicates success
        assert "✅" in status_msg, "Status should indicate success"
        
        # Verify that backend services were called correctly
        mock_backend_services['perception'].transcribe_audio.assert_called_once_with(sample_audio_input)
        mock_backend_services['perception'].analyze_sentiment.assert_called_once_with("test transcription")
        mock_backend_services['agent'].run.assert_called_once()
        mock_backend_services['memory'].get_long_term_summary.assert_called_once_with("TEST_USER")

    def test_process_interaction_no_session(self, mock_backend_services, initial_session_state, sample_audio_input):
        """Test voice input processing without active session."""
        chat_history, audio_path, status_msg = process_interaction(
            sample_audio_input, initial_session_state
        )
        
        # Should return empty results with error message
        assert chat_history == [], "Chat history should be empty without session"
        assert audio_path == "", "Audio path should be empty"
        assert "❌" in status_msg, "Status should indicate error"
        assert "No active session" in status_msg, "Error should mention session requirement"

    def test_process_interaction_no_audio(self, mock_backend_services, active_session_state):
        """Test voice input processing with no audio input."""
        chat_history, audio_path, status_msg = process_interaction(
            None, active_session_state
        )
        
        # Should handle gracefully
        assert chat_history == [], "Chat history should be empty with no audio"
        assert audio_path == "", "Audio path should be empty"
        assert "No audio detected" in status_msg, "Status should indicate no audio"

    def test_process_interaction_system_not_ready(self, monkeypatch, active_session_state, sample_audio_input):
        """Test voice input processing when system is not initialized."""
        # Mock failed initialization status
        mock_init_status = {
            "success": False,
            "error": "System initialization failed",
            "timestamp": "12:00:00"
        }
        monkeypatch.setattr('project_kural.app.initialization_status', mock_init_status)
        
        chat_history, audio_path, status_msg = process_interaction(
            sample_audio_input, active_session_state
        )
        
        # Should return error status
        assert chat_history == [], "Chat history should be empty when system not ready"
        assert audio_path == "", "Audio path should be empty"
        assert "🔴 System not ready" in status_msg, "Status should indicate system not ready"

    def test_end_session_flow(self, active_session_state):
        """Test successful session termination."""
        # Call end_session with active session
        result = end_session(active_session_state)
        
        # Unpack the tuple result
        new_session, login_visible, chat_visible, status_msg, chat_history, audio_output = result
        
        # Assert that the returned new_session is reset
        assert new_session["active"] is False, "Session should be inactive"
        assert new_session["user_id"] is None, "User ID should be cleared"
        assert new_session["chat_history"] == [], "Chat history should be cleared"
        
        # Assert UI updates correctly
        assert login_visible is True, "Login UI should be visible"
        assert chat_visible is False, "Chat UI should be hidden"
        assert status_msg == "", "Status message should be empty"
        assert chat_history == [], "Chat history should be empty"
        assert audio_output == "", "Audio output should be empty"

    def test_end_session_inactive_session(self, initial_session_state):
        """Test ending session when no session is active."""
        result = end_session(initial_session_state)
        
        # Should still work and return reset state
        new_session, login_visible, chat_visible, status_msg, chat_history, audio_output = result
        
        assert new_session["active"] is False, "Session should remain inactive"
        assert login_visible is True, "Login UI should be visible"
        assert chat_visible is False, "Chat UI should be hidden"

    def test_clear_conversation_active_session(self, active_session_state):
        """Test clearing conversation with active session."""
        # Add some chat history first
        active_session_state["chat_history"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        chat_history, status_msg = clear_conversation(active_session_state)
        
        # Assert conversation is cleared
        assert chat_history == [], "Chat history should be empty"
        assert "✅" in status_msg, "Status should indicate success"
        assert active_session_state["chat_history"] == [], "Session chat history should be cleared"

    def test_clear_conversation_no_session(self, initial_session_state):
        """Test clearing conversation without active session."""
        chat_history, status_msg = clear_conversation(initial_session_state)
        
        # Should return error
        assert chat_history == [], "Chat history should be empty"
        assert "❌" in status_msg, "Status should indicate error"
        assert "No active session" in status_msg, "Error should mention session requirement"

    def test_get_system_status_operational(self, monkeypatch):
        """Test system status display when operational."""
        # Mock successful initialization
        mock_init_status = {
            "success": True,
            "error": None,
            "timestamp": "12:00:00",
            "init_time": "2.5s"
        }
        monkeypatch.setattr('project_kural.app.initialization_status', mock_init_status)
        
        status_html = get_system_status()
        
        # Assert status contains operational indicators
        assert "✅" in status_html, "Status should show success icon"
        assert "OPERATIONAL" in status_html, "Status should show operational text"
        assert "2.5s" in status_html, "Status should show initialization time"
        assert "Voice Recognition: Active" in status_html, "Should show component status"
        assert "File Upload: Ready" in status_html, "Should show dual input status"

    def test_get_system_status_error(self, monkeypatch):
        """Test system status display when in error state."""
        # Mock failed initialization
        mock_init_status = {
            "success": False,
            "error": "Component initialization failed",
            "timestamp": "12:00:00",
            "init_time": "Failed"
        }
        monkeypatch.setattr('project_kural.app.initialization_status', mock_init_status)
        
        status_html = get_system_status()
        
        # Assert status contains error indicators
        assert "❌" in status_html, "Status should show error icon"
        assert "ERROR" in status_html, "Status should show error text"
        assert "Component initialization failed" in status_html, "Should show error message"
        assert "System Error:" in status_html, "Should show error section"

    def test_generate_voice_response_success(self, monkeypatch):
        """Test successful voice response generation."""
        # Mock gTTS to avoid actual TTS calls
        mock_tts = MagicMock()
        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance
        
        with patch('project_kural.app.gTTS', mock_tts):
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test_audio.mp3"
                
                result = generate_voice_response("Hello world", "en")
                
                assert result == "/tmp/test_audio.mp3", "Should return temp file path"
                mock_tts.assert_called_once_with(text="Hello world", lang="en", slow=False)

    def test_generate_voice_response_failure(self, monkeypatch):
        """Test voice response generation failure."""
        # Mock gTTS to raise an exception
        with patch('project_kural.app.gTTS', side_effect=Exception("TTS failed")):
            result = generate_voice_response("Hello world", "en")
            
            assert result is None, "Should return None on failure"

    def test_initialize_components_success(self, monkeypatch):
        """Test successful component initialization."""
        # Mock all the component classes
        mock_perception_class = MagicMock()
        mock_memory_class = MagicMock()
        mock_agent_class = MagicMock()
        mock_vector_store_func = MagicMock()
        
        monkeypatch.setattr('project_kural.app.PerceptionModule', mock_perception_class)
        monkeypatch.setattr('project_kural.app.MemoryModule', mock_memory_class)
        monkeypatch.setattr('project_kural.app.KuralAgent', mock_agent_class)
        monkeypatch.setattr('project_kural.app.initialize_knowledge_base', mock_vector_store_func)
        
        result = initialize_components()
        
        assert result is True, "Initialization should return True on success"
        
        # Import the module to check the global status
        import project_kural.app as app_module
        assert app_module.initialization_status["success"] is True, "Status should indicate success"
        
        # Verify all components were initialized
        mock_perception_class.assert_called_once()
        mock_memory_class.assert_called_once()
        mock_agent_class.assert_called_once()
        mock_vector_store_func.assert_called_once()

    def test_initialize_components_failure(self, monkeypatch):
        """Test component initialization failure."""
        # Mock PerceptionModule to raise an exception
        monkeypatch.setattr('project_kural.app.PerceptionModule', 
                          MagicMock(side_effect=Exception("Init failed")))
        
        result = initialize_components()
        
        assert result is False, "Initialization should return False on failure"
        
        # Import the module to check the global status
        import project_kural.app as app_module
        assert app_module.initialization_status["success"] is False, "Status should indicate failure"
        assert "Init failed" in app_module.initialization_status["error"], "Should contain error message"

    def test_session_state_persistence(self, active_session_state, sample_audio_input, mock_backend_services):
        """Test that session state is properly maintained across operations."""
        # Process voice input and verify session state is updated
        chat_history, _, _ = process_interaction(sample_audio_input, active_session_state)
        
        # Session state should be modified to include chat history
        assert active_session_state["chat_history"] == chat_history, "Session should be updated with chat history"
        assert active_session_state["user_id"] == "TEST_USER", "User ID should be preserved"
        assert active_session_state["active"] is True, "Session should remain active"

    def test_language_handling(self, mock_backend_services, active_session_state, sample_audio_input):
        """Test proper handling of different languages."""
        # Mock transcription to return different language
        mock_backend_services['perception'].transcribe_audio.return_value = {
            'text': 'bonjour le monde', 
            'language': 'fr'
        }
        
        chat_history, audio_path, status_msg = process_interaction(
            sample_audio_input, active_session_state
        )
        
        # Verify the agent was called with French language
        call_args = mock_backend_services['agent'].run.call_args
        assert call_args[1]['language'] == 'fr', "Agent should receive correct language"
        assert call_args[1]['user_input'] == 'bonjour le monde', "Agent should receive correct text"

    def test_error_handling_edge_cases(self, mock_backend_services, active_session_state):
        """Test error handling for various edge cases."""
        # Test with invalid audio format
        invalid_audio = "not_audio_data"
        chat_history, audio_path, status_msg = process_interaction(
            invalid_audio, active_session_state
        )
        
        assert "❌" in status_msg, "Should handle invalid audio gracefully"
        assert "Invalid audio format" in status_msg, "Should indicate audio format error"

    def test_memory_integration(self, mock_backend_services, active_session_state, sample_audio_input):
        """Test integration with memory module."""
        # Set up a session with chat history to trigger memory saving
        active_session_state["chat_history"] = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        # Process new input (this should trigger memory saving since we'll have 4+ messages)
        process_interaction(sample_audio_input, active_session_state)
        
        # Verify memory methods were called
        mock_backend_services['memory'].get_long_term_summary.assert_called_with("TEST_USER")
        mock_backend_services['memory'].get_short_term_memory.assert_called_once()
        
        # After processing, we should have 4 messages, which should trigger save
        mock_backend_services['memory'].save_conversation_summary.assert_called_once()

    def test_dual_voice_input_file_upload(self, mock_backend_services, active_session_state):
        """Test file upload functionality - MANDATE 3 verification."""
        # Mock librosa for file processing
        with patch('project_kural.app.librosa') as mock_librosa:
            mock_librosa.load.return_value = (np.zeros(16000), 16000)
            
            # Test with file path (simulating file upload)
            file_path = "/tmp/test_audio.wav"
            chat_history, audio_path, status_msg = process_interaction(
                file_path, active_session_state
            )
            
            # Should process successfully
            assert len(chat_history) == 2, "Should process uploaded file successfully"
            assert "✅" in status_msg, "Should indicate successful processing"
            mock_librosa.load.assert_called_once_with(file_path, sr=16000)

    def test_multilingual_voice_output(self, mock_backend_services, active_session_state, sample_audio_input):
        """Test multilingual voice output generation - MANDATE 2 verification."""
        # Mock different language detection
        mock_backend_services['perception'].transcribe_audio.return_value = {
            'text': 'வணக்கம்', 
            'language': 'ta'  # Tamil
        }
        
        with patch('project_kural.app.generate_voice_response') as mock_voice_gen:
            mock_voice_gen.return_value = "/tmp/tamil_response.mp3"
            
            chat_history, audio_path, status_msg = process_interaction(
                sample_audio_input, active_session_state
            )
            
            # Verify voice generation was called with Tamil language
            mock_voice_gen.assert_called_once()
            call_args = mock_voice_gen.call_args
            assert call_args[0][1] == 'ta', "Voice generation should use detected language"
            assert audio_path == "/tmp/tamil_response.mp3", "Should return generated audio path"