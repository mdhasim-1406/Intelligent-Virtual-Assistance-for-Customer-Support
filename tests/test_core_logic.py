"""
Comprehensive Test Suite for Project Kural Core Logic

This test suite provides complete coverage of the core components:
- KuralAgent persona loading and execution
- MemoryModule conversation management  
- PerceptionModule audio transcription and sentiment analysis
- Tools for billing and network status
"""

import pytest
import os
import json
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project-kural to Python path
sys.path.insert(0, '/home/hasim001/Intelligent_virtual_assistance_for_customer_service_with_fraud_detection/project-kural')

# Import Project Kural modules with explicit imports
from core.agent import KuralAgent, ChatOpenRouter
from core.memory import MemoryModule
from core.perception import PerceptionModule
from core.tools import get_billing_info, check_network_status, _get_billing_info, _check_network_status

# Mock LangChain components since they might not be installed
class MockHumanMessage:
    def __init__(self, content):
        self.content = content

class MockAIMessage:
    def __init__(self, content):
        self.content = content

class MockConversationBufferMemory:
    def __init__(self):
        self.chat_memory = Mock()
        self.chat_memory.add_user_message = Mock()
        self.chat_memory.add_ai_message = Mock()

class MockBaseTool:
    def __init__(self, name):
        self.name = name
        self.invoke = Mock()

# Use mock classes instead of LangChain imports
HumanMessage = MockHumanMessage
AIMessage = MockAIMessage
ConversationBufferMemory = MockConversationBufferMemory

# ===== FIXTURES =====

@pytest.fixture
def api_key():
    """Fixture to provide OpenRouter API key from environment."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key or api_key == "your_key_here":
        pytest.skip("OpenRouter API key not available or is placeholder. Set OPENROUTER_API_KEY environment variable.")
    return api_key

@pytest.fixture
def temp_db(tmp_path):
    """Fixture to create temporary user database for testing."""
    db_file = tmp_path / "test_users.json"
    db_file.write_text("{}")
    return str(db_file)

@pytest.fixture
def mock_whisper(monkeypatch):
    """Fixture to mock whisper module with correct patch target."""
    # Mock whisper module at the location where it's imported in perception.py
    mock_whisper_module = Mock()
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "text": "Hello, I have a billing question",
        "language": "en"
    }
    mock_whisper_module.load_model.return_value = mock_model
    
    # Patch whisper at the location where it's imported in core.perception
    monkeypatch.setattr("core.perception.whisper", mock_whisper_module)
    
    return {
        "whisper_module": mock_whisper_module,
        "whisper_model": mock_model
    }

@pytest.fixture
def mock_openrouter_api(monkeypatch):
    """Fixture to mock OpenRouter API requests."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 100}
    }
    
    # Patch requests.post for OpenRouter API calls
    monkeypatch.setattr("requests.post", Mock(return_value=mock_response))
    
    return mock_response

@pytest.fixture
def mock_gtts(monkeypatch):
    """Fixture to mock gTTS if present."""
    try:
        mock_gtts_class = Mock()
        mock_gtts_instance = Mock()
        mock_gtts_instance.save = Mock()
        mock_gtts_class.return_value = mock_gtts_instance
        monkeypatch.setattr("gtts.gTTS", mock_gtts_class)
        return mock_gtts_instance
    except ImportError:
        return None

@pytest.fixture
def mock_modules(mock_whisper, mock_openrouter_api, mock_gtts):
    """Master fixture that combines all mocked modules."""
    return {
        "whisper_model": mock_whisper["whisper_model"],
        "requests_response": mock_openrouter_api,
        "gtts": mock_gtts
    }

@pytest.fixture
def initialized_memory_module(temp_db):
    """Fixture to provide initialized MemoryModule with temporary database."""
    return MemoryModule(db_path=temp_db)

@pytest.fixture
def initialized_perception_module(mock_modules):
    """Fixture to provide initialized PerceptionModule with mocked dependencies."""
    return PerceptionModule()

@pytest.fixture
def initialized_kural_agent(api_key, mock_modules):
    """Fixture to provide initialized KuralAgent with properly configured mock tools."""
    # Create "intelligent" mocks for the tools with correct name attributes
    mock_billing_tool = MagicMock()
    mock_billing_tool.name = "get_billing_info"
    mock_billing_tool.description = "Retrieve customer billing information"
    mock_billing_tool.invoke = Mock(return_value="Mock billing info")
    
    mock_network_tool = MagicMock()
    mock_network_tool.name = "check_network_status"
    mock_network_tool.description = "Check network status for specific area codes"
    mock_network_tool.invoke = Mock(return_value="Mock network status")
    
    # Create list of properly configured mock tools
    mock_tools = [mock_billing_tool, mock_network_tool]
    
    return KuralAgent(openrouter_api_key=api_key, tools=mock_tools)

# ===== TEST CASES =====

def test_tools_functionality():
    """Test that all tools return expected string formats with valid inputs using modern .invoke() method."""
    # Test get_billing_info with valid input using modern invoke method
    billing_result = get_billing_info.invoke({"user_id": "TEST_USER_123"})
    assert isinstance(billing_result, str)
    assert "Billing Information" in billing_result
    assert "Customer ID: TEST_USER_123" in billing_result
    assert "Current Bill Amount:" in billing_result
    assert "Due Date:" in billing_result
    
    # Test check_network_status with valid input using modern invoke method
    network_result = check_network_status.invoke({"area_code": "555"})
    assert isinstance(network_result, str)
    assert "Network Status for Area Code 555:" in network_result
    assert any(status in network_result for status in ["NORMAL", "OUTAGE", "DEGRADED", "optimal", "good", "fair", "poor"])

def test_tools_edge_cases():
    """Test edge cases and error conditions for raw tool functions - bypasses Pydantic validation."""
    # Test _get_billing_info with None input - should return error message
    result = _get_billing_info(None)
    assert isinstance(result, str)
    assert "Error:" in result
    assert "valid user ID must be provided" in result
    
    # Test _get_billing_info with non-string input - should return error message
    result = _get_billing_info(123)
    assert isinstance(result, str)
    assert "Error:" in result
    assert "Invalid user ID format" in result
    
    # Test _get_billing_info with empty string - should return error message
    result = _get_billing_info("")
    assert isinstance(result, str)
    assert "Error:" in result
    assert "cannot be empty" in result
    
    # Test _check_network_status with None input - should return error message
    result = _check_network_status(None)
    assert isinstance(result, str)
    assert "Error:" in result
    assert "valid area code must be provided" in result
    
    # Test _check_network_status with non-string input - should return error message
    result = _check_network_status(123)
    assert isinstance(result, str)
    assert "Error:" in result
    assert "Invalid area code format" in result
    
    # Test _check_network_status with invalid length - should return error message
    result = _check_network_status("12")  # Too short
    assert isinstance(result, str)
    assert "Error:" in result
    assert "not a valid area code" in result
    
    result = _check_network_status("1234")  # Too long
    assert isinstance(result, str)
    assert "Error:" in result
    assert "not a valid area code" in result
    
    # Test _check_network_status with non-numeric input - should return error message
    result = _check_network_status("55a")  # Contains letters
    assert isinstance(result, str)
    assert "Error:" in result
    assert "non-numeric characters" in result

def test_perception_sentiment_analysis(initialized_perception_module, mock_modules):
    """Test sentiment analysis with different API responses."""
    perception = initialized_perception_module
    
    # Test Positive sentiment
    mock_modules["requests_response"].json.return_value = {
        "choices": [{"message": {"content": "Positive"}}]
    }
    
    result = perception.analyze_sentiment("I love this service!")
    assert result == "Positive"
    
    # Test Negative sentiment
    mock_modules["requests_response"].json.return_value = {
        "choices": [{"message": {"content": "Negative"}}]
    }
    
    result = perception.analyze_sentiment("This is terrible!")
    assert result == "Negative"
    
    # Test Neutral sentiment
    mock_modules["requests_response"].json.return_value = {
        "choices": [{"message": {"content": "Neutral"}}]
    }
    
    result = perception.analyze_sentiment("What is my balance?")
    assert result == "Neutral"
    
    # Test API error - should return Neutral as fallback
    mock_modules["requests_response"].status_code = 500
    result = perception.analyze_sentiment("Some text")
    assert result == "Neutral"
    
    # Test empty text
    result = perception.analyze_sentiment("")
    assert result == "Neutral"

def test_memory_lifecycle(initialized_memory_module, mock_modules):
    """Test complete memory lifecycle operations."""
    memory = initialized_memory_module
    user_id = "test_user_123"
    
    # Test get_long_term_summary for new user
    summary = memory.get_long_term_summary(user_id)
    assert summary == ""
    
    # Mock API response for summary generation
    mock_modules["requests_response"].json.return_value = {
        "choices": [{"message": {"content": "Test summary"}}]
    }
    
    # Create mock chat history
    chat_history = [
        HumanMessage(content="Hello, I need help with my bill"),
        AIMessage(content="I can help you with billing questions"),
        HumanMessage(content="What is my current balance?"),
        AIMessage(content="Your current balance is $45.99")
    ]
    
    # Save conversation summary (now uses environment variable for API key)
    success = memory.save_conversation_summary(user_id, chat_history)
    assert success is True
    
    # Get long-term summary after saving
    summary = memory.get_long_term_summary(user_id)
    assert summary == "Test summary"
    
    # Verify database file contains correct data
    with open(memory.db_path, 'r') as f:
        users_data = json.load(f)
    
    assert user_id in users_data
    assert users_data[user_id]["summary"] == "Test summary"
    assert "last_updated" in users_data[user_id]
    assert users_data[user_id]["conversation_count"] == 1

def test_agent_persona_loading_logic(initialized_kural_agent, monkeypatch):
    """Test that agent loads correct persona based on sentiment."""
    agent = initialized_kural_agent
    
    # Mock the simple agent execution to capture the master prompt
    captured_prompts = []
    
    def mock_simple_execution(user_input, master_prompt):
        captured_prompts.append(master_prompt)
        return "Mock response"
    
    monkeypatch.setattr(agent, '_simple_agent_execution', mock_simple_execution)
    
    # Test Negative sentiment - should load empathetic_deescalation persona
    agent.run(
        user_id="test_user",
        user_input="I'm very frustrated with this service!",
        sentiment="Negative"
    )
    
    assert len(captured_prompts) == 1
    # Check that empathetic deescalation keywords are present
    prompt = captured_prompts[0].lower()
    assert any(keyword in prompt for keyword in ["empathetic", "understanding", "apologize", "deescalation"])
    
    # Test Positive sentiment - should load efficient_friendly persona
    captured_prompts.clear()
    agent.run(
        user_id="test_user",
        user_input="Thank you for the great service!",
        sentiment="Positive"
    )
    
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0].lower()
    assert any(keyword in prompt for keyword in ["efficient", "friendly", "positive", "helpful"])
    
    # Test Neutral sentiment - should load professional_direct persona
    captured_prompts.clear()
    agent.run(
        user_id="test_user",
        user_input="What are my account details?",
        sentiment="Neutral"
    )
    
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0].lower()
    assert any(keyword in prompt for keyword in ["professional", "direct", "clear", "accurate"])

def test_perception_audio_transcription(initialized_perception_module, mock_modules):
    """Test audio transcription functionality."""
    perception = initialized_perception_module
    
    # Mock successful transcription
    mock_modules["whisper_model"].transcribe.return_value = {
        "text": "Hello, I need help with my billing",
        "language": "en"
    }
    
    # Test with valid audio file (mock file path)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio.write(b"mock audio data")
        temp_audio_path = temp_audio.name
    
    try:
        result = perception.transcribe_audio(temp_audio_path)
        assert result["text"] == "Hello, I need help with my billing"
        assert result["language"] == "en"
        assert result["language_name"] == "English"
        assert "error" not in result
    finally:
        os.unlink(temp_audio_path)
    
    # Test with non-existent file
    result = perception.transcribe_audio("non_existent_file.wav")
    assert result["text"] == ""
    assert result["language"] == "en"
    assert "error" in result
    assert "Audio file not found" in result["error"]

def test_memory_error_handling(initialized_memory_module, mock_modules):
    """Test memory module error handling scenarios."""
    memory = initialized_memory_module
    
    # Test with empty user_id
    summary = memory.get_long_term_summary("")
    assert summary == ""
    
    # Test save_conversation_summary with empty user_id
    success = memory.save_conversation_summary("", [])
    assert success is False
    
    # Test save_conversation_summary with empty chat history
    success = memory.save_conversation_summary("test_user", [])
    assert success is True  # Should succeed with empty history

def test_agent_health_check(initialized_kural_agent):
    """Test agent health check functionality."""
    agent = initialized_kural_agent
    
    health_status = agent.health_check()
    
    assert health_status["agent_initialized"] is True
    assert health_status["tools_available"] == 2
    assert "get_billing_info" in health_status["tool_names"]
    assert "check_network_status" in health_status["tool_names"]
    assert health_status["api_key_present"] is True
    assert "persona_files" in health_status
    
    # Check persona file status
    persona_files = health_status["persona_files"]
    assert "Negative" in persona_files
    assert "Positive" in persona_files
    assert "Neutral" in persona_files

def test_chat_openrouter_integration(api_key, mock_modules):
    """Test ChatOpenRouter class functionality."""
    chat = ChatOpenRouter()
    
    # Test successful API call
    mock_modules["requests_response"].json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 50}
    }
    
    messages = [{"role": "user", "content": "Hello"}]
    result = chat.invoke(messages)
    
    assert result["content"] == "Test response"
    assert result["usage"]["total_tokens"] == 50
    assert result["model"] == "mistralai/mistral-7b-instruct"
    
    # Test API error
    mock_modules["requests_response"].status_code = 500
    result = chat.invoke(messages)
    
    assert "error" in result
    assert "trouble processing" in result["content"]

def test_memory_user_stats(initialized_memory_module):
    """Test memory user statistics functionality."""
    memory = initialized_memory_module
    
    # Test stats for non-existent user
    stats = memory.get_user_stats("non_existent_user")
    assert stats["user_id"] == "non_existent_user"
    assert stats["conversation_count"] == 0
    assert stats["last_updated"] == ""
    assert stats["has_summary"] is False
    
    # Add a user manually to test existing user stats
    user_data = {
        "test_user": {
            "summary": "Test summary",
            "last_updated": datetime.now().isoformat(),
            "conversation_count": 3
        }
    }
    
    with open(memory.db_path, 'w') as f:
        json.dump(user_data, f)
    
    stats = memory.get_user_stats("test_user")
    assert stats["user_id"] == "test_user"
    assert stats["conversation_count"] == 3
    assert stats["has_summary"] is True
    assert stats["last_updated"] != ""

def test_robust_tool_validation():
    """Test that raw tool functions handle all edge cases gracefully without raising exceptions."""
    # Test various invalid inputs for _get_billing_info
    invalid_inputs = [None, "", "  ", 123, [], {}, True, False]
    
    for invalid_input in invalid_inputs:
        result = _get_billing_info(invalid_input)
        assert isinstance(result, str)
        assert "Error:" in result
        # Should not raise any exceptions
    
    # Test various invalid inputs for _check_network_status
    for invalid_input in invalid_inputs:
        result = _check_network_status(invalid_input)
        assert isinstance(result, str)
        assert "Error:" in result
        # Should not raise any exceptions
    
    # Test specific invalid formats
    invalid_area_codes = ["12", "1234", "abc", "55a", "a55"]
    for invalid_code in invalid_area_codes:
        result = _check_network_status(invalid_code)
        assert isinstance(result, str)
        assert "Error:" in result

# ===== INTEGRATION TESTS =====

def test_full_workflow_integration(initialized_kural_agent, initialized_memory_module, 
                                 initialized_perception_module, mock_modules):
    """Test complete workflow from perception to agent response."""
    agent = initialized_kural_agent
    memory = initialized_memory_module
    perception = initialized_perception_module
    
    # Mock sentiment analysis
    mock_modules["requests_response"].json.return_value = {
        "choices": [{"message": {"content": "Negative"}}]
    }
    
    # Analyze sentiment
    sentiment = perception.analyze_sentiment("I'm very frustrated!")
    assert sentiment == "Negative"
    
    # Mock agent response
    captured_prompts = []
    def mock_execution(user_input, master_prompt):
        captured_prompts.append(master_prompt)
        return "I understand your frustration. Let me help you."
    
    import unittest.mock
    with unittest.mock.patch.object(agent, '_simple_agent_execution', mock_execution):
        response = agent.run(
            user_id="test_user",
            user_input="I'm very frustrated!",
            sentiment=sentiment
        )
    
    assert response == "I understand your frustration. Let me help you."
    assert len(captured_prompts) == 1
    assert "empathetic" in captured_prompts[0].lower() or "understanding" in captured_prompts[0].lower()

def test_decorated_tools_with_invoke():
    """Test that decorated tools work correctly with the modern .invoke() method."""
    # Test valid inputs with decorated tools
    result = get_billing_info.invoke({"user_id": "VALID_USER_123"})
    assert isinstance(result, str)
    assert "Billing Information" in result
    assert "VALID_USER_123" in result
    
    result = check_network_status.invoke({"area_code": "555"})
    assert isinstance(result, str)
    assert "Network Status for Area Code 555" in result
    
    # Test that decorated tools have the expected attributes
    assert hasattr(get_billing_info, 'name')
    assert hasattr(check_network_status, 'name')
    assert hasattr(get_billing_info, 'invoke')
    assert hasattr(check_network_status, 'invoke')