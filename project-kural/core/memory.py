"""
Memory Module for Conversation Management

This module handles both short-term conversation memory (current session)
and long-term memory (user conversation summaries stored in JSON).
"""

import json
import os
import logging
import requests
from typing import Dict, List, Optional
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define BASE_DIR for cross-platform file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to project-kural root


class MemoryModule:
    """
    Manages conversation memory for customer service interactions.
    Handles both short-term (session) and long-term (persistent) memory.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the memory module.
        
        Args:
            db_path (str): Path to the user database JSON file
        """
        if db_path is None:
            self.db_path = os.path.join(BASE_DIR, "user_database", "users.json")
        else:
            self.db_path = db_path
        
        self.file_lock = threading.Lock()
        
        # Ensure the database file exists
        self._ensure_db_exists()
        
        logger.info(f"Memory module initialized with database: {self.db_path}")
    
    def _ensure_db_exists(self) -> None:
        """Ensure the user database file exists with proper structure."""
        try:
            if not os.path.exists(self.db_path):
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                
                # Create empty database file
                with open(self.db_path, 'w') as f:
                    json.dump({}, f, indent=2)
                    
                logger.info(f"Created new database file: {self.db_path}")
        except OSError as e:
            logger.error(f"Failed to create database file: {e}")
            raise RuntimeError(f"Could not initialize database: {e}")
    
    def get_short_term_memory(self) -> ConversationBufferMemory:
        """
        Create and return a new ConversationBufferMemory instance for the current session.
        
        Returns:
            ConversationBufferMemory: New memory instance for current conversation
        """
        try:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            )
            
            logger.info("Created new short-term memory instance")
            return memory
            
        except ImportError as e:
            logger.error(f"Failed to create short-term memory - LangChain not available: {e}")
            # Return a basic memory instance as fallback
            return ConversationBufferMemory(memory_key="chat_history")
    
    def get_long_term_summary(self, user_id: str) -> str:
        """
        Retrieve long-term conversation summary for a specific user.
        
        Args:
            user_id (str): The user identifier
            
        Returns:
            str: User's conversation summary or empty string if not found
        """
        if not user_id:
            logger.warning("Empty user_id provided to get_long_term_summary")
            return ""
        
        try:
            with self.file_lock:
                with open(self.db_path, 'r') as f:
                    users_data = json.load(f)
                
                user_summary = users_data.get(user_id, {})
                
                if isinstance(user_summary, dict):
                    summary = user_summary.get("summary", "")
                    last_updated = user_summary.get("last_updated", "")
                    
                    if summary:
                        logger.info(f"Retrieved summary for user {user_id} (last updated: {last_updated})")
                        return summary
                    else:
                        logger.info(f"No summary found for user {user_id}")
                        return ""
                else:
                    # Handle legacy format where summary was stored as string
                    logger.info(f"Retrieved legacy summary for user {user_id}")
                    return str(user_summary) if user_summary else ""
                    
        except FileNotFoundError:
            logger.warning(f"Database file not found: {self.db_path}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in database file: {e}")
            return ""
        except IOError as e:
            logger.error(f"I/O error retrieving summary for user {user_id}: {e}")
            return ""
    
    def save_conversation_summary(self, user_id: str, chat_history: List[BaseMessage]) -> bool:
        """
        Generate and save a conversation summary for a user.
        
        Args:
            user_id (str): The user identifier
            chat_history (List[BaseMessage]): List of conversation messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not user_id:
            logger.warning("Empty user_id provided to save_conversation_summary")
            return False
        
        if not chat_history:
            logger.info("No chat history to summarize")
            return True
        
        # Get API key from environment
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            return False
        
        try:
            # Convert chat history to text format
            conversation_text = self._format_chat_history(chat_history)
            
            # Generate summary using OpenRouter
            summary = self._generate_summary(conversation_text, openrouter_api_key)
            
            if not summary:
                logger.warning("Failed to generate conversation summary")
                return False
            
            # Save summary to database
            return self._save_summary_to_db(user_id, summary)
            
        except KeyError as e:
            logger.error(f"Missing required data for summary generation: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid data format for summary generation: {e}")
            return False
    
    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        """
        Format chat history into readable text for summarization.
        
        Args:
            chat_history (List[BaseMessage]): List of conversation messages
            
        Returns:
            str: Formatted conversation text
        """
        formatted_messages = []
        
        for message in chat_history:
            # Get message type and content
            message_type = message.__class__.__name__
            content = message.content
            
            if "Human" in message_type or "User" in message_type:
                formatted_messages.append(f"Customer: {content}")
            elif "AI" in message_type or "Assistant" in message_type:
                formatted_messages.append(f"Agent: {content}")
            else:
                formatted_messages.append(f"System: {content}")
        
        return "\n".join(formatted_messages)
    
    def _generate_summary(self, conversation_text: str, openrouter_api_key: str) -> Optional[str]:
        """
        Generate conversation summary using OpenRouter API.
        
        Args:
            conversation_text (str): The conversation to summarize
            openrouter_api_key (str): OpenRouter API key
            
        Returns:
            Optional[str]: Generated summary or None if failed
        """
        try:
            summary_prompt = f"""
            Summarize this customer service conversation in 2-3 sentences for future context.
            Focus on the customer's main issues, any resolutions provided, and their overall satisfaction.
            
            Conversation:
            {conversation_text}
            
            Provide a concise summary that will help future agents understand this customer's history.
            """
            
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }
            
            logger.info("Generating conversation summary with OpenRouter")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                summary = response_data["choices"][0]["message"]["content"].strip()
                
                logger.info("Conversation summary generated successfully")
                return summary
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("OpenRouter API request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"OpenRouter API connection failed: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            return None
        except KeyError as e:
            logger.error(f"Invalid API response format: {e}")
            return None
    
    def _save_summary_to_db(self, user_id: str, summary: str) -> bool:
        """
        Save user summary to database file.
        
        Args:
            user_id (str): The user identifier
            summary (str): The conversation summary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.file_lock:
                # Read current data
                try:
                    with open(self.db_path, 'r') as f:
                        users_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    users_data = {}
                
                # Update user data
                users_data[user_id] = {
                    "summary": summary,
                    "last_updated": datetime.now().isoformat(),
                    "conversation_count": users_data.get(user_id, {}).get("conversation_count", 0) + 1
                }
                
                # Write back to file
                with open(self.db_path, 'w') as f:
                    json.dump(users_data, f, indent=2)
                
                logger.info(f"Summary saved successfully for user {user_id}")
                return True
                
        except IOError as e:
            logger.error(f"Failed to save summary to database: {e}")
            return False
        except json.JSONEncodeError as e:
            logger.error(f"Failed to encode summary data: {e}")
            return False
    
    def get_user_stats(self, user_id: str) -> Dict:
        """
        Get statistics for a specific user.
        
        Args:
            user_id (str): The user identifier
            
        Returns:
            Dict: User statistics including conversation count and last updated
        """
        try:
            with self.file_lock:
                with open(self.db_path, 'r') as f:
                    users_data = json.load(f)
                
                user_data = users_data.get(user_id, {})
                
                return {
                    "user_id": user_id,
                    "conversation_count": user_data.get("conversation_count", 0),
                    "last_updated": user_data.get("last_updated", ""),
                    "has_summary": bool(user_data.get("summary", ""))
                }
                
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            logger.error(f"Error retrieving user stats: {e}")
            return {"user_id": user_id, "conversation_count": 0, "last_updated": "", "has_summary": False}