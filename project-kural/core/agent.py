"""
Central Agent Orchestrator for Project Kural

This module contains the main KuralAgent class that coordinates all components:
persona management, memory integration, tool usage, LLM interaction, and knowledge base retrieval.
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define BASE_DIR for cross-platform file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to project-kural root


class ChatOpenRouter:
    """
    Custom ChatOpenRouter implementation for OpenRouter API integration.
    """
    
    def __init__(self, model: str = "mistralai/mistral-7b-instruct"):
        # Get API key from environment
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Invoke the OpenRouter API with messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dict containing the response
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert messages to OpenRouter format
            api_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    api_messages.append(msg)
                else:
                    # Handle LangChain message objects
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    role = "user" if "Human" in str(type(msg)) else "assistant"
                    api_messages.append({"role": role, "content": content})
            
            data = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return {
                    "content": response_data["choices"][0]["message"]["content"],
                    "usage": response_data.get("usage", {}),
                    "model": self.model
                }
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return {"content": "I apologize, but I'm having trouble processing your request right now. Please try again.", "error": True}
                
        except requests.exceptions.Timeout:
            logger.error("OpenRouter API request timed out")
            return {"content": "I apologize, but the request timed out. Please try again.", "error": True}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"OpenRouter API connection failed: {e}")
            return {"content": "I apologize, but I'm experiencing connection issues. Please try again.", "error": True}
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            return {"content": "I apologize, but I'm experiencing technical difficulties. Please try again.", "error": True}
        except KeyError as e:
            logger.error(f"Invalid API response format: {e}")
            return {"content": "I apologize, but I received an invalid response. Please try again.", "error": True}


class KuralAgent:
    """
    Main agent orchestrator that combines all Project Kural components with knowledge base retrieval.
    """
    
    def __init__(self, openrouter_api_key: str, tools: List[BaseTool], vector_store=None):
        """
        Initialize the Kural Agent.
        
        Args:
            openrouter_api_key (str): OpenRouter API key for LLM access (deprecated - use environment variable)
            tools (List[BaseTool]): List of tools the agent can use
            vector_store: Optional vector store for knowledge base retrieval
        """
        # Store API key for backward compatibility but log deprecation warning
        if openrouter_api_key:
            logger.warning("Passing API key as parameter is deprecated. Use OPENROUTER_API_KEY environment variable instead.")
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        
        self.tools = tools
        self.llm = ChatOpenRouter()
        self.vector_store = vector_store
        
        # Define persona file mappings with cross-platform paths
        self.persona_mappings = {
            "Negative": os.path.join(BASE_DIR, "personas", "empathetic_deescalation.txt"),
            "Positive": os.path.join(BASE_DIR, "personas", "efficient_friendly.txt"), 
            "Neutral": os.path.join(BASE_DIR, "personas", "professional_direct.txt")
        }
        
        logger.info("KuralAgent initialized successfully")
    
    def _load_persona_prompt(self, sentiment: str) -> str:
        """
        Load the appropriate persona prompt based on sentiment.
        
        Args:
            sentiment (str): Detected sentiment ('Negative', 'Positive', 'Neutral')
            
        Returns:
            str: Persona prompt content
        """
        persona_file = self.persona_mappings.get(sentiment, self.persona_mappings["Neutral"])
        
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona_prompt = f.read().strip()
            
            logger.info(f"Loaded {sentiment} persona from {persona_file}")
            return persona_prompt
            
        except FileNotFoundError:
            logger.error(f"Persona file not found: {persona_file}")
            # Return a default professional persona
            return """
            You are a professional customer service representative. 
            Provide helpful, accurate, and courteous assistance to customers.
            Be clear, concise, and focus on resolving their issues effectively.
            """
        except IOError as e:
            logger.error(f"I/O error loading persona file {persona_file}: {e}")
            return "You are a helpful customer service representative."
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error loading persona file {persona_file}: {e}")
            return "You are a helpful customer service representative."
    
    def _retrieve_knowledge_base_response(self, user_input: str) -> Optional[str]:
        """
        Retrieve pre-approved response from knowledge base using vector similarity search.
        
        Args:
            user_input (str): User's input query
            
        Returns:
            Optional[str]: Retrieved response template or None if not found
        """
        try:
            if not self.vector_store:
                logger.info("No vector store available for knowledge base retrieval")
                return None
            
            # Perform similarity search to find most relevant document
            similar_docs = self.vector_store.similarity_search(user_input, k=1)
            
            if not similar_docs:
                logger.info("No similar documents found in knowledge base")
                return None
            
            # Get the most relevant document
            best_doc = similar_docs[0]
            
            # Extract the pre-approved response from metadata
            response_template = best_doc.metadata.get('response')
            
            if not response_template:
                logger.warning("No response found in document metadata")
                return None
            
            logger.info(f"Retrieved knowledge base response for query: '{user_input[:50]}...'")
            return response_template
            
        except Exception as e:
            logger.error(f"Knowledge base retrieval failed: {e}")
            return None
    
    def _format_knowledge_base_response(self, user_input: str, response_template: str, persona_prompt: str, language: str) -> str:
        """
        Format the knowledge base response template using LLM for personalization and placeholder filling.
        
        Args:
            user_input (str): Original user question
            response_template (str): Pre-approved response template
            persona_prompt (str): Persona-specific instructions
            language (str): Target language for response
            
        Returns:
            str: Formatted and personalized response
        """
        try:
            # Language response instructions
            language_instructions = {
                "en": "Respond in English.",
                "ta": "Respond in Tamil (தமிழ்).",
                "hi": "Respond in Hindi (हिंदी).",
                "es": "Respond in Spanish.",
                "fr": "Respond in French."
            }
            
            language_instruction = language_instructions.get(language, "Respond in English.")
            
            # Create formatting prompt for the LLM
            formatting_prompt = f"""
{persona_prompt}

TASK: Format and personalize the following pre-approved response template for the customer.

CUSTOMER QUESTION: "{user_input}"

PRE-APPROVED RESPONSE TEMPLATE: "{response_template}"

INSTRUCTIONS:
1. Use the pre-approved response template as your base - do not change the core information or policies
2. Personalize the response to address the customer's specific question
3. If the customer mentioned specific details (names, order numbers, dates), incorporate them naturally
4. Fill in any {{placeholder}} values with appropriate information from the customer's question
5. Maintain the tone specified in your persona guidelines
6. {language_instruction}
7. Keep the response helpful, accurate, and professional

IMPORTANT: Your response should be based on the pre-approved template. Do not invent new information or policies.
"""
            
            # Send to LLM for formatting
            messages = [
                {"role": "system", "content": formatting_prompt},
                {"role": "user", "content": f"Please format the response for: {user_input}"}
            ]
            
            response = self.llm.invoke(messages)
            
            if response.get("error"):
                logger.warning("LLM formatting failed, returning template as-is")
                return response_template
            
            return response["content"]
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return response_template
    
    def _construct_master_prompt(self, persona_prompt: str, long_term_summary: str, 
                               language: str) -> str:
        """
        Construct the master system prompt combining persona, context, and instructions.
        
        Args:
            persona_prompt (str): The persona-specific prompt
            long_term_summary (str): User's conversation history summary
            language (str): Detected language for response
            
        Returns:
            str: Complete system prompt
        """
        # Language response instructions
        language_instructions = {
            "en": "Respond in English.",
            "ta": "Respond in Tamil (தமிழ்).",
            "hi": "Respond in Hindi (हिंदी).",
            "es": "Respond in Spanish.",
            "fr": "Respond in French."
        }
        
        language_instruction = language_instructions.get(language, "Respond in English.")
        
        # Construct the master prompt
        master_prompt = f"""
{persona_prompt}

IMPORTANT CONTEXT:
{f"Previous context with this user: {long_term_summary}" if long_term_summary else "This is a new customer interaction."}

LANGUAGE INSTRUCTION:
{language_instruction}

AVAILABLE TOOLS:
You have access to tools that can help you:
- get_billing_info: Retrieve customer billing information
- check_network_status: Check network status for specific area codes

RESPONSE GUIDELINES:
1. Always be helpful and professional
2. Use tools when appropriate to provide accurate information
3. Respond in the detected language ({language})
4. Follow the persona guidelines above
5. If you cannot help with something, explain why and suggest alternatives
6. Keep responses conversational and natural

Remember: You are representing the company, so maintain high standards of service while following your persona guidelines.
"""
        
        return master_prompt.strip()
    
    def _simple_agent_execution(self, user_input: str, master_prompt: str) -> str:
        """
        Execute a simple agent interaction without complex ReAct patterns.
        
        Args:
            user_input (str): User's input message
            master_prompt (str): Complete system prompt
            
        Returns:
            str: Agent's response
        """
        try:
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": master_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            if response.get("error"):
                return response["content"]
            
            agent_response = response["content"]
            
            # Check if the agent wants to use tools
            if "get_billing_info" in agent_response.lower() or "billing" in user_input.lower():
                # Extract user ID and call billing tool
                for tool in self.tools:
                    if tool.name == "get_billing_info":
                        # For demo, use a default user ID
                        billing_info = tool.invoke({"user_id": "DEMO_USER"})
                        agent_response += f"\n\n{billing_info}"
                        break
            
            if "check_network_status" in agent_response.lower() or "network" in user_input.lower():
                # Extract area code and call network tool
                for tool in self.tools:
                    if tool.name == "check_network_status":
                        # For demo, use a default area code
                        network_info = tool.invoke({"area_code": "555"})
                        agent_response += f"\n\n{network_info}"
                        break
            
            return agent_response
            
        except ValueError as e:
            logger.error(f"Invalid input for agent execution: {e}")
            return "I apologize, but I couldn't process your request. Please try rephrasing your question."
        except TypeError as e:
            logger.error(f"Type error in agent execution: {e}")
            return "I apologize, but I encountered a processing error. Please try again."
    
    def run(self, user_id: str, user_input: str, language: str = "en", 
            sentiment: str = "Neutral", short_term_memory: Optional[ConversationBufferMemory] = None,
            long_term_summary: str = "") -> str:
        """
        Main agent execution method that coordinates all components with knowledge base integration.
        
        Args:
            user_id (str): User identifier
            user_input (str): User's input message
            language (str): Detected language code
            sentiment (str): Detected sentiment
            short_term_memory (Optional[ConversationBufferMemory]): Session memory
            long_term_summary (str): User's conversation history summary
            
        Returns:
            str: Agent's response
        """
        try:
            logger.info(f"Processing request for user {user_id} with sentiment {sentiment}")
            
            # Load appropriate persona
            persona_prompt = self._load_persona_prompt(sentiment)
            
            # First, try to retrieve a response from the knowledge base
            knowledge_base_response = self._retrieve_knowledge_base_response(user_input)
            
            if knowledge_base_response:
                # Format the knowledge base response using the persona and context
                response = self._format_knowledge_base_response(
                    user_input, knowledge_base_response, persona_prompt, language
                )
                logger.info("Used knowledge base response")
            else:
                # Fall back to general agent execution
                master_prompt = self._construct_master_prompt(
                    persona_prompt, long_term_summary, language
                )
                response = self._simple_agent_execution(user_input, master_prompt)
                logger.info("Used general agent execution")
            
            # Update short-term memory if provided
            if short_term_memory:
                try:
                    short_term_memory.chat_memory.add_user_message(user_input)
                    short_term_memory.chat_memory.add_ai_message(response)
                except AttributeError as e:
                    logger.warning(f"Failed to update short-term memory - invalid memory object: {e}")
                except ImportError as e:
                    logger.warning(f"Failed to update short-term memory - LangChain not available: {e}")
            
            logger.info(f"Successfully processed request for user {user_id}")
            return response
            
        except ValueError as e:
            logger.error(f"Invalid input for agent execution (user {user_id}): {e}")
            return "I apologize, but I couldn't process your request. Please check your input and try again."
        except RuntimeError as e:
            logger.error(f"Runtime error in agent execution (user {user_id}): {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team."
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List[str]: List of tool names
        """
        return [tool.name for tool in self.tools]
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent components.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_status = {
            "agent_initialized": True,
            "tools_available": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],  # Fixed: Return list of strings, not objects
            "persona_files": {},
            "api_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
            "vector_store_available": self.vector_store is not None
        }
        
        # Check persona files
        for sentiment, file_path in self.persona_mappings.items():
            health_status["persona_files"][sentiment] = os.path.exists(file_path)
        
        # Check vector store stats if available
        if self.vector_store:
            try:
                health_status["vector_store_stats"] = self.vector_store.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get vector store stats: {e}")
                health_status["vector_store_stats"] = {"error": str(e)}
        
        return health_status