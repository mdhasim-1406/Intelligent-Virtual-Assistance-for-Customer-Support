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
    
    def __init__(self, model: str = "google/gemini-flash-1.5"):
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
        FINAL RE-ARCHITECTURE: Two-Step "Analyst-Translator" Chain
        
        This method implements a revolutionary two-step approach:
        Step 1: "Analyst" - Produces clean English response with perfect logic
        Step 2: "Translator" - Converts to native-level target language if needed
        
        Args:
            user_id (str): User identifier
            user_input (str): User's input message
            language (str): Detected language code
            sentiment (str): Detected sentiment
            short_term_memory (Optional[ConversationBufferMemory]): Session memory
            long_term_summary (str): User's conversation history summary
            
        Returns:
            str: Agent's response in the user's native language
        """
        try:
            logger.info(f"🔬 ANALYST-TRANSLATOR CHAIN: Processing user {user_id} | Language: {language} | Sentiment: {sentiment}")
            
            # Load appropriate persona for context
            persona_prompt = self._load_persona_prompt(sentiment)
            
            # Retrieve knowledge base response if available
            retrieved_response_template = self._retrieve_knowledge_base_response(user_input)
            
            # STEP 1: THE "ANALYST" CALL
            # Construct English-focused prompt for clean logic and response
            analyst_prompt_template = f"""
You are a highly intelligent customer service analyst. Your task is to determine the correct response to a user's query.

CONTEXT:
- User's Sentiment: {sentiment}
- User's Statement: "{user_input}"
- Most Relevant Knowledge Base Entry: "{retrieved_response_template if retrieved_response_template else 'No relevant entry found'}"

YOUR MISSION:
1. Analyze if the "Relevant Knowledge Base Entry" is a logical response to the "User's Statement".
2. If it is relevant, your output should be the text from that template.
3. If it is NOT relevant, your output should be a new, polite, empathetic, and helpful response that directly addresses the user's statement.

**Your final output for this step MUST be ONLY the clean, user-facing response text in ENGLISH.** Do not include any other analysis or explanation.
"""
            
            # Make the first LLM call to get English response
            analyst_messages = [
                {"role": "user", "content": analyst_prompt_template}
            ]
            
            analyst_response = self.llm.invoke(analyst_messages)
            
            if analyst_response.get("error"):
                logger.error("❌ Analyst step failed")
                english_response = "I apologize, but I'm experiencing technical difficulties. Please try again."
            else:
                english_response = analyst_response["content"].strip()
                logger.info(f"✅ Analyst step completed: '{english_response[:50]}...'")
            
            # STEP 2: THE "TRANSLATOR" LOGIC
            if language == 'en':
                # Mission complete - return English response
                logger.info("🎯 Language is English - Mission complete")
                final_response = english_response
            else:
                # STEP 3: THE "TRANSLATOR" CALL
                # Language code to full name mapping
                language_mapping = {
                    'ta': 'Tamil (தமிழ்)',
                    'hi': 'Hindi (हिंदी)', 
                    'es': 'Spanish (Español)',
                    'fr': 'French (Français)',
                    'de': 'German (Deutsch)',
                    'it': 'Italian (Italiano)',
                    'pt': 'Portuguese (Português)',
                    'ru': 'Russian (Русский)',
                    'ja': 'Japanese (日本語)',
                    'ko': 'Korean (한국어)',
                    'zh': 'Chinese (中文)',
                    'ar': 'Arabic (العربية)'
                }
                
                target_language_full_name = language_mapping.get(language, f'the language with code {language}')
                
                # Construct translation prompt
                translator_prompt_template = f"""
You are a world-class, native-level translator. Your ONLY task is to translate the following English text into natural, fluent, and grammatically perfect {target_language_full_name}.

Your final output MUST ONLY be the translation. Do not add any extra text, commentary, or apologies.

ENGLISH TEXT TO TRANSLATE:
"{english_response}"
"""
                
                # Make the second LLM call for translation
                translator_messages = [
                    {"role": "user", "content": translator_prompt_template}
                ]
                
                translator_response = self.llm.invoke(translator_messages)
                
                if translator_response.get("error"):
                    logger.error(f"❌ Translator step failed for language: {language}")
                    final_response = english_response  # Fallback to English
                else:
                    final_response = translator_response["content"].strip()
                    logger.info(f"✅ Translator step completed: {language} translation generated")
            
            # Update short-term memory if provided
            if short_term_memory:
                try:
                    short_term_memory.chat_memory.add_user_message(user_input)
                    short_term_memory.chat_memory.add_ai_message(final_response)
                except Exception as e:
                    logger.warning(f"⚠️ Memory update failed: {e}")
            
            logger.info(f"🎉 ANALYST-TRANSLATOR CHAIN COMPLETE for user {user_id}")
            return final_response
            
        except Exception as e:
            logger.error(f"💥 CRITICAL FAILURE in Analyst-Translator chain for user {user_id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def _construct_master_prompt_with_language_enforcement(self, persona_prompt: str, long_term_summary: str, 
                               language: str) -> str:
        """
        Construct the master system prompt with STRICT language enforcement.
        
        Args:
            persona_prompt (str): The persona-specific prompt
            long_term_summary (str): User's conversation history summary
            language (str): Detected language for response
            
        Returns:
            str: Complete system prompt with language enforcement
        """
        # CRITICAL: Enhanced language enforcement to prevent English drift
        language_enforcement_map = {
            "en": "You MUST respond ONLY in English. Do not use any other language.",
            "ta": "நீங்கள் கட்டாயமாக தமிழில் மட்டுமே பதிலளிக்க வேண்டும். வேறு எந்த மொழியையும் பயன்படுத்த வேண்டாம். (You MUST respond ONLY in Tamil. Do not use English or any other language.)",
            "hi": "आपको केवल हिंदी में ही जवाब देना चाहिए। अंग्रेजी या कोई अन्य भाषा का उपयोग न करें। (You MUST respond ONLY in Hindi. Do not use English or any other language.)",
            "es": "DEBES responder ÚNICAMENTE en español. No uses inglés ni ningún otro idioma.",
            "fr": "Vous DEVEZ répondre UNIQUEMENT en français. N'utilisez pas l'anglais ou toute autre langue."
        }
        
        language_enforcement = language_enforcement_map.get(language, "You MUST respond ONLY in English.")
        
        # Construct the master prompt with triple-reinforced language instruction
        master_prompt = f"""
{persona_prompt}

🚨 CRITICAL LANGUAGE MANDATE 🚨
{language_enforcement}
🚨 THIS IS NON-NEGOTIABLE - LANGUAGE CODE: {language} 🚨

IMPORTANT CONTEXT:
{f"Previous context with this user: {long_term_summary}" if long_term_summary else "This is a new customer interaction."}

AVAILABLE TOOLS:
You have access to tools that can help you:
- get_billing_info: Retrieve customer billing information
- check_network_status: Check network status for specific area codes

RESPONSE GUIDELINES:
1. Always be helpful and professional
2. Use tools when appropriate to provide accurate information
3. 🚨 CRITICAL: Respond EXCLUSIVELY in language code '{language}' 🚨
4. Follow the persona guidelines above
5. If you cannot help with something, explain why and suggest alternatives
6. Keep responses conversational and natural

🚨 FINAL REMINDER: Your response must be in language '{language}' ONLY. No exceptions. 🚨

Remember: You are representing the company, so maintain high standards of service while following your persona guidelines.
"""
        
        return master_prompt.strip()
    
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