"""
Vector Store Module for Customer Service Knowledge Base

This module handles CSV-based data ingestion, creates FAISS vector embeddings,
and provides semantic search capabilities for customer service interactions.
"""

import os
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define BASE_DIR for cross-platform file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to project-kural root


class CustomerServiceVectorStore:
    """
    Vector store implementation for customer service knowledge base using CSV data.
    """
    
    def __init__(self, embeddings_model=None, csv_path: str = None):
        """
        Initialize the vector store with embeddings model and CSV data path.
        
        Args:
            embeddings_model: Embeddings model for vector generation
            csv_path: Path to the CSV file containing training data
        """
        self.embeddings_model = embeddings_model or self._get_default_embeddings()
        self.csv_path = csv_path or os.path.join(BASE_DIR, "training_data", "Intelligent Virtual Assistants for Customer Support (1).csv")
        self.vector_store = None
        self.documents = []
        
        logger.info(f"CustomerServiceVectorStore initialized with CSV: {self.csv_path}")
    
    def _get_default_embeddings(self):
        """
        Get default embeddings model with local-first approach and robust fallback handling.
        
        Returns:
            Embeddings model instance
        """
        try:
            # Try to use OpenAI embeddings if API key is available
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbeddings(openai_api_key=openai_api_key)
            else:
                logger.info("OpenAI API key not found, attempting local HuggingFace embeddings")
                
        except ImportError as e:
            logger.warning(f"OpenAI embeddings not available: {e}")
        
        # Use local HuggingFace embeddings with local-first approach
        try:
            logger.info("Initializing HuggingFace embeddings from local model...")
            
            # Define local model path - dynamically construct absolute path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_model_path = os.path.join(project_root, "all-MiniLM-L6-v2")
            
            # Verify local model exists
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(
                    f"Local embeddings model not found at: {local_model_path}\n\n"
                    "🔧 **REQUIRED SETUP**: You must download the model manually before running the application.\n\n"
                    "Run this command from the project root directory:\n"
                    "git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2\n\n"
                    "This is a one-time download of approximately 90-100MB.\n"
                    "After downloading, restart the application.\n\n"
                    "Note: Ensure you have Git LFS installed:\n"
                    "- Linux: sudo apt-get install git-lfs\n"
                    "- Mac: brew install git-lfs\n"
                    "- Run: git lfs install (after installation)"
                )
            
            # 🔍 FORENSIC CHECK: Pre-flight validation for Git LFS pointer files
            safetensors_path = os.path.join(local_model_path, "model.safetensors")
            pytorch_model_path = os.path.join(local_model_path, "pytorch_model.bin")
            
            # Check if critical model files exist and are properly downloaded
            if os.path.exists(safetensors_path):
                file_size = os.path.getsize(safetensors_path)
                if file_size < 1024 * 1024:  # Less than 1MB indicates pointer file
                    # Read first few bytes to confirm it's a pointer file
                    with open(safetensors_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                    
                    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
                        raise FileNotFoundError(
                            f"🚨 **GIT LFS POINTER FILE DETECTED**\n\n"
                            f"The file {safetensors_path} is a Git LFS pointer file (size: {file_size} bytes), "
                            f"not the actual model file (~91MB).\n\n"
                            f"This indicates that Git LFS was not properly configured when you downloaded the model.\n\n"
                            f"**SOLUTION**:\n"
                            f"1. Install Git LFS: sudo apt-get install git-lfs\n"
                            f"2. Initialize Git LFS: git lfs install\n"
                            f"3. Delete the corrupted directory: rm -rf {local_model_path}\n"
                            f"4. Re-download: git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2\n"
                            f"5. Verify success: ls -lh {local_model_path}/model.safetensors (should show ~91M)\n\n"
                            f"**Current file content preview**: {first_line[:100]}..."
                        )
                    else:
                        raise FileNotFoundError(
                            f"🚨 **CORRUPTED MODEL FILE DETECTED**\n\n"
                            f"The file {safetensors_path} is unexpectedly small (size: {file_size} bytes) "
                            f"and appears to be corrupted or incomplete.\n\n"
                            f"**SOLUTION**:\n"
                            f"1. Delete the corrupted directory: rm -rf {local_model_path}\n"
                            f"2. Ensure Git LFS is installed: git lfs install\n"
                            f"3. Re-download: git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2\n"
                            f"4. Verify success: ls -lh {local_model_path}/model.safetensors (should show ~91M)"
                        )
            
            # Additional check for pytorch_model.bin
            if os.path.exists(pytorch_model_path):
                file_size = os.path.getsize(pytorch_model_path)
                if file_size < 1024 * 1024:  # Less than 1MB indicates pointer file
                    logger.warning(f"pytorch_model.bin is suspiciously small: {file_size} bytes")
            
            # Load model from local path (offline-capable) using the new langchain-huggingface package
            embeddings = HuggingFaceEmbeddings(
                model_name=local_model_path,  # Use local path instead of Hub ID
                model_kwargs={'device': 'cpu'},  # Ensure CPU usage for compatibility
                encode_kwargs={'normalize_embeddings': True}  # Improve retrieval quality
            )
            
            logger.info(f"HuggingFace embeddings loaded successfully from local path: {local_model_path}")
            return embeddings
            
        except ImportError as e:
            logger.error(f"HuggingFace embeddings not available: {e}")
            raise RuntimeError(
                "No embeddings libraries available. Please install one of:\n"
                "- OpenAI: pip install openai\n"
                "- HuggingFace: pip install langchain-huggingface"
            )
        except FileNotFoundError as e:
            # Re-raise the detailed FileNotFoundError with setup instructions
            raise e
        except OSError as e:
            logger.error(f"OS error accessing local model: {e}")
            raise RuntimeError(
                f"Cannot access local embeddings model. This could be due to:\n"
                "1. Permission issues with the model directory\n"
                "2. Corrupted model files\n"
                "3. Insufficient disk space\n"
                f"Original error: {e}\n\n"
                "Solutions:\n"
                "- Check file permissions on the all-MiniLM-L6-v2 directory\n"
                "- Re-download the model: rm -rf all-MiniLM-L6-v2 && git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2\n"
                "- Ensure adequate disk space (model requires ~100MB)"
            )
        except Exception as e:
            logger.error(f"Unexpected error initializing local embeddings: {e}")
            raise RuntimeError(
                f"Failed to initialize local embeddings model: {e}\n\n"
                "This is likely a model compatibility issue. Try:\n"
                "1. Re-downloading the model\n"
                "2. Checking your langchain-huggingface version\n"
                "3. Using OpenAI embeddings instead (set OPENAI_API_KEY)"
            )
    
    def load_csv_data(self) -> List[Document]:
        """
        Load customer service data from CSV file and convert to LangChain Documents.
        
        Returns:
            List[Document]: List of Document objects with instructions as content and responses as metadata
        """
        try:
            # Load CSV data using pandas
            logger.info(f"Loading CSV data from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Validate required columns exist
            required_columns = ['instruction', 'response']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}. Available columns: {list(df.columns)}")
            
            # Check for empty DataFrame
            if df.empty:
                raise ValueError("CSV file is empty or contains no valid data")
            
            # Remove rows with missing instruction or response
            df_cleaned = df.dropna(subset=['instruction', 'response'])
            
            if df_cleaned.empty:
                raise ValueError("No valid instruction-response pairs found in CSV after cleaning")
            
            logger.info(f"Loaded {len(df_cleaned)} valid instruction-response pairs from CSV")
            
            # Convert DataFrame rows to LangChain Documents
            documents = []
            for index, row in df_cleaned.iterrows():
                try:
                    # Create Document with instruction as page_content and response as metadata
                    doc = Document(
                        page_content=str(row['instruction']).strip(),
                        metadata={
                            'response': str(row['response']).strip(),
                            'source': 'customer_service_csv',
                            'row_index': index,
                            'category': row.get('category', 'general'),  # Optional category field
                            'confidence': row.get('confidence', 1.0)     # Optional confidence field
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to process row {index}: {e}")
                    continue
            
            if not documents:
                raise ValueError("No valid documents could be created from CSV data")
            
            logger.info(f"Successfully created {len(documents)} documents from CSV")
            self.documents = documents
            return documents
            
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
            raise FileNotFoundError(f"Training data CSV file not found at: {self.csv_path}")
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            raise ValueError("CSV file is empty or corrupted")
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file: {e}")
            raise ValueError(f"Invalid CSV file format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading CSV data: {e}")
            raise RuntimeError(f"Failed to load CSV data: {e}")
    
    def create_vector_store(self, documents: List[Document] = None) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            FAISS: Initialized FAISS vector store
        """
        try:
            if documents is None:
                documents = self.documents
            
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            logger.info(f"Creating FAISS vector store with {len(documents)} documents")
            
            # Optional: Split long documents for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Split documents if they're too long
            split_docs = []
            for doc in documents:
                if len(doc.page_content) > 1000:
                    splits = text_splitter.split_documents([doc])
                    split_docs.extend(splits)
                else:
                    split_docs.append(doc)
            
            logger.info(f"Split into {len(split_docs)} chunks for indexing")
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=split_docs,
                embedding=self.embeddings_model
            )
            
            logger.info("FAISS vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise RuntimeError(f"Vector store creation failed: {e}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List[Document]: Most similar documents
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_vector_store() first.")
            
            if not query.strip():
                raise ValueError("Empty query provided for similarity search")
            
            logger.info(f"Performing similarity search for: '{query[:50]}...'")
            
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        Perform similarity search with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List[tuple]: (Document, similarity_score) tuples
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Call create_vector_store() first.")
            
            logger.info(f"Performing similarity search with scores for: '{query[:50]}...'")
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search with scores failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            logger.info(f"Saving vector store to: {path}")
            self.vector_store.save_local(path)
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise RuntimeError(f"Vector store save failed: {e}")
    
    def load_vector_store(self, path: str) -> FAISS:
        """
        Load a previously saved vector store from disk.
        
        Args:
            path: Path to load the vector store from
            
        Returns:
            FAISS: Loaded vector store
        """
        try:
            logger.info(f"Loading vector store from: {path}")
            self.vector_store = FAISS.load_local(path, self.embeddings_model)
            logger.info("Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise RuntimeError(f"Vector store load failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about the vector store
        """
        return {
            "total_documents": len(self.documents),
            "csv_path": self.csv_path,
            "vector_store_initialized": self.vector_store is not None,
            "embeddings_model": str(type(self.embeddings_model).__name__),
            "index_size": self.vector_store.index.ntotal if self.vector_store else 0
        }


def initialize_knowledge_base(csv_path: str = None) -> CustomerServiceVectorStore:
    """
    Initialize the customer service knowledge base from CSV data.
    
    Args:
        csv_path: Optional path to CSV file
        
    Returns:
        CustomerServiceVectorStore: Initialized vector store
    """
    try:
        # Initialize vector store
        vector_store = CustomerServiceVectorStore(csv_path=csv_path)
        
        # Load CSV data
        documents = vector_store.load_csv_data()
        
        # Create FAISS index
        vector_store.create_vector_store(documents)
        
        logger.info("Knowledge base initialized successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize knowledge base: {e}")