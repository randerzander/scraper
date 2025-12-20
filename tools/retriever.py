#!/usr/bin/env python3
"""
Vector embedding and retrieval tool using LanceDB.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime
import lancedb
import yaml
import sentencepiece as spm
import numpy as np


logger = logging.getLogger(__name__)

# Load config for API settings
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "lancedb"
TABLE_NAME = "embeddings"

# Embedding model settings
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

# Tokenizer (lazy loaded)
_tokenizer = None
_embedding_model = None


def _get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight, fast model
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _embedding_model


def _get_tokenizer():
    """Get or initialize the sentencepiece tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        # Download a generic tokenizer model if not exists
        tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizer.model"
        
        # For now, use a simple character-based estimation
        # In production, you'd download a specific model like llama2 tokenizer
        # We'll create a simple wrapper that estimates tokens
        class SimpleTokenizer:
            def encode(self, text):
                # Rough estimate: ~4 chars per token
                return [0] * (len(text) // 4)
        
        _tokenizer = SimpleTokenizer()
    return _tokenizer


def chunk(text: str, tokens: int = 1024) -> list[str]:
    """
    Split text into chunks by token count using sentencepiece tokenizer.
    
    Args:
        text: Text to chunk
        tokens: Target tokens per chunk (default: 1024)
        
    Returns:
        List of text chunks
    """
    tokenizer = _get_tokenizer()
    chunks = []
    
    # Split on paragraphs first to avoid breaking mid-paragraph
    paragraphs = text.split('\n\n')
    current_chunk = ""
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))
        
        # If paragraph itself is larger than chunk size, split it
        if para_tokens > tokens:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            
            # Split large paragraph by sentences
            sentences = para.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
            for sentence in sentences:
                sentence_tokens = len(tokenizer.encode(sentence))
                if current_tokens + sentence_tokens > tokens:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_tokens = sentence_tokens
                else:
                    current_chunk += sentence + " "
                    current_tokens += sentence_tokens
        else:
            # Add paragraph to current chunk
            if current_tokens + para_tokens > tokens:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                current_tokens = para_tokens
            else:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _get_embedding(text: str) -> list[float]:
    """
    Generate embedding for text using local sentence-transformers model.
    
    Args:
        text: Text to embed
        
    Returns:
        List of embedding values
    """
    model = _get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.tolist()


def write(source: str, text: str, metadata: Optional[dict] = None, chunk_size: int = 1024) -> str:
    """
    Generate embedding for text and write to LanceDB table.
    Automatically chunks large documents and stores each chunk with sequence numbering.
    
    Args:
        source: URL or filename identifying the source
        text: Text content to embed
        metadata: Optional additional metadata to store
        chunk_size: Target tokens per chunk (default: 1024)
        
    Returns:
        Status message
    """
    try:
        # Create database directory if it doesn't exist
        DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        db = lancedb.connect(str(DB_PATH))
        
        # Chunk the text
        chunks = chunk(text, chunk_size)
        
        # Prepare all chunk data
        chunk_data = []
        for seq_num, chunk_text in enumerate(chunks):
            # Generate unique ID from source, sequence, and chunk text
            content_hash = hashlib.sha256(f"{source}:{seq_num}:{chunk_text}".encode()).hexdigest()[:16]
            
            # Generate embedding
            embedding = _get_embedding(chunk_text)
            
            # Prepare chunk metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["sequence_num"] = seq_num
            chunk_metadata["total_chunks"] = len(chunks)
            
            data = {
                "id": content_hash,
                "source": source,
                "text": chunk_text,
                "vector": embedding,
                "sequence_num": seq_num,
                "metadata": chunk_metadata,
                "created_at": datetime.utcnow().isoformat()
            }
            chunk_data.append(data)
        
        # Open or create table
        try:
            table = db.open_table(TABLE_NAME)
            # Add to existing table
            table.add(chunk_data)
        except Exception:
            # Create new table with schema
            table = db.create_table(TABLE_NAME, chunk_data)
        
        total_chars = len(text)
        tokenizer = _get_tokenizer()
        token_estimate = len(tokenizer.encode(text))
        return f"Stored {len(chunks)} chunks for {source} ({total_chars} chars, ~{token_estimate} tokens)"
        
    except Exception as e:
        logger.error(f"Error writing to vector database: {str(e)}")
        return f"Error: {str(e)}"


def search(query: str, limit: int = 5) -> str:
    """
    Search for similar content using embedding similarity.
    
    Args:
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        Formatted search results
    """
    try:
        # Connect to database
        db = lancedb.connect(str(DB_PATH))
        
        # Open table
        table = db.open_table(TABLE_NAME)
        
        # Generate query embedding
        query_embedding = _get_embedding(query)
        
        # Search
        results = table.search(query_embedding).limit(limit).to_list()
        
        if not results:
            return "No results found"
        
        # Format results
        output = f"# Search Results ({len(results)} found)\n\n"
        for i, result in enumerate(results, 1):
            output += f"## Result {i}\n"
            output += f"**Source:** {result['source']}\n"
            output += f"**Distance:** {result.get('_distance', 'N/A'):.4f}\n"
            output += f"**Text:** {result['text'][:500]}...\n\n"
            if result.get('metadata'):
                output += f"**Metadata:** {result['metadata']}\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching vector database: {str(e)}")
        return f"Error: {str(e)}"
