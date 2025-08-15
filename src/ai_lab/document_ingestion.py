"""Document ingestion system for AI Solutions Lab."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import pypdf
except ImportError:
    print("Please install: pip install pypdf")
    raise

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    source_file: str
    chunk_id: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

class DocumentIngester:
    """Handles document ingestion and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_markdown(self, file_path: Path) -> str:
        """Load markdown file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading markdown file {file_path}: {e}")
            return ""
    
    def load_pdf(self, file_path: Path) -> str:
        """Load PDF file content."""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        
        # If text is shorter than chunk size, return as single chunk
        if text_length <= self.chunk_size:
            chunk = DocumentChunk(
                text=text,
                source_file=source_file,
                chunk_id=f"{source_file}_chunk_0",
                start_char=0,
                end_char=text_length,
                metadata={"file_type": Path(source_file).suffix}
            )
            chunks.append(chunk)
            return chunks
        
        # Create overlapping chunks
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start and sentence_end - start > self.chunk_size * 0.7:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = DocumentChunk(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_id=f"{source_file}_chunk_{chunk_id}",
                    start_char=start,
                    end_char=end,
                    metadata={
                        "file_type": Path(source_file).suffix,
                        "chunk_size": len(chunk_text),
                        "total_chunks": (text_length // self.chunk_size) + 1
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single file and return chunks."""
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return []
        
        file_path = file_path.resolve()
        
        # Load content based on file type
        if file_path.suffix.lower() == '.md':
            content = self.load_markdown(file_path)
        elif file_path.suffix.lower() == '.pdf':
            content = self.load_pdf(file_path)
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return []
        
        if not content:
            return []
        
        # Clean and chunk the content
        cleaned_content = self.clean_text(content)
        chunks = self.chunk_text(cleaned_content, str(file_path))
        
        print(f"Processed {file_path.name}: {len(chunks)} chunks")
        return chunks
    
    def process_directory(self, directory_path: Path, file_patterns: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Process all supported files in a directory."""
        if not directory_path.exists():
            print(f"Directory not found: {directory_path}")
            return []
        
        if file_patterns is None:
            file_patterns = ['*.md', '*.pdf']
        
        all_chunks = []
        
        for pattern in file_patterns:
            files = list(directory_path.glob(pattern))
            for file_path in files:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
        
        print(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def get_chunk_texts(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extract just the text content from chunks."""
        return [chunk.text for chunk in chunks]

def main():
    """Demo the document ingestion system."""
    ingester = DocumentIngester(chunk_size=500, chunk_overlap=100)
    
    # Process the sample document
    sample_file = Path("data/docs/sample.md")
    if sample_file.exists():
        chunks = ingester.process_file(sample_file)
        
        print(f"\nProcessed {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.source_file}")
            print(f"Length: {len(chunk.text)} characters")
            print(f"Text: {chunk.text[:100]}...")
    else:
        print("Sample document not found. Create data/docs/sample.md first.")

if __name__ == "__main__":
    main()
