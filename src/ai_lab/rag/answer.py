"""
RAG Answer Generation Module for AI Solutions Lab.

Provides functionality for:
- Combining retrieved chunks with LLM responses
- Generating context-aware answers
- Managing answer quality and citations
- Supporting different answer generation strategies
"""

import time
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..llm.router import LLMRouter
from .retrieve import VectorRetriever


class RAGAnswerer:
    """RAG answer generator that combines retrieval with LLM generation."""
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize RAG answerer."""
        self.settings = get_settings()
        self.retriever = VectorRetriever(index_path)
        self.llm_router = LLMRouter()
        
        # Answer generation configuration
        self.default_top_k = self.settings.top_k
        self.default_score_threshold = 0.5
        self.max_context_length = 4000  # Maximum context length for LLM
    
    def is_ready(self) -> bool:
        """Check if the answerer is ready to generate answers."""
        return self.retriever.is_loaded()
    
    async def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        include_sources: bool = True,
        answer_strategy: str = "contextual"
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            include_sources: Whether to include source information
            answer_strategy: Strategy for answer generation
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_ready():
            raise RuntimeError("RAG answerer not ready. Index may not be loaded.")
        
        start_time = time.time()
        
        try:
            # Set defaults
            top_k = top_k or self.default_top_k
            score_threshold = score_threshold or self.default_score_threshold
            
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            chunks = await self.retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                include_metadata=True
            )
            retrieval_time = time.time() - retrieval_start
            
            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "query": query,
                    "retrieval_time": retrieval_time,
                    "generation_time": 0,
                    "total_time": time.time() - start_time,
                    "chunks_retrieved": 0,
                    "answer_strategy": answer_strategy
                }
            
            # Step 2: Prepare context for LLM
            context = self._prepare_context(chunks, query)
            
            # Step 3: Generate answer using LLM
            generation_start = time.time()
            answer = await self._generate_answer(query, context, answer_strategy)
            generation_time = time.time() - generation_start
            
            # Step 4: Prepare sources information
            sources = []
            if include_sources:
                sources = self._prepare_sources(chunks)
            
            total_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "chunks_retrieved": len(chunks),
                "answer_strategy": answer_strategy,
                "context_length": len(context)
            }
            
        except Exception as e:
            raise RuntimeError(f"RAG answer generation error: {str(e)}")
    
    def _prepare_context(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """Prepare context from retrieved chunks for LLM."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Add chunk content with source information
            chunk_text = f"Source {i+1} ({chunk['title']}):\n{chunk['content']}\n"
            context_parts.append(chunk_text)
        
        # Combine all chunks
        full_context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > self.max_context_length:
            # Try to keep most relevant chunks
            truncated_parts = []
            current_length = 0
            
            for part in context_parts:
                if current_length + len(part) <= self.max_context_length:
                    truncated_parts.append(part)
                    current_length += len(part)
                else:
                    break
            
            full_context = "\n".join(truncated_parts)
            full_context += f"\n\n[Context truncated due to length. Showing {len(truncated_parts)} of {len(chunks)} sources.]"
        
        return full_context
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        strategy: str
    ) -> str:
        """Generate answer using LLM with context."""
        
        if strategy == "contextual":
            return await self._generate_contextual_answer(query, context)
        elif strategy == "summarize":
            return await self._generate_summary_answer(query, context)
        elif strategy == "qa":
            return await self._generate_qa_answer(query, context)
        else:
            # Default to contextual
            return await self._generate_contextual_answer(query, context)
    
    async def _generate_contextual_answer(self, query: str, context: str) -> str:
        """Generate contextual answer using retrieved information."""
        system_prompt = f"""You are a helpful AI assistant that answers questions based on provided context.

Context information:
{context}

Instructions:
1. Answer the user's question using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so
3. Be specific and cite sources when possible (e.g., "According to Source 1...")
4. If you're unsure about something, acknowledge the uncertainty
5. Keep your answer concise but comprehensive

User question: {query}"""

        try:
            response = await self.llm_router.chat(
                message=query,
                system_prompt=system_prompt
            )
            
            return response["content"]
            
        except Exception as e:
            return f"I encountered an error while generating the answer: {str(e)}"
    
    async def _generate_summary_answer(self, query: str, context: str) -> str:
        """Generate summary-based answer."""
        system_prompt = f"""You are a helpful AI assistant that summarizes information to answer questions.

Context information:
{context}

Instructions:
1. Summarize the key points from the context that are relevant to the question
2. Organize your answer in a clear, structured way
3. Highlight the most important information
4. If the context doesn't address the question, say so
5. Keep your summary focused and relevant

User question: {query}"""

        try:
            response = await self.llm_router.chat(
                message=query,
                system_prompt=system_prompt
            )
            
            return response["content"]
            
        except Exception as e:
            return f"I encountered an error while generating the summary: {str(e)}"
    
    async def _generate_qa_answer(self, query: str, context: str) -> str:
        """Generate Q&A style answer."""
        system_prompt = f"""You are a helpful AI assistant that answers questions in a Q&A format.

Context information:
{context}

Instructions:
1. Answer the question directly and clearly
2. Use the context information to provide accurate answers
3. If the context doesn't contain the answer, say so
4. Be concise but thorough
5. Use a conversational tone

User question: {query}"""

        try:
            response = await self.llm_router.chat(
                message=query,
                system_prompt=system_prompt
            )
            
            return response["content"]
            
        except Exception as e:
            return f"I encountered an error while generating the answer: {str(e)}"
    
    def _prepare_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for the answer."""
        sources = []
        
        for chunk in chunks:
            source = {
                "title": chunk["title"],
                "source_path": chunk["source_path"],
                "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "score": chunk["score"],
                "rank": chunk["rank"],
                "chunk_index": chunk["chunk_index"]
            }
            sources.append(source)
        
        # Sort by rank
        sources.sort(key=lambda x: x["rank"])
        
        return sources
    
    async def answer_with_followup(
        self,
        query: str,
        followup_questions: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with follow-up questions.
        
        Args:
            query: Main question
            followup_questions: Optional list of follow-up questions
            **kwargs: Additional arguments for answer generation
            
        Returns:
            Dictionary with answer, sources, and follow-up questions
        """
        # Generate main answer
        result = await self.answer(query, **kwargs)
        
        # Generate follow-up questions if requested
        if followup_questions:
            followup_answers = []
            for followup in followup_questions:
                try:
                    followup_result = await self.answer(followup, **kwargs)
                    followup_answers.append({
                        "question": followup,
                        "answer": followup_result["answer"],
                        "sources": followup_result["sources"]
                    })
                except Exception as e:
                    followup_answers.append({
                        "question": followup,
                        "answer": f"Error generating answer: {str(e)}",
                        "sources": []
                    })
            
            result["followup_answers"] = followup_answers
        
        return result
    
    async def batch_answer(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of questions
            **kwargs: Additional arguments for answer generation
            
        Returns:
            List of answer results
        """
        results = []
        
        for query in queries:
            try:
                result = await self.answer(query, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    "answer": f"Error generating answer: {str(e)}",
                    "sources": [],
                    "query": query,
                    "error": str(e)
                })
        
        return results
    
    def get_index_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded index."""
        return self.retriever.get_index_info()
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the index."""
        return self.retriever.list_documents()
    
    async def search_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search only without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant chunks
        """
        return await self.retriever.retrieve(
            query=query,
            top_k=top_k or self.default_top_k,
            score_threshold=score_threshold or self.default_score_threshold
        )
