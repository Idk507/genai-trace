"""
End-to-end tracer for Retrieval-Augmented Generation pipelines.

Captures: query embedding, vector search, chunk scores, context assembly,
LLM generation, and answer grounding.

Usage:
    with trace_rag(name="product_qa", query=user_question) as rag:
        # Step 1: Retrieval
        chunks = vector_db.search(user_question, top_k=5)
        rag.record_retrieval(chunks)

        # Step 2: LLM generation
        response = llm.generate(build_context(chunks) + user_question)
        rag.record_generation(response, context_used=True)
"""

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Generator
import asyncio

from ...core.tracer import get_tracer
from ...core.types import SpanType
from ...core.span import Span


@dataclass
class ChunkRecord:
    """Represents a single retrieved chunk from a vector database."""
    
    chunk_id: str
    content: str
    score: float
    source_doc_id: Optional[str] = None
    source_doc_page: Optional[int] = None
    source_doc_title: Optional[str] = None
    fetch_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.chunk_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": self.score,
            "source_doc_id": self.source_doc_id,
            "source_doc_page": self.source_doc_page,
            "source_doc_title": self.source_doc_title,
            "fetch_timestamp": self.fetch_timestamp,
        }


@dataclass
class RAGTrace:
    """
    Manages tracing for a RAG pipeline execution.
    
    Provides methods to record retrieval results, generation outputs,
    and compute groundedness metrics.
    """
    
    span: Span
    query: str
    chunks: List[ChunkRecord] = field(default_factory=list)
    context_tokens: int = 0
    context_used: bool = False
    _embedding_time_ms: Optional[float] = None
    _retrieval_time_ms: Optional[float] = None
    _generation_time_ms: Optional[float] = None
    
    def record_embedding(
        self,
        embedding: Optional[List[float]] = None,
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        time_ms: Optional[float] = None,
    ) -> "RAGTrace":
        """
        Record query embedding step.
        
        Args:
            embedding: The embedding vector (optional, not stored for privacy)
            model: Embedding model used
            dimensions: Embedding dimensions
            time_ms: Time taken for embedding
        """
        self.span.set_attribute("rag.embedding.model", model)
        if dimensions:
            self.span.set_attribute("rag.embedding.dimensions", dimensions)
        elif embedding:
            self.span.set_attribute("rag.embedding.dimensions", len(embedding))
        if time_ms:
            self._embedding_time_ms = time_ms
            self.span.set_attribute("rag.embedding.time_ms", time_ms)
        
        self.span.add_event("embedding_complete", {
            "model": model,
            "dimensions": dimensions or (len(embedding) if embedding else None),
        })
        
        return self
    
    def record_retrieval(
        self,
        chunks: List[Dict[str, Any]],
        source_key: str = "content",
        score_key: str = "score",
        id_key: str = "id",
        time_ms: Optional[float] = None,
        vector_db: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> "RAGTrace":
        """
        Record retrieved chunks from vector database.
        
        Args:
            chunks: List of dicts from vector DB response
            source_key: Key for chunk content in the dict
            score_key: Key for similarity score in the dict
            id_key: Key for chunk ID in the dict
            time_ms: Time taken for retrieval
            vector_db: Name of vector database used
            top_k: Number of chunks requested
        """
        self.chunks = []
        
        for i, chunk in enumerate(chunks):
            cr = ChunkRecord(
                chunk_id=str(chunk.get(id_key, str(i))),
                content=chunk.get(source_key, ""),
                score=float(chunk.get(score_key, 0.0)),
                source_doc_id=chunk.get("doc_id"),
                source_doc_page=chunk.get("page"),
                source_doc_title=chunk.get("title"),
                fetch_timestamp=datetime.utcnow().isoformat(),
                metadata={k: v for k, v in chunk.items() 
                         if k not in [source_key, score_key, id_key, "doc_id", "page", "title"]},
            )
            self.chunks.append(cr)
        
        scores = [c.score for c in self.chunks]
        
        self.span.set_attribute("rag.chunk_count", len(self.chunks))
        self.span.set_attribute("rag.top_score", max(scores) if scores else 0.0)
        self.span.set_attribute("rag.avg_score", sum(scores) / len(scores) if scores else 0.0)
        self.span.set_attribute("rag.min_score", min(scores) if scores else 0.0)
        
        source_docs = [c.source_doc_id for c in self.chunks if c.source_doc_id]
        if source_docs:
            self.span.set_attribute("rag.source_docs", list(set(source_docs)))
            self.span.set_attribute("rag.unique_sources", len(set(source_docs)))
        
        self.span.retrieval_chunks = [
            {"id": c.chunk_id, "score": c.score, "source": c.source_doc_id}
            for c in self.chunks
        ]
        
        if time_ms:
            self._retrieval_time_ms = time_ms
            self.span.set_attribute("rag.retrieval.time_ms", time_ms)
        
        if vector_db:
            self.span.set_attribute("rag.vector_db", vector_db)
        
        if top_k:
            self.span.set_attribute("rag.top_k", top_k)
        
        self.span.add_event("retrieval_complete", {
            "chunk_count": len(self.chunks),
            "top_score": max(scores) if scores else 0.0,
        })
        
        return self
    
    def record_reranking(
        self,
        reranked_chunks: List[Dict[str, Any]],
        reranker_model: str = "cross-encoder",
        original_order: Optional[List[str]] = None,
        time_ms: Optional[float] = None,
    ) -> "RAGTrace":
        """
        Record reranking step (optional).
        
        Args:
            reranked_chunks: Chunks after reranking with new scores
            reranker_model: Model used for reranking
            original_order: Original chunk IDs before reranking
            time_ms: Time taken for reranking
        """
        if original_order is None:
            original_order = [c.chunk_id for c in self.chunks]
        
        self.record_retrieval(reranked_chunks)
        
        self.span.set_attribute("rag.reranker.model", reranker_model)
        self.span.set_attribute("rag.reranker.enabled", True)
        
        if time_ms:
            self.span.set_attribute("rag.reranker.time_ms", time_ms)
        
        new_order = [c.chunk_id for c in self.chunks]
        if original_order != new_order:
            self.span.set_attribute("rag.reranker.order_changed", True)
        
        self.span.add_event("reranking_complete", {
            "model": reranker_model,
            "order_changed": original_order != new_order,
        })
        
        return self
    
    def record_context_assembly(
        self,
        context: str,
        token_count: Optional[int] = None,
        max_tokens: Optional[int] = None,
        truncated: bool = False,
    ) -> "RAGTrace":
        """
        Record context assembly step.
        
        Args:
            context: The assembled context string
            token_count: Number of tokens in context
            max_tokens: Maximum tokens allowed
            truncated: Whether context was truncated
        """
        self.context_tokens = token_count or len(context.split())
        
        self.span.set_attribute("rag.context.char_count", len(context))
        self.span.set_attribute("rag.context.token_count", self.context_tokens)
        self.span.set_attribute("rag.context.truncated", truncated)
        
        if max_tokens:
            self.span.set_attribute("rag.context.max_tokens", max_tokens)
            self.span.set_attribute("rag.context.utilization", 
                                   self.context_tokens / max_tokens if max_tokens > 0 else 0)
        
        self.span.add_event("context_assembled", {
            "token_count": self.context_tokens,
            "truncated": truncated,
        })
        
        return self
    
    def record_generation(
        self,
        response: Any,
        context_used: bool = True,
        time_ms: Optional[float] = None,
    ) -> "RAGTrace":
        """
        Record LLM generation step and compute groundedness.
        
        Args:
            response: The LLM response (string or API response object)
            context_used: Whether retrieved context was used
            time_ms: Time taken for generation
        """
        self.context_used = context_used
        self.span.set_attribute("rag.context_used", context_used)
        
        self.span.record_response(response)
        
        if time_ms:
            self._generation_time_ms = time_ms
            self.span.set_attribute("rag.generation.time_ms", time_ms)
        
        if self.chunks and context_used:
            completion = self._extract_completion(response)
            groundedness = self._compute_groundedness(completion)
            
            self.span.set_attribute("rag.grounded", groundedness > 0.1)
            self.span.set_attribute("eval.groundedness", min(groundedness * 2, 1.0))
            self.span.set_attribute("rag.groundedness_score", groundedness)
        
        total_time = 0
        if self._embedding_time_ms:
            total_time += self._embedding_time_ms
        if self._retrieval_time_ms:
            total_time += self._retrieval_time_ms
        if self._generation_time_ms:
            total_time += self._generation_time_ms
        
        if total_time > 0:
            self.span.set_attribute("rag.total_pipeline_time_ms", total_time)
        
        self.span.add_event("generation_complete", {
            "context_used": context_used,
            "grounded": self.span.get_attribute("rag.grounded"),
        })
        
        return self
    
    def _extract_completion(self, response: Any) -> str:
        """Extract completion text from various response formats."""
        if isinstance(response, str):
            return response
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content or ""
            if hasattr(choice, 'text'):
                return choice.text or ""
        
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list):
                return " ".join(
                    block.text for block in response.content 
                    if hasattr(block, 'text')
                )
            return str(response.content)
        
        completion = self.span.get_attribute("llm.completion")
        if completion:
            return str(completion)
        
        return str(response)
    
    def _compute_groundedness(self, completion: str) -> float:
        """
        Compute groundedness score using word overlap heuristic.
        
        This is a simple heuristic that checks what fraction of words
        from the top chunks appear in the response. For production use,
        consider using an LLM-based groundedness evaluator.
        """
        if not self.chunks or not completion:
            return 0.0
        
        top_chunks = sorted(self.chunks, key=lambda c: c.score, reverse=True)[:3]
        
        chunk_words = set()
        for chunk in top_chunks:
            words = set(chunk.content.lower().split())
            words = {w for w in words if len(w) > 3}
            chunk_words.update(words)
        
        response_words = set(completion.lower().split())
        response_words = {w for w in response_words if len(w) > 3}
        
        if not chunk_words:
            return 0.0
        
        overlap = len(chunk_words & response_words)
        score = overlap / len(chunk_words)
        
        return min(score, 1.0)
    
    def record_citation(
        self,
        citations: List[Dict[str, Any]],
    ) -> "RAGTrace":
        """
        Record citations in the response.
        
        Args:
            citations: List of citation dicts with chunk_id and position
        """
        self.span.set_attribute("rag.citation_count", len(citations))
        self.span.set_attribute("rag.citations", citations)
        
        cited_chunks = set(c.get("chunk_id") for c in citations if c.get("chunk_id"))
        retrieved_chunks = set(c.chunk_id for c in self.chunks)
        
        if retrieved_chunks:
            citation_coverage = len(cited_chunks & retrieved_chunks) / len(retrieved_chunks)
            self.span.set_attribute("rag.citation_coverage", citation_coverage)
        
        return self
    
    def record_feedback(
        self,
        helpful: Optional[bool] = None,
        accurate: Optional[bool] = None,
        score: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> "RAGTrace":
        """
        Record user feedback on the RAG response.
        
        Args:
            helpful: Whether the response was helpful
            accurate: Whether the response was accurate
            score: Numeric score (1-5)
            comment: Free-text feedback
        """
        if helpful is not None:
            self.span.set_attribute("rag.feedback.helpful", helpful)
        if accurate is not None:
            self.span.set_attribute("rag.feedback.accurate", accurate)
        if score is not None:
            self.span.set_attribute("rag.feedback.score", score)
        if comment:
            self.span.set_attribute("rag.feedback.comment", comment)
        
        self.span.add_event("feedback_recorded", {
            "helpful": helpful,
            "accurate": accurate,
            "score": score,
        })
        
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the RAG trace."""
        return {
            "query": self.query,
            "chunk_count": len(self.chunks),
            "top_score": max((c.score for c in self.chunks), default=0.0),
            "context_used": self.context_used,
            "grounded": self.span.get_attribute("rag.grounded"),
            "groundedness_score": self.span.get_attribute("rag.groundedness_score"),
            "total_time_ms": self.span.get_attribute("rag.total_pipeline_time_ms"),
        }


@contextlib.contextmanager
def trace_rag(
    name: str = "rag_pipeline",
    query: str = "",
    **attributes: Any,
) -> Generator[RAGTrace, None, None]:
    """
    Context manager for tracing RAG pipelines.
    
    Usage:
        with trace_rag(name="product_qa", query=user_question) as rag:
            chunks = vector_db.search(user_question, top_k=5)
            rag.record_retrieval(chunks)
            
            response = llm.generate(build_context(chunks) + user_question)
            rag.record_generation(response, context_used=True)
    
    Args:
        name: Name for the RAG span
        query: The user query being processed
        **attributes: Additional attributes to set on the span
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name, SpanType.RAG_PIPELINE) as span:
        span.set_attribute("rag.query", query)
        
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        rag = RAGTrace(span=span, query=query)
        
        try:
            yield rag
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("rag.error", str(e))
            raise


@contextlib.asynccontextmanager
async def trace_rag_async(
    name: str = "rag_pipeline",
    query: str = "",
    **attributes: Any,
):
    """
    Async context manager for tracing RAG pipelines.
    
    Usage:
        async with trace_rag_async(name="product_qa", query=user_question) as rag:
            chunks = await vector_db.search(user_question, top_k=5)
            rag.record_retrieval(chunks)
            
            response = await llm.generate(build_context(chunks) + user_question)
            rag.record_generation(response, context_used=True)
    """
    tracer = get_tracer()
    
    async with tracer.start_as_current_span_async(name, SpanType.RAG_PIPELINE) as span:
        span.set_attribute("rag.query", query)
        
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        rag = RAGTrace(span=span, query=query)
        
        try:
            yield rag
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("rag.error", str(e))
            raise
