"""
Groundedness evaluation for RAG responses.

Measures how well responses are grounded in retrieved context.
"""

from typing import Dict, Any, List, Set
from ..evaluation.base_evaluator import BaseEvaluator


class GroundednessEvaluator(BaseEvaluator):
    """
    Evaluates whether RAG responses are grounded in retrieved context.
    
    Methods:
    - overlap: Word overlap heuristic
    - nli: Natural language inference
    - llm_judge: LLM-based evaluation
    """
    
    def __init__(
        self,
        method: str = "overlap",
        model: str = "gpt-4o-mini",
        min_overlap: float = 0.1,
    ):
        """
        Initialize the groundedness evaluator.
        
        Args:
            method: Evaluation method (overlap, nli, llm_judge)
            model: Model for LLM-based evaluation
            min_overlap: Minimum overlap threshold
        """
        self.method = method
        self.model = model
        self.min_overlap = min_overlap
    
    @property
    def name(self) -> str:
        return "groundedness"
    
    async def evaluate(self, span: Any) -> Dict[str, float]:
        """
        Evaluate groundedness of the response.
        
        Returns:
            Dict with groundedness_score (0=ungrounded, 1=fully grounded)
        """
        completion = span.get_attribute("llm.completion") or ""
        
        chunks = span.retrieval_chunks or []
        context = span.get_attribute("rag.context") or ""
        
        if not context and chunks:
            context = " ".join([c.get("content", "") for c in chunks])
        
        if not completion:
            return {"eval.groundedness": 0.0}
        
        if not context:
            return {"eval.groundedness": 0.5}
        
        if self.method == "overlap":
            score = self._overlap_check(completion, context)
        elif self.method == "nli":
            score = self._nli_check(completion, context)
        elif self.method == "llm_judge":
            score = await self._llm_judge_check(completion, context)
        else:
            score = self._overlap_check(completion, context)
        
        return {"eval.groundedness": score}
    
    def _overlap_check(self, completion: str, context: str) -> float:
        """Word overlap-based groundedness check."""
        def get_significant_words(text: str) -> Set[str]:
            words = text.lower().split()
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                        "being", "have", "has", "had", "do", "does", "did", "will",
                        "would", "could", "should", "may", "might", "must", "shall",
                        "can", "to", "of", "in", "for", "on", "with", "at", "by",
                        "from", "as", "into", "through", "during", "before", "after",
                        "above", "below", "between", "under", "again", "further",
                        "then", "once", "here", "there", "when", "where", "why",
                        "how", "all", "each", "few", "more", "most", "other", "some",
                        "such", "no", "nor", "not", "only", "own", "same", "so",
                        "than", "too", "very", "just", "and", "but", "if", "or",
                        "because", "until", "while", "this", "that", "these", "those",
                        "it", "its", "i", "you", "he", "she", "we", "they"}
            return {w for w in words if len(w) > 2 and w not in stopwords}
        
        completion_words = get_significant_words(completion)
        context_words = get_significant_words(context)
        
        if not completion_words:
            return 1.0
        
        overlap = len(completion_words & context_words)
        score = overlap / len(completion_words)
        
        return min(1.0, score * 1.5)
    
    def _nli_check(self, completion: str, context: str) -> float:
        """NLI-based groundedness check."""
        try:
            from transformers import pipeline
            
            nli = pipeline("text-classification", model="facebook/bart-large-mnli")
            
            sentences = [s.strip() for s in completion.split('.') if s.strip()]
            
            if not sentences:
                return 0.0
            
            entailment_count = 0
            for sentence in sentences[:5]:
                result = nli(f"{context[:1000]} [SEP] {sentence}")
                if result[0]["label"] == "ENTAILMENT":
                    entailment_count += 1
            
            return entailment_count / len(sentences[:5])
            
        except ImportError:
            return self._overlap_check(completion, context)
        except Exception:
            return self._overlap_check(completion, context)
    
    async def _llm_judge_check(self, completion: str, context: str) -> float:
        """LLM-based groundedness evaluation."""
        judge_prompt = f"""Evaluate how well the response is grounded in the provided context.

Context:
{context[:2000]}

Response:
{completion[:1000]}

Rate the groundedness from 0 to 1:
- 0.0: Response contains claims not supported by the context
- 0.5: Response is partially grounded, some claims unsupported
- 1.0: Response is fully grounded in the provided context

Respond with only a number between 0 and 1."""

        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10,
                temperature=0,
            )
            score_text = response.choices[0].message.content.strip()
            return min(1.0, max(0.0, float(score_text)))
        except Exception:
            return self._overlap_check(completion, context)
    
    def should_evaluate(self, span: Any) -> bool:
        """Check if span should be evaluated (has RAG context)."""
        has_completion = bool(span.get_attribute("llm.completion"))
        has_context = bool(span.retrieval_chunks) or bool(span.get_attribute("rag.context"))
        return has_completion and has_context


class CitationAccuracyEvaluator(BaseEvaluator):
    """
    Evaluates accuracy of citations in RAG responses.
    """
    
    @property
    def name(self) -> str:
        return "citation_accuracy"
    
    async def evaluate(self, span: Any) -> Dict[str, float]:
        """Evaluate citation accuracy."""
        completion = span.get_attribute("llm.completion") or ""
        citations = span.get_attribute("rag.citations") or []
        chunks = span.retrieval_chunks or []
        
        if not citations or not chunks:
            return {"eval.citation_accuracy": 1.0}
        
        chunk_ids = {c.get("id") for c in chunks}
        cited_ids = {c.get("chunk_id") for c in citations}
        
        valid_citations = len(cited_ids & chunk_ids)
        total_citations = len(cited_ids)
        
        if total_citations == 0:
            return {"eval.citation_accuracy": 1.0}
        
        accuracy = valid_citations / total_citations
        
        return {"eval.citation_accuracy": accuracy}
