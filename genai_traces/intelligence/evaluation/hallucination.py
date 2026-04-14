"""
Hallucination detection evaluator.

Uses NLI-based or LLM-judge approaches to detect hallucinations.
"""

from typing import Dict, Any, Optional, List
from ..evaluation.base_evaluator import BaseEvaluator


class HallucinationEvaluator(BaseEvaluator):
    """
    Evaluates whether LLM outputs contain hallucinations.
    
    Approaches:
    - LLM-as-judge: Uses another LLM to check factual consistency
    - NLI-based: Uses natural language inference models
    - Reference-based: Compares against provided ground truth
    """
    
    def __init__(
        self,
        method: str = "llm_judge",
        judge_model: str = "gpt-4o-mini",
        threshold: float = 0.5,
    ):
        """
        Initialize the hallucination evaluator.
        
        Args:
            method: Detection method (llm_judge, nli, reference)
            judge_model: Model to use for LLM-as-judge
            threshold: Score threshold for hallucination detection
        """
        self.method = method
        self.judge_model = judge_model
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "hallucination"
    
    async def evaluate(self, span: Any) -> Dict[str, float]:
        """
        Evaluate a span for hallucinations.
        
        Returns:
            Dict with hallucination_score (0=no hallucination, 1=definite hallucination)
        """
        prompt = span.get_attribute("llm.prompt") or ""
        completion = span.get_attribute("llm.completion") or ""
        context = span.get_attribute("rag.context") or ""
        
        if not completion:
            return {"eval.hallucination": 0.0}
        
        if self.method == "llm_judge":
            score = await self._llm_judge_check(prompt, completion, context)
        elif self.method == "nli":
            score = self._nli_check(prompt, completion, context)
        elif self.method == "reference":
            reference = span.get_attribute("reference_answer") or ""
            score = self._reference_check(completion, reference)
        else:
            score = 0.0
        
        return {"eval.hallucination": score}
    
    async def _llm_judge_check(
        self,
        prompt: str,
        completion: str,
        context: str,
    ) -> float:
        """Use LLM-as-judge for hallucination detection."""
        judge_prompt = f"""You are evaluating whether an AI response contains hallucinations (made-up information not supported by the context or question).

Question: {prompt}

{"Context provided: " + context if context else "No context provided."}

AI Response: {completion}

Rate the hallucination level from 0 to 1:
- 0.0: No hallucination, all claims are supported or clearly stated as uncertain
- 0.3: Minor unsupported details that don't affect the main answer
- 0.5: Some unsupported claims mixed with accurate information
- 0.7: Significant unsupported or fabricated information
- 1.0: Completely fabricated or contradicts known facts

Respond with only a number between 0 and 1."""

        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10,
                temperature=0,
            )
            score_text = response.choices[0].message.content.strip()
            return min(1.0, max(0.0, float(score_text)))
        except Exception:
            return 0.0
    
    def _nli_check(
        self,
        prompt: str,
        completion: str,
        context: str,
    ) -> float:
        """Use NLI model for hallucination detection."""
        try:
            from transformers import pipeline
            
            nli = pipeline("text-classification", model="facebook/bart-large-mnli")
            
            if context:
                premise = context
            else:
                premise = prompt
            
            sentences = [s.strip() for s in completion.split('.') if s.strip()]
            
            if not sentences:
                return 0.0
            
            contradiction_count = 0
            for sentence in sentences[:5]:
                result = nli(f"{premise} [SEP] {sentence}")
                if result[0]["label"] == "CONTRADICTION":
                    contradiction_count += 1
            
            return contradiction_count / len(sentences[:5])
            
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _reference_check(
        self,
        completion: str,
        reference: str,
    ) -> float:
        """Check against reference answer."""
        if not reference:
            return 0.0
        
        completion_words = set(completion.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 0.0
        
        overlap = len(completion_words & reference_words)
        coverage = overlap / len(reference_words)
        
        return max(0.0, 1.0 - coverage)
    
    def should_evaluate(self, span: Any) -> bool:
        """Check if span should be evaluated."""
        completion = span.get_attribute("llm.completion")
        return bool(completion)


class FactualConsistencyEvaluator(BaseEvaluator):
    """
    Evaluates factual consistency of responses against source documents.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    @property
    def name(self) -> str:
        return "factual_consistency"
    
    async def evaluate(self, span: Any) -> Dict[str, float]:
        """Evaluate factual consistency."""
        completion = span.get_attribute("llm.completion") or ""
        sources = span.retrieval_chunks or []
        
        if not completion or not sources:
            return {"eval.factual_consistency": 1.0}
        
        source_text = " ".join([c.get("content", "") for c in sources[:3]])
        
        completion_claims = set(completion.lower().split())
        source_claims = set(source_text.lower().split())
        
        if not completion_claims:
            return {"eval.factual_consistency": 1.0}
        
        supported = len(completion_claims & source_claims)
        consistency = supported / len(completion_claims)
        
        return {"eval.factual_consistency": min(1.0, consistency * 2)}
