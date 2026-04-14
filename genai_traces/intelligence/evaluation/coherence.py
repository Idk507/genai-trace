"""
Coherence evaluation for LLM outputs.

Measures logical flow, consistency, and discourse coherence.
"""

from typing import Dict, Any, List
from ..evaluation.base_evaluator import BaseEvaluator


class CoherenceEvaluator(BaseEvaluator):
    """
    Evaluates the coherence and logical flow of LLM responses.
    
    Measures:
    - Sentence connectivity
    - Topic consistency
    - Logical structure
    """
    
    def __init__(
        self,
        method: str = "heuristic",
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the coherence evaluator.
        
        Args:
            method: Evaluation method (heuristic, llm_judge, perplexity)
            model: Model for LLM-based evaluation
        """
        self.method = method
        self.model = model
    
    @property
    def name(self) -> str:
        return "coherence"
    
    async def evaluate(self, span: Any) -> Dict[str, float]:
        """
        Evaluate coherence of the response.
        
        Returns:
            Dict with coherence_score (0=incoherent, 1=highly coherent)
        """
        completion = span.get_attribute("llm.completion") or ""
        
        if not completion or len(completion) < 20:
            return {"eval.coherence": 1.0}
        
        if self.method == "heuristic":
            score = self._heuristic_check(completion)
        elif self.method == "llm_judge":
            score = await self._llm_judge_check(completion)
        else:
            score = self._heuristic_check(completion)
        
        return {"eval.coherence": score}
    
    def _heuristic_check(self, text: str) -> float:
        """Heuristic-based coherence evaluation."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0
        
        scores = []
        
        transition_words = {
            "however", "therefore", "furthermore", "additionally",
            "moreover", "consequently", "thus", "hence", "also",
            "first", "second", "third", "finally", "next", "then",
            "in addition", "as a result", "for example", "in contrast",
        }
        
        text_lower = text.lower()
        transition_count = sum(1 for w in transition_words if w in text_lower)
        transition_score = min(1.0, transition_count / max(1, len(sentences) - 1) * 2)
        scores.append(transition_score)
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_sentence_length <= 30:
            length_score = 1.0
        elif avg_sentence_length < 5:
            length_score = avg_sentence_length / 5
        else:
            length_score = max(0.5, 1.0 - (avg_sentence_length - 30) / 50)
        scores.append(length_score)
        
        word_sets = [set(s.lower().split()) for s in sentences]
        overlaps = []
        for i in range(len(word_sets) - 1):
            common = len(word_sets[i] & word_sets[i+1])
            total = len(word_sets[i] | word_sets[i+1])
            if total > 0:
                overlaps.append(common / total)
        
        if overlaps:
            overlap_score = sum(overlaps) / len(overlaps) * 2
            scores.append(min(1.0, overlap_score))
        
        return sum(scores) / len(scores) if scores else 0.5
    
    async def _llm_judge_check(self, text: str) -> float:
        """Use LLM to evaluate coherence."""
        judge_prompt = f"""Rate the coherence of the following text on a scale of 0 to 1.

Consider:
- Logical flow between sentences
- Consistent topic and theme
- Clear structure and organization
- Smooth transitions

Text: {text[:2000]}

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
            return self._heuristic_check(text)
    
    def should_evaluate(self, span: Any) -> bool:
        """Check if span should be evaluated."""
        completion = span.get_attribute("llm.completion")
        return bool(completion) and len(completion) > 50


class ReadabilityEvaluator(BaseEvaluator):
    """
    Evaluates readability of LLM outputs.
    """
    
    @property
    def name(self) -> str:
        return "readability"
    
    async def evaluate(self, span: Any) -> Dict[str, float]:
        """Evaluate readability using Flesch-Kincaid."""
        completion = span.get_attribute("llm.completion") or ""
        
        if not completion:
            return {"eval.readability": 1.0}
        
        score = self._flesch_kincaid_grade(completion)
        
        normalized = max(0.0, min(1.0, 1.0 - (score - 6) / 12))
        
        return {"eval.readability": normalized}
    
    def _flesch_kincaid_grade(self, text: str) -> float:
        """Calculate Flesch-Kincaid grade level."""
        sentences = [s for s in text.split('.') if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        asl = len(words) / len(sentences)
        asw = syllable_count / len(words) if words else 0
        
        grade = 0.39 * asl + 11.8 * asw - 15.59
        
        return max(0, grade)
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 0
        
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
