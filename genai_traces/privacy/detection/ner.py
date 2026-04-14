"""
NER-based PII detection for GenAI-Traces.

Uses spaCy or transformers for named entity recognition.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass


@dataclass
class NEREntity:
    """A named entity detected in text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


PII_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
    "CARDINAL",
    "MONEY",
    "NORP",
}


class NERDetector:
    """
    NER-based PII detector using spaCy.
    
    Usage:
        detector = NERDetector()
        entities = detector.detect("John Smith works at Acme Corp")
        # [NEREntity(text="John Smith", label="PERSON", ...)]
    """
    
    def __init__(
        self,
        model: str = "en_core_web_sm",
        pii_types: Optional[Set[str]] = None,
    ):
        self._model_name = model
        self._pii_types = pii_types or PII_ENTITY_TYPES
        self._nlp = None
    
    def _load_model(self):
        """Load the spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self._model_name)
            except ImportError:
                raise ImportError("spaCy is required for NER detection")
            except OSError:
                raise OSError(
                    f"spaCy model '{self._model_name}' not found. "
                    f"Install with: python -m spacy download {self._model_name}"
                )
        return self._nlp
    
    def detect(self, text: str) -> List[NEREntity]:
        """
        Detect named entities in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        nlp = self._load_model()
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            if ent.label_ in self._pii_types:
                entities.append(NEREntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                ))
        
        return entities
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII and group by type.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping entity type to list of values
        """
        entities = self.detect(text)
        
        pii_by_type: Dict[str, List[str]] = {}
        for entity in entities:
            if entity.label not in pii_by_type:
                pii_by_type[entity.label] = []
            pii_by_type[entity.label].append(entity.text)
        
        return pii_by_type
    
    def has_pii(self, text: str) -> bool:
        """Check if text contains any PII."""
        entities = self.detect(text)
        return len(entities) > 0


class TransformerNERDetector:
    """
    NER detector using HuggingFace transformers.
    
    Usage:
        detector = TransformerNERDetector()
        entities = detector.detect("John Smith lives in New York")
    """
    
    def __init__(
        self,
        model: str = "dslim/bert-base-NER",
        pii_types: Optional[Set[str]] = None,
    ):
        self._model_name = model
        self._pii_types = pii_types or {"PER", "ORG", "LOC", "MISC"}
        self._pipeline = None
    
    def _load_pipeline(self):
        """Load the transformers pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "ner",
                    model=self._model_name,
                    aggregation_strategy="simple",
                )
            except ImportError:
                raise ImportError("transformers is required for transformer NER")
        return self._pipeline
    
    def detect(self, text: str) -> List[NEREntity]:
        """
        Detect named entities in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        pipe = self._load_pipeline()
        results = pipe(text)
        
        entities = []
        for result in results:
            label = result.get("entity_group", result.get("entity", ""))
            
            label_clean = label.replace("B-", "").replace("I-", "")
            
            if label_clean in self._pii_types:
                entities.append(NEREntity(
                    text=result["word"],
                    label=label_clean,
                    start=result["start"],
                    end=result["end"],
                    confidence=result.get("score", 1.0),
                ))
        
        return entities
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII and group by type."""
        entities = self.detect(text)
        
        pii_by_type: Dict[str, List[str]] = {}
        for entity in entities:
            if entity.label not in pii_by_type:
                pii_by_type[entity.label] = []
            pii_by_type[entity.label].append(entity.text)
        
        return pii_by_type


def create_ner_detector(
    backend: str = "spacy",
    model: Optional[str] = None,
) -> Any:
    """
    Create a NER detector with the specified backend.
    
    Args:
        backend: "spacy" or "transformers"
        model: Optional model name
        
    Returns:
        NER detector instance
    """
    if backend == "spacy":
        return NERDetector(model=model or "en_core_web_sm")
    elif backend == "transformers":
        return TransformerNERDetector(model=model or "dslim/bert-base-NER")
    else:
        raise ValueError(f"Unknown backend: {backend}")
