"""
Conversation analytics for GenAI-Traces.

Provides turn-level topic drift, intent tracking, and conversation analysis.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
import re


@dataclass
class TurnAnalysis:
    """Analysis of a single conversation turn."""
    turn_number: int
    role: str
    content_length: int
    word_count: int
    question_count: int
    sentiment: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class ConversationAnalytics:
    """
    Analytics for a conversation.
    
    Attributes:
        conversation_id: The conversation identifier
        total_turns: Total number of turns
        user_turns: Number of user turns
        assistant_turns: Number of assistant turns
        avg_user_message_length: Average user message length
        avg_assistant_message_length: Average assistant message length
        topics: Detected topics in the conversation
        topic_drift_score: Score indicating topic drift (0-1)
        intents: Detected user intents
        turn_analyses: Per-turn analysis
    """
    conversation_id: str
    total_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    avg_user_message_length: float = 0.0
    avg_assistant_message_length: float = 0.0
    topics: List[str] = field(default_factory=list)
    topic_drift_score: float = 0.0
    intents: List[str] = field(default_factory=list)
    turn_analyses: List[TurnAnalysis] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "user_turns": self.user_turns,
            "assistant_turns": self.assistant_turns,
            "avg_user_message_length": self.avg_user_message_length,
            "avg_assistant_message_length": self.avg_assistant_message_length,
            "topics": self.topics,
            "topic_drift_score": self.topic_drift_score,
            "intents": self.intents,
            "duration_seconds": self.duration_seconds,
        }


def analyze_conversation(
    messages: List[Dict[str, Any]],
    conversation_id: str = "unknown",
) -> ConversationAnalytics:
    """
    Analyze a conversation.
    
    Args:
        messages: List of messages with 'role' and 'content' keys
        conversation_id: Optional conversation identifier
        
    Returns:
        ConversationAnalytics with analysis results
    """
    analytics = ConversationAnalytics(conversation_id=conversation_id)
    
    user_lengths = []
    assistant_lengths = []
    all_topics = []
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        turn_analysis = _analyze_turn(i, role, content)
        analytics.turn_analyses.append(turn_analysis)
        
        if role == "user":
            analytics.user_turns += 1
            user_lengths.append(len(content))
            all_topics.extend(turn_analysis.topics)
        elif role == "assistant":
            analytics.assistant_turns += 1
            assistant_lengths.append(len(content))
    
    analytics.total_turns = len(messages)
    
    if user_lengths:
        analytics.avg_user_message_length = sum(user_lengths) / len(user_lengths)
    if assistant_lengths:
        analytics.avg_assistant_message_length = sum(assistant_lengths) / len(assistant_lengths)
    
    topic_counts = Counter(all_topics)
    analytics.topics = [topic for topic, _ in topic_counts.most_common(5)]
    
    analytics.topic_drift_score = _compute_topic_drift(analytics.turn_analyses)
    
    analytics.intents = _extract_intents(messages)
    
    return analytics


def _analyze_turn(turn_number: int, role: str, content: str) -> TurnAnalysis:
    """Analyze a single turn."""
    words = content.split()
    questions = len(re.findall(r'\?', content))
    
    topics = _extract_topics(content)
    intent = _classify_intent(content) if role == "user" else None
    sentiment = _simple_sentiment(content)
    
    return TurnAnalysis(
        turn_number=turn_number,
        role=role,
        content_length=len(content),
        word_count=len(words),
        question_count=questions,
        sentiment=sentiment,
        topics=topics,
        intent=intent,
    )


def _extract_topics(text: str) -> List[str]:
    """Extract simple topics from text using keyword matching."""
    text_lower = text.lower()
    
    topic_keywords = {
        "coding": ["code", "programming", "function", "class", "variable", "bug", "error"],
        "data": ["data", "database", "sql", "query", "table", "csv", "json"],
        "ai": ["ai", "machine learning", "model", "neural", "training", "prediction"],
        "web": ["web", "html", "css", "javascript", "api", "http", "url"],
        "help": ["help", "how to", "what is", "explain", "tutorial"],
        "debug": ["debug", "error", "fix", "issue", "problem", "wrong"],
    }
    
    found_topics = []
    for topic, keywords in topic_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found_topics.append(topic)
    
    return found_topics


def _classify_intent(text: str) -> str:
    """Classify user intent from text."""
    text_lower = text.lower()
    
    if any(q in text_lower for q in ["what is", "what are", "explain", "define"]):
        return "information_seeking"
    elif any(q in text_lower for q in ["how to", "how do", "how can"]):
        return "how_to"
    elif any(q in text_lower for q in ["fix", "error", "bug", "wrong", "not working"]):
        return "troubleshooting"
    elif any(q in text_lower for q in ["write", "create", "generate", "make"]):
        return "generation"
    elif any(q in text_lower for q in ["compare", "difference", "vs", "versus"]):
        return "comparison"
    elif "?" in text:
        return "question"
    else:
        return "statement"


def _simple_sentiment(text: str) -> str:
    """Simple sentiment analysis."""
    text_lower = text.lower()
    
    positive_words = ["thanks", "great", "good", "excellent", "helpful", "perfect", "awesome"]
    negative_words = ["bad", "wrong", "error", "problem", "issue", "fail", "broken", "frustrated"]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def _compute_topic_drift(turns: List[TurnAnalysis]) -> float:
    """Compute topic drift score (0 = no drift, 1 = complete drift)."""
    if len(turns) < 2:
        return 0.0
    
    user_turns = [t for t in turns if t.role == "user"]
    if len(user_turns) < 2:
        return 0.0
    
    first_topics = set(user_turns[0].topics) if user_turns[0].topics else set()
    last_topics = set(user_turns[-1].topics) if user_turns[-1].topics else set()
    
    if not first_topics and not last_topics:
        return 0.0
    
    if not first_topics or not last_topics:
        return 0.5
    
    intersection = first_topics & last_topics
    union = first_topics | last_topics
    
    if not union:
        return 0.0
    
    similarity = len(intersection) / len(union)
    return 1.0 - similarity


def _extract_intents(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract unique intents from user messages."""
    intents = []
    for msg in messages:
        if msg.get("role") == "user":
            intent = _classify_intent(msg.get("content", ""))
            if intent not in intents:
                intents.append(intent)
    return intents
