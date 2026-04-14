"""
Type definitions and attribute key constants for GenAI-Traces.
"""

from enum import Enum


class SpanType(Enum):
    """Types of spans that can be created."""
    
    # Session / request
    REQUEST = "request"
    SESSION = "session"

    # Workflow
    AGENT = "agent"
    CHAIN = "chain"
    WORKFLOW = "workflow"

    # Core LLM operations
    LLM = "llm"
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"

    # Retrieval (RAG)
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    SEARCH = "search"
    RAG_PIPELINE = "rag_pipeline"
    CHUNK_SCORE = "chunk_score"

    # Tool operations
    TOOL = "tool"
    FUNCTION_CALL = "function_call"
    API_CALL = "api_call"

    # Intelligence
    EVALUATION = "evaluation"
    FEEDBACK = "feedback"
    GUARDRAIL = "guardrail"
    ANNOTATION = "annotation"

    # Data ops
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"

    # Security
    INJECTION_CHECK = "injection_check"
    OUTPUT_FILTER = "output_filter"

    # Router
    ROUTER_DECISION = "router_decision"
    FALLBACK = "fallback"

    # Cache
    CACHE_LOOKUP = "cache_lookup"

    # Multi-modal
    VISION = "vision"
    AUDIO = "audio"


class SpanStatus(Enum):
    """Status of a span."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"  # Guardrail blocked


class InjectionType(Enum):
    """Types of prompt injection attacks."""
    
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    DAN = "dan"
    GOAL_HIJACKING = "goal_hijacking"
    DATA_EXFILTRATION = "data_exfiltration"
    NONE = "none"


# --- LLM attributes ---
LLM_PROVIDER = "llm.provider"
LLM_MODEL_NAME = "llm.model.name"
LLM_MODEL_VERSION = "llm.model.version"
LLM_REQUEST_TYPE = "llm.request.type"
LLM_TEMPERATURE = "llm.request.temperature"
LLM_MAX_TOKENS = "llm.request.max_tokens"
LLM_TOP_P = "llm.request.top_p"
LLM_STOP_SEQUENCES = "llm.request.stop_sequences"
LLM_SEED = "llm.request.seed"
LLM_PROMPT = "llm.prompt"
LLM_PROMPT_HASH = "llm.prompt.hash"
LLM_PROMPT_TOKENS = "llm.prompt.tokens"
LLM_MESSAGES = "llm.messages"
LLM_SYSTEM_PROMPT = "llm.system_prompt"
LLM_COMPLETION = "llm.completion"
LLM_COMPLETION_HASH = "llm.completion.hash"
LLM_COMPLETION_TOKENS = "llm.completion.tokens"
LLM_TOTAL_TOKENS = "llm.total_tokens"
LLM_DURATION_MS = "llm.duration_ms"
LLM_TTFT_MS = "llm.ttft_ms"
LLM_TOKENS_PER_SECOND = "llm.tokens_per_second"
LLM_STREAMING = "llm.streaming"
LLM_FUNCTIONS = "llm.functions"
LLM_FUNCTION_CALL = "llm.function_call"
LLM_TOOL_CALLS = "llm.tool_calls"

# --- Cost attributes ---
COST_TOTAL_USD = "cost.total_usd"
COST_PROMPT_USD = "cost.prompt_usd"
COST_COMPLETION_USD = "cost.completion_usd"
COST_CACHE_HIT = "cost.cache_hit"
COST_CACHE_SAVINGS_USD = "cost.cache_savings_usd"

# --- Error attributes ---
ERROR_TYPE = "error.type"
ERROR_MESSAGE = "error.message"
ERROR_STACK_TRACE = "error.stack_trace"
RETRY_COUNT = "retry.count"
RETRY_REASON = "retry.reason"

# --- Evaluation attributes ---
EVAL_RELEVANCE = "eval.relevance"
EVAL_HALLUCINATION = "eval.hallucination"
EVAL_TOXICITY = "eval.toxicity"
EVAL_COHERENCE = "eval.coherence"
EVAL_GROUNDEDNESS = "eval.groundedness"
EVAL_HELPFULNESS = "eval.helpfulness"
EVAL_ACCURACY = "eval.accuracy"
EVAL_OVERALL_QUALITY = "eval.quality"
EVAL_METHOD = "eval.method"
EVAL_MODEL = "eval.model"

# --- Feedback attributes ---
FEEDBACK_SCORE = "feedback.score"
FEEDBACK_RATING = "feedback.rating"
FEEDBACK_COMMENT = "feedback.comment"
FEEDBACK_SOURCE = "feedback.source"
FEEDBACK_USER_ID = "feedback.user_id"
FEEDBACK_DIMENSIONS = "feedback.dimensions"

# --- Conversation attributes ---
CONVERSATION_ID = "conversation.id"
CONVERSATION_TURN = "conversation.turn"
CONVERSATION_ROLE = "conversation.role"
CONVERSATION_TOPIC = "conversation.topic"

# --- Privacy attributes ---
PRIVACY_PII_DETECTED = "privacy.pii_detected"
PRIVACY_PII_TYPES = "privacy.pii_types"
PRIVACY_REDACTED = "privacy.redacted"
PRIVACY_ENCRYPTED = "privacy.encrypted"

# --- Agent attributes ---
AGENT_NAME = "agent.name"
AGENT_TYPE = "agent.type"
AGENT_GOAL = "agent.goal"
AGENT_REASONING = "agent.reasoning"
AGENT_DECISION = "agent.decision"
AGENT_TOOL_SELECTED = "agent.tool_selected"
AGENT_ITERATIONS = "agent.iterations"

# --- Security attributes ---
SECURITY_INJECTION_DETECTED = "security.injection_detected"
SECURITY_INJECTION_TYPE = "security.injection_type"
SECURITY_INJECTION_SCORE = "security.injection_score"
SECURITY_GUARDRAIL_TRIGGERED = "security.guardrail_triggered"
SECURITY_ACTION_TAKEN = "security.action_taken"

# --- Prompt management attributes ---
PROMPT_NAME = "prompt.name"
PROMPT_VERSION = "prompt.version"
PROMPT_HASH = "prompt.hash"
EXPERIMENT_ID = "experiment.id"
EXPERIMENT_VARIANT = "experiment.variant"
EXPERIMENT_TRAFFIC = "experiment.traffic_pct"

# --- RAG attributes ---
RAG_QUERY = "rag.query"
RAG_CHUNK_COUNT = "rag.chunk_count"
RAG_TOP_SCORE = "rag.top_score"
RAG_CONTEXT_USED = "rag.context_used"
RAG_GROUNDED = "rag.grounded"
RAG_SOURCE_DOCS = "rag.source_docs"

# --- Cache attributes ---
CACHE_HIT = "cache.hit"
CACHE_SIMILARITY = "cache.similarity_score"
CACHE_KEY_HASH = "cache.key_hash"
CACHE_TTL_SECONDS = "cache.ttl_seconds"
CACHE_SAVINGS_USD = "cache.savings_usd"

# --- Router attributes ---
ROUTER_PRIMARY_MODEL = "router.primary_model"
ROUTER_SELECTED_MODEL = "router.selected_model"
ROUTER_REASON = "router.reason"
ROUTER_FALLBACK_COUNT = "router.fallback_count"

# --- Multi-modal attributes ---
MODAL_INPUT_TYPE = "modal.input_type"
MODAL_IMAGE_COUNT = "modal.image_count"
MODAL_AUDIO_SECONDS = "modal.audio_seconds"
MODAL_CONTENT_HASH = "modal.content_hash"

# --- Service attributes ---
SERVICE_NAME = "service.name"
SERVICE_ENVIRONMENT = "service.environment"
SERVICE_VERSION = "service.version"
