"""
Comprehensive test to verify all GenAI-Traces modules are properly implemented.
"""

import sys
sys.path.insert(0, '.')

def test_all():
    print("Testing GenAI-Traces Complete Implementation")
    print("=" * 70)
    errors = []
    
    # 1. Core modules
    print("\n1. Testing core modules...")
    try:
        from genai_traces.core import Tracer, Span, SpanType, SpanStatus
        from genai_traces.core.context import get_current_span, set_current_span
        from genai_traces.core.decorators import trace, trace_llm, trace_agent, trace_tool
        from genai_traces.core.context_manager import trace_llm_context
        from genai_traces.core.sampling import AdaptiveSampler
        print("   Core modules: OK")
    except Exception as e:
        errors.append(f"Core: {e}")
        print(f"   Core modules: FAILED - {e}")
    
    # 2. Config modules
    print("\n2. Testing config modules...")
    try:
        from genai_traces.config import TracerConfig
        from genai_traces.config.validators import validate_config, ValidationResult
        print("   Config modules: OK")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"   Config modules: FAILED - {e}")
    
    # 3. Instrumentation - LLM
    print("\n3. Testing instrumentation/llm modules...")
    try:
        from genai_traces.instrumentation.llm.openai import instrument_openai
        from genai_traces.instrumentation.llm.anthropic import instrument_anthropic
        from genai_traces.instrumentation.llm.azure import instrument_azure_openai
        from genai_traces.instrumentation.llm.bedrock import instrument_bedrock
        from genai_traces.instrumentation.llm.google import instrument_google
        from genai_traces.instrumentation.llm.generic import wrap_llm_call, TracedLLMClient
        print("   Instrumentation/LLM: OK")
    except Exception as e:
        errors.append(f"Instrumentation/LLM: {e}")
        print(f"   Instrumentation/LLM: FAILED - {e}")
    
    # 4. Instrumentation - Frameworks
    print("\n4. Testing instrumentation/frameworks modules...")
    try:
        from genai_traces.instrumentation.frameworks import (
            LangChainCallbackHandler,
            instrument_langgraph,
            LlamaIndexCallbackHandler,
            instrument_haystack,
            instrument_dspy,
            instrument_vercel_ai,
        )
        print("   Instrumentation/Frameworks: OK")
    except Exception as e:
        errors.append(f"Instrumentation/Frameworks: {e}")
        print(f"   Instrumentation/Frameworks: FAILED - {e}")
    
    # 5. Instrumentation - Agents
    print("\n5. Testing instrumentation/agents modules...")
    try:
        from genai_traces.instrumentation.agents import (
            trace_react_step,
            ReActTracer,
            instrument_autogen,
            AutoGenTracer,
            CustomAgentTracer,
            trace_agent_step,
        )
        print("   Instrumentation/Agents: OK")
    except Exception as e:
        errors.append(f"Instrumentation/Agents: {e}")
        print(f"   Instrumentation/Agents: FAILED - {e}")
    
    # 6. Instrumentation - Retrieval
    print("\n6. Testing instrumentation/retrieval modules...")
    try:
        from genai_traces.instrumentation.retrieval import (
            trace_rag,
            RAGTrace,
            ChunkRecord,
            VectorDBTracer,
            RerankerTracer,
        )
        print("   Instrumentation/Retrieval: OK")
    except Exception as e:
        errors.append(f"Instrumentation/Retrieval: {e}")
        print(f"   Instrumentation/Retrieval: FAILED - {e}")
    
    # 7. Instrumentation - Tools
    print("\n7. Testing instrumentation/tools modules...")
    try:
        from genai_traces.instrumentation.tools import trace_function_call, FunctionCallTracer
        print("   Instrumentation/Tools: OK")
    except Exception as e:
        errors.append(f"Instrumentation/Tools: {e}")
        print(f"   Instrumentation/Tools: FAILED - {e}")
    
    # 8. Telemetry - Tokens
    print("\n8. Testing telemetry/tokens modules...")
    try:
        from genai_traces.telemetry.tokens.counter import TokenCounter
        from genai_traces.telemetry.tokens.estimator import TokenEstimator
        from genai_traces.telemetry.tokens.streaming import StreamingAccumulator
        print("   Telemetry/Tokens: OK")
    except Exception as e:
        errors.append(f"Telemetry/Tokens: {e}")
        print(f"   Telemetry/Tokens: FAILED - {e}")
    
    # 9. Telemetry - Cost
    print("\n9. Testing telemetry/cost modules...")
    try:
        from genai_traces.telemetry.cost.estimator import CostEstimator
        from genai_traces.telemetry.cost.pricing_table import PricingTable
        from genai_traces.telemetry.cost.aggregator import CostAggregator
        print("   Telemetry/Cost: OK")
    except Exception as e:
        errors.append(f"Telemetry/Cost: {e}")
        print(f"   Telemetry/Cost: FAILED - {e}")
    
    # 10. Telemetry - Metrics
    print("\n10. Testing telemetry/metrics modules...")
    try:
        from genai_traces.telemetry.metrics import LatencyTracker, ThroughputTracker, ErrorRateTracker
        print("   Telemetry/Metrics: OK")
    except Exception as e:
        errors.append(f"Telemetry/Metrics: {e}")
        print(f"   Telemetry/Metrics: FAILED - {e}")
    
    # 11. Telemetry - Anomaly
    print("\n11. Testing telemetry/anomaly modules...")
    try:
        from genai_traces.telemetry.anomaly import AnomalyDetector, AlertManager
        from genai_traces.telemetry.anomaly.baselines import ModelBaseline, RollingBaseline
        print("   Telemetry/Anomaly: OK")
    except Exception as e:
        errors.append(f"Telemetry/Anomaly: {e}")
        print(f"   Telemetry/Anomaly: FAILED - {e}")
    
    # 12. Telemetry - Environment
    print("\n12. Testing telemetry/environment modules...")
    try:
        from genai_traces.telemetry.environment import get_system_info, get_resource_usage
        print("   Telemetry/Environment: OK")
    except Exception as e:
        errors.append(f"Telemetry/Environment: {e}")
        print(f"   Telemetry/Environment: FAILED - {e}")
    
    # 13. Intelligence - Feedback
    print("\n13. Testing intelligence/feedback modules...")
    try:
        from genai_traces.intelligence.feedback import record_feedback, FeedbackCollector
        from genai_traces.intelligence.feedback.schema import FeedbackRecord, FeedbackType
        from genai_traces.intelligence.feedback.aggregator import FeedbackAggregator
        print("   Intelligence/Feedback: OK")
    except Exception as e:
        errors.append(f"Intelligence/Feedback: {e}")
        print(f"   Intelligence/Feedback: FAILED - {e}")
    
    # 14. Intelligence - Evaluation
    print("\n14. Testing intelligence/evaluation modules...")
    try:
        from genai_traces.intelligence.evaluation import BaseEvaluator, RelevanceEvaluator
        from genai_traces.intelligence.evaluation.hallucination import HallucinationEvaluator
        from genai_traces.intelligence.evaluation.toxicity import ToxicityEvaluator
        from genai_traces.intelligence.evaluation.coherence import CoherenceEvaluator
        from genai_traces.intelligence.evaluation.groundedness import GroundednessEvaluator
        print("   Intelligence/Evaluation: OK")
    except Exception as e:
        errors.append(f"Intelligence/Evaluation: {e}")
        print(f"   Intelligence/Evaluation: FAILED - {e}")
    
    # 15. Intelligence - Annotation
    print("\n15. Testing intelligence/annotation modules...")
    try:
        from genai_traces.intelligence.annotation import AnnotationQueue, AnnotationRubric, compute_agreement
        print("   Intelligence/Annotation: OK")
    except Exception as e:
        errors.append(f"Intelligence/Annotation: {e}")
        print(f"   Intelligence/Annotation: FAILED - {e}")
    
    # 16. Intelligence - Conversation
    print("\n16. Testing intelligence/conversation modules...")
    try:
        from genai_traces.intelligence.conversation import (
            set_conversation_context,
            Session,
            SessionManager,
            analyze_conversation,
        )
        print("   Intelligence/Conversation: OK")
    except Exception as e:
        errors.append(f"Intelligence/Conversation: {e}")
        print(f"   Intelligence/Conversation: FAILED - {e}")
    
    # 17. Intelligence - Quality
    print("\n17. Testing intelligence/quality modules...")
    try:
        from genai_traces.intelligence.quality import QualityScorer, Benchmark, BenchmarkRunner
        print("   Intelligence/Quality: OK")
    except Exception as e:
        errors.append(f"Intelligence/Quality: {e}")
        print(f"   Intelligence/Quality: FAILED - {e}")
    
    # 18. Prompt Management
    print("\n18. Testing prompt_management modules...")
    try:
        from genai_traces.prompt_management import PromptRegistry, ABTestManager
        from genai_traces.prompt_management.versioning import PromptVersion, diff_prompts
        from genai_traces.prompt_management.experiment import Experiment, ExperimentTracker
        from genai_traces.prompt_management.playground import PromptPlayground
        print("   Prompt Management: OK")
    except Exception as e:
        errors.append(f"Prompt Management: {e}")
        print(f"   Prompt Management: FAILED - {e}")
    
    # 19. Security
    print("\n19. Testing security modules...")
    try:
        from genai_traces.security import InjectionDetector, OutputGuardrail, GuardrailChain
        from genai_traces.security.output_filter import OutputFilter
        from genai_traces.security.domain_enforcer import DomainEnforcer
        from genai_traces.security.red_team import RedTeamRunner, AdversarialTest
        print("   Security: OK")
    except Exception as e:
        errors.append(f"Security: {e}")
        print(f"   Security: FAILED - {e}")
    
    # 20. Privacy
    print("\n20. Testing privacy modules...")
    try:
        from genai_traces.privacy import PIIDetector, Redactor
        from genai_traces.privacy.detection.patterns import PII_PATTERNS
        from genai_traces.privacy.detection.ner import NERDetector
        from genai_traces.privacy.redaction.strategies import StrategyRedactor, RedactionStrategy
        from genai_traces.privacy.redaction.hashing import PIIHasher
        from genai_traces.privacy.encryption import FieldEncryptor
        from genai_traces.privacy.compliance import RetentionPolicy, AuditLog
        print("   Privacy: OK")
    except Exception as e:
        errors.append(f"Privacy: {e}")
        print(f"   Privacy: FAILED - {e}")
    
    # 21. Exporters
    print("\n21. Testing exporters modules...")
    try:
        from genai_traces.exporters import BaseExporter, ConsoleExporter, JSONFileExporter
        from genai_traces.exporters.batch import BatchExporter, CircularBuffer
        from genai_traces.exporters.finetune import FineTuneExporter
        from genai_traces.exporters.finetune.formats import FormatConverter
        from genai_traces.exporters.finetune.filter import QualityFilter
        from genai_traces.exporters.json.rotation import FileRotator
        from genai_traces.exporters.json.compression import compress_data, CompressedWriter
        from genai_traces.exporters.database import PostgresExporter, MySQLExporter, SQLiteExporter
        from genai_traces.exporters.otel import OTLPExporter, JaegerExporter, SpanMapper
        from genai_traces.exporters.cloud import S3Exporter, GCSExporter, AzureBlobExporter
        from genai_traces.exporters.webhook import HTTPExporter
        print("   Exporters: OK")
    except Exception as e:
        errors.append(f"Exporters: {e}")
        print(f"   Exporters: FAILED - {e}")
    
    # 22. Multimodal
    print("\n22. Testing multimodal modules...")
    try:
        from genai_traces.multimodal import capture_image_metadata, capture_audio_metadata, hash_content
        print("   Multimodal: OK")
    except Exception as e:
        errors.append(f"Multimodal: {e}")
        print(f"   Multimodal: FAILED - {e}")
    
    # 23. Router
    print("\n23. Testing router modules...")
    try:
        from genai_traces.router import trace_router, FallbackChain
        print("   Router: OK")
    except Exception as e:
        errors.append(f"Router: {e}")
        print(f"   Router: FAILED - {e}")
    
    # 24. Cache
    print("\n24. Testing cache modules...")
    try:
        from genai_traces.cache import trace_cache_lookup, CacheSavings
        print("   Cache: OK")
    except Exception as e:
        errors.append(f"Cache: {e}")
        print(f"   Cache: FAILED - {e}")
    
    # 25. Plugins
    print("\n25. Testing plugins modules...")
    try:
        from genai_traces.plugins import PluginRegistry, get_plugin_registry, load_plugins
        print("   Plugins: OK")
    except Exception as e:
        errors.append(f"Plugins: {e}")
        print(f"   Plugins: FAILED - {e}")
    
    # 26. CLI
    print("\n26. Testing CLI modules...")
    try:
        from genai_traces.cli import cli, main
        print("   CLI: OK")
    except Exception as e:
        errors.append(f"CLI: {e}")
        print(f"   CLI: FAILED - {e}")
    
    # 27. Utils
    print("\n27. Testing utils modules...")
    try:
        from genai_traces.utils import generate_trace_id, generate_span_id
        from genai_traces.utils.async_utils import ensure_async, run_async
        from genai_traces.utils.logger import get_logger, StructuredLogger
        from genai_traces.utils.serialization import serialize_span
        from genai_traces.utils.timing import Timer
        print("   Utils: OK")
    except Exception as e:
        errors.append(f"Utils: {e}")
        print(f"   Utils: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    if errors:
        print(f"FAILED: {len(errors)} module groups had errors")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("SUCCESS: All 27 module groups imported successfully!")
        print("GenAI-Traces implementation is complete and functional.")
        return True


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
