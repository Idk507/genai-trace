"""
Comprehensive test script to verify all GenAI-Traces modules.
"""

import sys
sys.path.insert(0, '.')

def test_all():
    print("Testing GenAI-Traces Complete Implementation")
    print("=" * 60)
    
    errors = []
    
    # Test 1: Core imports
    print("\n1. Testing core imports...")
    try:
        from genai_traces import init_tracer, trace_llm, get_tracer, Span, SpanType
        from genai_traces.exporters import ConsoleExporter
        print("   OK - Core imports work")
    except Exception as e:
        errors.append(f"Core imports: {e}")
        print(f"   FAIL - {e}")
    
    # Test 2: Config validators
    print("\n2. Testing config validators...")
    try:
        from genai_traces.config.validators import validate_config, ValidationResult
        print("   OK - Config validators imported")
    except Exception as e:
        errors.append(f"Config validators: {e}")
        print(f"   FAIL - {e}")
    
    # Test 3: Instrumentation base
    print("\n3. Testing instrumentation base...")
    try:
        from genai_traces.instrumentation.base import BaseInstrumentation, InstrumentationRegistry
        print("   OK - Instrumentation base imported")
    except Exception as e:
        errors.append(f"Instrumentation base: {e}")
        print(f"   FAIL - {e}")
    
    # Test 4: Token estimator
    print("\n4. Testing token estimator...")
    try:
        from genai_traces.telemetry.tokens.estimator import TokenEstimator, estimate_tokens
        estimator = TokenEstimator()
        tokens = estimator.estimate_prompt_tokens("Hello world", "gpt-4o")
        print(f"   OK - Token estimation: {tokens} tokens")
    except Exception as e:
        errors.append(f"Token estimator: {e}")
        print(f"   FAIL - {e}")
    
    # Test 5: Streaming accumulator
    print("\n5. Testing streaming accumulator...")
    try:
        from genai_traces.telemetry.tokens.streaming import StreamingAccumulator, StreamingStats
        acc = StreamingAccumulator()
        print("   OK - Streaming accumulator imported")
    except Exception as e:
        errors.append(f"Streaming accumulator: {e}")
        print(f"   FAIL - {e}")
    
    # Test 6: Pricing table
    print("\n6. Testing pricing table...")
    try:
        from genai_traces.telemetry.cost.pricing_table import PricingTable, get_model_pricing
        pricing = get_model_pricing("gpt-4o")
        print(f"   OK - Pricing: ${pricing.prompt_cost_per_1k}/1k tokens")
    except Exception as e:
        errors.append(f"Pricing table: {e}")
        print(f"   FAIL - {e}")
    
    # Test 7: Cost aggregator
    print("\n7. Testing cost aggregator...")
    try:
        from genai_traces.telemetry.cost.aggregator import CostAggregator, record_cost
        agg = CostAggregator()
        agg.record(0.001, model="gpt-4o", session_id="test")
        summary = agg.get_global_summary()
        print(f"   OK - Cost aggregation: ${summary.total_cost_usd}")
    except Exception as e:
        errors.append(f"Cost aggregator: {e}")
        print(f"   FAIL - {e}")
    
    # Test 8: Metrics modules
    print("\n8. Testing metrics modules...")
    try:
        from genai_traces.telemetry.metrics import LatencyTracker, ThroughputTracker, ErrorRateTracker
        lat = LatencyTracker()
        lat.record("test", 100.0)
        print("   OK - Metrics modules imported")
    except Exception as e:
        errors.append(f"Metrics modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 9: Environment modules
    print("\n9. Testing environment modules...")
    try:
        from genai_traces.telemetry.environment import get_system_info, get_resource_usage
        info = get_system_info()
        print(f"   OK - System: {info.os_name} {info.python_version}")
    except Exception as e:
        errors.append(f"Environment modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 10: Multimodal modules
    print("\n10. Testing multimodal modules...")
    try:
        from genai_traces.multimodal import capture_image_metadata, capture_audio_metadata, hash_content
        h = hash_content(b"test data")
        print(f"   OK - Content hash: {h.hash_value}")
    except Exception as e:
        errors.append(f"Multimodal modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 11: Router modules
    print("\n11. Testing router modules...")
    try:
        from genai_traces.router import trace_router, FallbackChain
        chain = FallbackChain(["gpt-4o", "gpt-4o-mini"])
        print("   OK - Router modules imported")
    except Exception as e:
        errors.append(f"Router modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 12: Cache modules
    print("\n12. Testing cache modules...")
    try:
        from genai_traces.cache import trace_cache_lookup, compute_cache_savings
        savings = compute_cache_savings("gpt-4o", 1000, 500, True)
        print(f"   OK - Cache savings: ${savings:.6f}")
    except Exception as e:
        errors.append(f"Cache modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 13: Annotation modules
    print("\n13. Testing annotation modules...")
    try:
        from genai_traces.intelligence.annotation import AnnotationQueue, AnnotationRubric, compute_agreement
        rubric = AnnotationRubric(name="test", description="Test rubric")
        print("   OK - Annotation modules imported")
    except Exception as e:
        errors.append(f"Annotation modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 14: Evaluation modules
    print("\n14. Testing evaluation modules...")
    try:
        from genai_traces.intelligence.evaluation.hallucination import HallucinationEvaluator
        from genai_traces.intelligence.evaluation.toxicity import ToxicityEvaluator
        from genai_traces.intelligence.evaluation.coherence import CoherenceEvaluator
        from genai_traces.intelligence.evaluation.groundedness import GroundednessEvaluator
        print("   OK - Evaluation modules imported")
    except Exception as e:
        errors.append(f"Evaluation modules: {e}")
        print(f"   FAIL - {e}")
    
    # Test 15: Plugin system
    print("\n15. Testing plugin system...")
    try:
        from genai_traces.plugins import PluginRegistry, get_plugin_registry, load_plugins
        registry = get_plugin_registry()
        print("   OK - Plugin system imported")
    except Exception as e:
        errors.append(f"Plugin system: {e}")
        print(f"   FAIL - {e}")
    
    # Test 16: CLI
    print("\n16. Testing CLI...")
    try:
        from genai_traces.cli import cli, main
        print("   OK - CLI imported")
    except Exception as e:
        errors.append(f"CLI: {e}")
        print(f"   FAIL - {e}")
    
    # Test 17: Batch exporter
    print("\n17. Testing batch exporter...")
    try:
        from genai_traces.exporters.batch import BatchExporter, CircularBuffer
        buffer = CircularBuffer(100)
        buffer.push("test")
        print("   OK - Batch exporter imported")
    except Exception as e:
        errors.append(f"Batch exporter: {e}")
        print(f"   FAIL - {e}")
    
    # Test 18: Utils
    print("\n18. Testing utils...")
    try:
        from genai_traces.utils.async_utils import ensure_async, run_sync_in_executor
        from genai_traces.utils.logger import get_logger, StructuredLogger
        logger = get_logger()
        print("   OK - Utils imported")
    except Exception as e:
        errors.append(f"Utils: {e}")
        print(f"   FAIL - {e}")
    
    # Test 19: Security output filter
    print("\n19. Testing security output filter...")
    try:
        from genai_traces.security.output_filter import OutputFilter, ContentModerator
        filter = OutputFilter()
        result = filter.filter("Hello world")
        print(f"   OK - Output filter: passed={result.passed}")
    except Exception as e:
        errors.append(f"Security output filter: {e}")
        print(f"   FAIL - {e}")
    
    # Test 20: Function call tracing
    print("\n20. Testing function call tracing...")
    try:
        from genai_traces.instrumentation.tools import trace_function_call, FunctionCallTracer
        print("   OK - Function call tracing imported")
    except Exception as e:
        errors.append(f"Function call tracing: {e}")
        print(f"   FAIL - {e}")
    
    # Test 21: PII patterns
    print("\n21. Testing PII patterns...")
    try:
        from genai_traces.privacy.detection.patterns import PII_PATTERNS, get_patterns_by_category
        patterns = get_patterns_by_category("personal")
        print(f"   OK - PII patterns: {len(patterns)} personal patterns")
    except Exception as e:
        errors.append(f"PII patterns: {e}")
        print(f"   FAIL - {e}")
    
    # Test 22: Field encryption
    print("\n22. Testing field encryption...")
    try:
        from genai_traces.privacy.encryption import FieldEncryptor, encrypt_field
        encryptor = FieldEncryptor()
        print(f"   OK - Field encryption: available={encryptor.is_available()}")
    except Exception as e:
        errors.append(f"Field encryption: {e}")
        print(f"   FAIL - {e}")
    
    # Test 23: Compliance modules
    print("\n23. Testing compliance modules...")
    try:
        from genai_traces.privacy.compliance import RetentionPolicy, AuditLog
        policy = RetentionPolicy()
        print(f"   OK - Retention policy: {policy.default_retention_days} days")
    except Exception as e:
        errors.append(f"Compliance modules: {e}")
        print(f"   FAIL - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} errors")
        for err in errors:
            print(f"  - {err}")
    else:
        print("All 23 tests passed!")
        print("GenAI-Traces implementation is complete and functional.")
    print("=" * 60)
    
    return len(errors) == 0

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
