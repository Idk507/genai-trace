"""
Main CLI entry point for GenAI-Traces.

Usage:
    genai-traces --help
    genai-traces export --input traces.jsonl --output dataset.jsonl
    genai-traces analyze --input traces.jsonl
    genai-traces prompt list
    genai-traces experiment results my-experiment
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="genai-traces",
        description="GenAI-Traces CLI - LLM observability toolkit",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    export_parser = subparsers.add_parser("export", help="Export traces to datasets")
    export_parser.add_argument("--input", "-i", required=True, help="Input traces file")
    export_parser.add_argument("--output", "-o", required=True, help="Output dataset file")
    export_parser.add_argument("--format", "-f", default="openai", 
                               choices=["openai", "hf", "alpaca", "sharegpt"],
                               help="Output format")
    export_parser.add_argument("--min-quality", type=float, default=0.7,
                               help="Minimum quality score")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze traces")
    analyze_parser.add_argument("--input", "-i", required=True, help="Input traces file")
    analyze_parser.add_argument("--output", "-o", help="Output report file")
    analyze_parser.add_argument("--format", "-f", default="text",
                               choices=["text", "json"],
                               help="Output format")
    
    prompt_parser = subparsers.add_parser("prompt", help="Prompt management")
    prompt_subparsers = prompt_parser.add_subparsers(dest="prompt_command")
    
    prompt_list = prompt_subparsers.add_parser("list", help="List prompts")
    prompt_list.add_argument("--registry", default="./prompt_registry.json",
                            help="Registry file path")
    
    prompt_get = prompt_subparsers.add_parser("get", help="Get a prompt")
    prompt_get.add_argument("name", help="Prompt name")
    prompt_get.add_argument("--version", "-v", help="Specific version")
    prompt_get.add_argument("--registry", default="./prompt_registry.json")
    
    prompt_diff = prompt_subparsers.add_parser("diff", help="Diff prompt versions")
    prompt_diff.add_argument("name", help="Prompt name")
    prompt_diff.add_argument("version1", help="First version")
    prompt_diff.add_argument("version2", help="Second version")
    prompt_diff.add_argument("--registry", default="./prompt_registry.json")
    
    exp_parser = subparsers.add_parser("experiment", help="A/B experiment management")
    exp_subparsers = exp_parser.add_subparsers(dest="exp_command")
    
    exp_list = exp_subparsers.add_parser("list", help="List experiments")
    exp_list.add_argument("--storage", default="./ab_experiments.json")
    
    exp_results = exp_subparsers.add_parser("results", help="Get experiment results")
    exp_results.add_argument("experiment_id", help="Experiment ID")
    exp_results.add_argument("--storage", default="./ab_experiments.json")
    
    serve_parser = subparsers.add_parser("serve", help="Start local trace viewer")
    serve_parser.add_argument("--port", "-p", type=int, default=8080)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--traces-dir", default="./traces")
    
    args = parser.parse_args()
    
    if args.version:
        from ..version import __version__
        print(f"genai-traces {__version__}")
        return 0
    
    if args.command == "export":
        return cmd_export(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "prompt":
        return cmd_prompt(args)
    elif args.command == "experiment":
        return cmd_experiment(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        parser.print_help()
        return 0


def cmd_export(args) -> int:
    """Handle export command."""
    from ..exporters.finetune.exporter import FineTuneExporter, DatasetFormat
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    format_map = {
        "openai": DatasetFormat.OPENAI,
        "hf": DatasetFormat.HUGGINGFACE,
        "alpaca": DatasetFormat.ALPACA,
        "sharegpt": DatasetFormat.SHAREGPT,
    }
    
    exporter = FineTuneExporter(
        min_quality_score=args.min_quality,
        format=format_map[args.format],
    )
    
    count = exporter.export_from_jsonl(str(input_path), args.output)
    
    print(f"Exported {count} records to {args.output}")
    print(f"Stats: {exporter.get_stats()}")
    return 0


def cmd_analyze(args) -> int:
    """Handle analyze command."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    spans = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                spans.append(json.loads(line))
    
    total_cost = 0.0
    total_tokens = 0
    models = {}
    errors = 0
    
    for span in spans:
        attrs = span.get("attributes", {})
        
        cost = attrs.get("cost.total_usd", 0)
        total_cost += cost if cost else 0
        
        tokens = attrs.get("llm.total_tokens", 0)
        total_tokens += tokens if tokens else 0
        
        model = attrs.get("llm.model.name", "unknown")
        models[model] = models.get(model, 0) + 1
        
        if span.get("status") == "error":
            errors += 1
    
    report = {
        "total_spans": len(spans),
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "error_count": errors,
        "error_rate": round(errors / len(spans), 4) if spans else 0,
        "models": models,
        "avg_tokens_per_span": round(total_tokens / len(spans), 1) if spans else 0,
        "avg_cost_per_span": round(total_cost / len(spans), 6) if spans else 0,
    }
    
    if args.format == "json":
        output = json.dumps(report, indent=2)
    else:
        output = f"""
GenAI-Traces Analysis Report
============================
Total Spans: {report['total_spans']}
Total Cost: ${report['total_cost_usd']:.6f}
Total Tokens: {report['total_tokens']:,}
Error Rate: {report['error_rate']:.2%}

Models Used:
"""
        for model, count in report['models'].items():
            output += f"  - {model}: {count}\n"
        
        output += f"""
Averages:
  - Tokens per span: {report['avg_tokens_per_span']:.1f}
  - Cost per span: ${report['avg_cost_per_span']:.6f}
"""
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    return 0


def cmd_prompt(args) -> int:
    """Handle prompt commands."""
    from ..prompt_management.registry import PromptRegistry
    
    if args.prompt_command == "list":
        registry = PromptRegistry(args.registry)
        prompts = registry.list_prompts()
        
        if not prompts:
            print("No prompts found.")
        else:
            print("Registered Prompts:")
            for name in prompts:
                versions = registry.list_versions(name)
                print(f"  - {name} ({len(versions)} versions)")
        return 0
    
    elif args.prompt_command == "get":
        registry = PromptRegistry(args.registry)
        pv = registry.get(args.name, version=args.version)
        
        if not pv:
            print(f"Prompt not found: {args.name}", file=sys.stderr)
            return 1
        
        print(f"Name: {pv.name}")
        print(f"Version: {pv.version}")
        print(f"Labels: {pv.labels}")
        print(f"Template:\n{pv.template}")
        return 0
    
    elif args.prompt_command == "diff":
        registry = PromptRegistry(args.registry)
        diff = registry.diff(args.name, args.version1, args.version2)
        
        if diff is None:
            print("Could not compute diff", file=sys.stderr)
            return 1
        
        print(diff)
        return 0
    
    return 0


def cmd_experiment(args) -> int:
    """Handle experiment commands."""
    from ..prompt_management.ab_testing import ABTestManager
    
    if args.exp_command == "list":
        manager = ABTestManager(args.storage)
        experiments = manager.list_experiments()
        
        if not experiments:
            print("No experiments found.")
        else:
            print("Experiments:")
            for exp in experiments:
                print(f"  - {exp.experiment_id} ({exp.status})")
        return 0
    
    elif args.exp_command == "results":
        manager = ABTestManager(args.storage)
        summary = manager.get_results_summary(args.experiment_id)
        
        if not summary:
            print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
            return 1
        
        print(json.dumps(summary, indent=2))
        return 0
    
    return 0


def cmd_serve(args) -> int:
    """Handle serve command."""
    print(f"Starting trace viewer on http://{args.host}:{args.port}")
    print(f"Traces directory: {args.traces_dir}")
    print("(Trace viewer not implemented in this version)")
    return 0


def main():
    """Main entry point."""
    sys.exit(cli())


if __name__ == "__main__":
    main()
