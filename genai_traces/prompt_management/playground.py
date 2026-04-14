"""
Prompt playground for GenAI-Traces.

Provides a CLI-driven prompt sandbox for testing prompts.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class PlaygroundRun:
    """A single run in the playground."""
    run_id: str
    prompt_name: str
    prompt_version: str
    variables: Dict[str, Any]
    rendered_prompt: str
    response: Optional[str] = None
    duration_ms: Optional[float] = None
    tokens: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "variables": self.variables,
            "rendered_prompt": self.rendered_prompt,
            "response": self.response,
            "duration_ms": self.duration_ms,
            "tokens": self.tokens,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class PromptPlayground:
    """
    Interactive prompt playground for testing and iterating on prompts.
    
    Usage:
        from genai_traces.prompt_management.playground import PromptPlayground
        
        playground = PromptPlayground(llm_fn=my_llm_call)
        
        # Test a prompt
        result = playground.run(
            prompt_template="Hello {name}, how can I help you today?",
            variables={"name": "Alice"},
        )
        
        # Compare versions
        comparison = playground.compare(
            templates=[template_v1, template_v2],
            variables={"name": "Bob"},
        )
    """
    
    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        registry=None,
    ):
        """
        Initialize the playground.
        
        Args:
            llm_fn: Function to call LLM with a prompt
            registry: Optional PromptRegistry instance
        """
        self._llm_fn = llm_fn
        self._registry = registry
        self._history: List[PlaygroundRun] = []
        self._run_counter = 0
    
    def set_llm_fn(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self._llm_fn = llm_fn
    
    def run(
        self,
        prompt_template: str,
        variables: Optional[Dict[str, Any]] = None,
        prompt_name: str = "playground",
        prompt_version: str = "dev",
    ) -> PlaygroundRun:
        """
        Run a prompt in the playground.
        
        Args:
            prompt_template: The prompt template
            variables: Variables to substitute
            prompt_name: Name for tracking
            prompt_version: Version for tracking
            
        Returns:
            PlaygroundRun with results
        """
        import time
        import uuid
        
        self._run_counter += 1
        run_id = f"run-{self._run_counter}-{uuid.uuid4().hex[:6]}"
        
        variables = variables or {}
        rendered = self._render_template(prompt_template, variables)
        
        run = PlaygroundRun(
            run_id=run_id,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            variables=variables,
            rendered_prompt=rendered,
        )
        
        if self._llm_fn:
            start_time = time.perf_counter()
            try:
                response = self._llm_fn(rendered)
                run.response = response
                run.duration_ms = (time.perf_counter() - start_time) * 1000
            except Exception as e:
                run.error = str(e)
                run.duration_ms = (time.perf_counter() - start_time) * 1000
        
        self._history.append(run)
        return run
    
    def run_from_registry(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> PlaygroundRun:
        """
        Run a prompt from the registry.
        
        Args:
            prompt_name: Name of the prompt in registry
            variables: Variables to substitute
            version: Optional specific version
            
        Returns:
            PlaygroundRun with results
        """
        if not self._registry:
            raise ValueError("No registry configured")
        
        prompt = self._registry.get(prompt_name, version=version)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        return self.run(
            prompt_template=prompt.template,
            variables=variables,
            prompt_name=prompt_name,
            prompt_version=prompt.version,
        )
    
    def compare(
        self,
        templates: List[str],
        variables: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> List[PlaygroundRun]:
        """
        Compare multiple prompt templates.
        
        Args:
            templates: List of prompt templates to compare
            variables: Variables to substitute
            labels: Optional labels for each template
            
        Returns:
            List of PlaygroundRun results
        """
        results = []
        labels = labels or [f"variant_{i}" for i in range(len(templates))]
        
        for template, label in zip(templates, labels):
            run = self.run(
                prompt_template=template,
                variables=variables,
                prompt_name="comparison",
                prompt_version=label,
            )
            results.append(run)
        
        return results
    
    def iterate(
        self,
        base_template: str,
        modifications: List[Dict[str, str]],
        variables: Optional[Dict[str, Any]] = None,
    ) -> List[PlaygroundRun]:
        """
        Iterate on a prompt with modifications.
        
        Args:
            base_template: The base prompt template
            modifications: List of {find: replace} modifications
            variables: Variables to substitute
            
        Returns:
            List of PlaygroundRun results
        """
        results = []
        
        results.append(self.run(
            prompt_template=base_template,
            variables=variables,
            prompt_name="iteration",
            prompt_version="base",
        ))
        
        current_template = base_template
        for i, mod in enumerate(modifications):
            for find, replace in mod.items():
                current_template = current_template.replace(find, replace)
            
            results.append(self.run(
                prompt_template=current_template,
                variables=variables,
                prompt_name="iteration",
                prompt_version=f"mod_{i+1}",
            ))
        
        return results
    
    def get_history(
        self,
        limit: int = 10,
        prompt_name: Optional[str] = None,
    ) -> List[PlaygroundRun]:
        """Get recent playground runs."""
        history = self._history
        
        if prompt_name:
            history = [r for r in history if r.prompt_name == prompt_name]
        
        return list(reversed(history[-limit:]))
    
    def clear_history(self) -> None:
        """Clear run history."""
        self._history.clear()
        self._run_counter = 0
    
    def export_history(self, path: str) -> None:
        """Export history to JSON file."""
        data = [run.to_dict() for run in self._history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render a template with variables."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


def create_playground(
    llm_fn: Optional[Callable[[str], str]] = None,
) -> PromptPlayground:
    """Create a new playground instance."""
    return PromptPlayground(llm_fn=llm_fn)
