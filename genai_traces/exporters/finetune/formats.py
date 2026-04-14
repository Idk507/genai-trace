"""
Fine-tuning format converters for GenAI-Traces.

Converts traces to various fine-tuning dataset formats.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str
    content: str


def to_openai_format(
    conversations: List[List[ConversationTurn]],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert to OpenAI fine-tuning format.
    
    Args:
        conversations: List of conversations (each is a list of turns)
        system_prompt: Optional system prompt to prepend
        
    Returns:
        List of records in OpenAI format
    """
    records = []
    
    for conversation in conversations:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for turn in conversation:
            messages.append({"role": turn.role, "content": turn.content})
        
        records.append({"messages": messages})
    
    return records


def to_huggingface_format(
    conversations: List[List[ConversationTurn]],
    format_type: str = "conversational",
) -> List[Dict[str, Any]]:
    """
    Convert to HuggingFace dataset format.
    
    Args:
        conversations: List of conversations
        format_type: "conversational" or "text"
        
    Returns:
        List of records in HuggingFace format
    """
    records = []
    
    for conversation in conversations:
        if format_type == "conversational":
            messages = [
                {"role": turn.role, "content": turn.content}
                for turn in conversation
            ]
            records.append({"messages": messages})
        else:
            text = "\n".join(
                f"{turn.role}: {turn.content}"
                for turn in conversation
            )
            records.append({"text": text})
    
    return records


def to_alpaca_format(
    conversations: List[List[ConversationTurn]],
) -> List[Dict[str, Any]]:
    """
    Convert to Alpaca format (instruction, input, output).
    
    Args:
        conversations: List of conversations (expects user/assistant pairs)
        
    Returns:
        List of records in Alpaca format
    """
    records = []
    
    for conversation in conversations:
        user_turns = [t for t in conversation if t.role == "user"]
        assistant_turns = [t for t in conversation if t.role == "assistant"]
        
        if user_turns and assistant_turns:
            records.append({
                "instruction": user_turns[0].content,
                "input": "",
                "output": assistant_turns[0].content,
            })
    
    return records


def to_sharegpt_format(
    conversations: List[List[ConversationTurn]],
) -> List[Dict[str, Any]]:
    """
    Convert to ShareGPT format.
    
    Args:
        conversations: List of conversations
        
    Returns:
        List of records in ShareGPT format
    """
    records = []
    
    for conversation in conversations:
        conv_data = []
        for turn in conversation:
            from_role = "human" if turn.role == "user" else "gpt"
            conv_data.append({
                "from": from_role,
                "value": turn.content,
            })
        
        records.append({"conversations": conv_data})
    
    return records


def from_traces(
    traces: List[Dict[str, Any]],
    extract_prompt: str = "llm.prompt",
    extract_completion: str = "llm.completion",
) -> List[List[ConversationTurn]]:
    """
    Extract conversations from traces.
    
    Args:
        traces: List of trace dictionaries
        extract_prompt: Attribute key for prompt
        extract_completion: Attribute key for completion
        
    Returns:
        List of conversations
    """
    conversations = []
    
    for trace in traces:
        attrs = trace.get("attributes", {})
        prompt = attrs.get(extract_prompt)
        completion = attrs.get(extract_completion)
        
        if prompt and completion:
            conversations.append([
                ConversationTurn(role="user", content=prompt),
                ConversationTurn(role="assistant", content=completion),
            ])
    
    return conversations


def export_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Export records to JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class FormatConverter:
    """
    Converts traces to various fine-tuning formats.
    
    Usage:
        converter = FormatConverter()
        
        # From traces
        conversations = converter.from_traces(traces)
        
        # To OpenAI format
        openai_data = converter.to_openai(conversations)
        converter.export("dataset.jsonl", openai_data)
    """
    
    def from_traces(
        self,
        traces: List[Dict[str, Any]],
        prompt_key: str = "llm.prompt",
        completion_key: str = "llm.completion",
    ) -> List[List[ConversationTurn]]:
        """Extract conversations from traces."""
        return from_traces(traces, prompt_key, completion_key)
    
    def to_openai(
        self,
        conversations: List[List[ConversationTurn]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert to OpenAI format."""
        return to_openai_format(conversations, system_prompt)
    
    def to_huggingface(
        self,
        conversations: List[List[ConversationTurn]],
        format_type: str = "conversational",
    ) -> List[Dict[str, Any]]:
        """Convert to HuggingFace format."""
        return to_huggingface_format(conversations, format_type)
    
    def to_alpaca(
        self,
        conversations: List[List[ConversationTurn]],
    ) -> List[Dict[str, Any]]:
        """Convert to Alpaca format."""
        return to_alpaca_format(conversations)
    
    def to_sharegpt(
        self,
        conversations: List[List[ConversationTurn]],
    ) -> List[Dict[str, Any]]:
        """Convert to ShareGPT format."""
        return to_sharegpt_format(conversations)
    
    def export(self, path: str, records: List[Dict[str, Any]]) -> None:
        """Export to JSONL file."""
        export_jsonl(records, path)
