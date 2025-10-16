#!/usr/bin/env python3
"""
Tool Integration System for LLM Behavior Lab

Implements different approaches for integrating tool outputs into LLM conversations,
measuring their impact on energy consumption and alignment.
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import time


class ToolIntegrationMethod(Enum):
    """Methods for integrating tool outputs."""
    NONE = "none"
    DIRECT_INSERTION = "direct_insertion"
    SUMMARIZED_INSERTION = "summarized_insertion"
    FILTERED_INSERTION = "filtered_insertion"
    STAGED_INSERTION = "staged_insertion"
    ADAPTIVE_INSERTION = "adaptive_insertion"


@dataclass
class ToolCall:
    """Represents a tool call and its result."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    tokens_input: int = 0
    tokens_output: int = 0


@dataclass
class ToolIntegrationConfig:
    """Configuration for tool integration."""
    method: ToolIntegrationMethod = ToolIntegrationMethod.NONE
    max_tool_calls: int = 3
    max_output_tokens_per_tool: int = 500
    summarization_enabled: bool = True
    filtering_enabled: bool = True
    staging_enabled: bool = False


class ToolIntegrator(ABC):
    """Abstract base class for tool integrators."""

    @abstractmethod
    async def integrate_tools(self, tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate tool outputs into conversation.

        Returns dict with:
        - modified_prompts: Dict of modified system/user prompts
        - integration_metadata: Dict with integration details
        - processed_tools: List of processed tool calls
        """
        pass

    @property
    @abstractmethod
    def integration_method(self) -> ToolIntegrationMethod:
        pass


class NoToolIntegration(ToolIntegrator):
    """No tool integration - baseline."""

    @property
    def integration_method(self) -> ToolIntegrationMethod:
        return ToolIntegrationMethod.NONE

    async def integrate_tools(self, tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "modified_prompts": {
                "system": conversation_context.get("system_prompt", ""),
                "user": conversation_context.get("user_prompt", ""),
            },
            "integration_metadata": {
                "method": "none",
                "tools_processed": 0,
                "total_tokens_added": 0,
                "processing_time": 0.0,
            },
            "processed_tools": [],
        }


class DirectInsertionIntegrator(ToolIntegrator):
    """Directly insert tool outputs into the conversation."""

    @property
    def integration_method(self) -> ToolIntegrationMethod:
        return ToolIntegrationMethod.DIRECT_INSERTION

    async def integrate_tools(self, tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:

        start_time = time.time()
        system_prompt = conversation_context.get("system_prompt", "")
        user_prompt = conversation_context.get("user_prompt", "")

        # Build tool results section
        tool_results = []
        total_tokens = 0

        for call in tool_calls:
            if call.success:
                result_text = f"Tool: {call.tool_name}\nInput: {json.dumps(call.tool_input, indent=2)}\nOutput: {str(call.tool_output)}"
                tool_results.append(result_text)
                total_tokens += call.tokens_input + call.tokens_output

        # Insert at the end of user prompt
        tool_section = "\n\nTool Results:\n" + "\n\n".join(tool_results) if tool_results else ""
        modified_user = user_prompt + tool_section

        processing_time = time.time() - start_time

        return {
            "modified_prompts": {
                "system": system_prompt,
                "user": modified_user,
            },
            "integration_metadata": {
                "method": "direct_insertion",
                "tools_processed": len(tool_calls),
                "successful_tools": len([c for c in tool_calls if c.success]),
                "total_tokens_added": total_tokens,
                "processing_time": processing_time,
                "insertion_position": "end_of_user",
            },
            "processed_tools": tool_calls,
        }


class SummarizedInsertionIntegrator(ToolIntegrator):
    """Summarize tool outputs before insertion."""

    @property
    def integration_method(self) -> ToolIntegrationMethod:
        return ToolIntegrationMethod.SUMMARIZED_INSERTION

    async def integrate_tools(self, tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:

        start_time = time.time()
        system_prompt = conversation_context.get("system_prompt", "")
        user_prompt = conversation_context.get("user_prompt", "")

        # Summarize each tool result
        summarized_results = []
        total_original_tokens = 0
        total_summary_tokens = 0

        for call in tool_calls:
            if call.success:
                total_original_tokens += call.tokens_input + call.tokens_output

                # Simple summarization (in practice, use LLM for better summarization)
                summary = await self._summarize_tool_output(call)
                summarized_results.append(f"{call.tool_name}: {summary}")
                total_summary_tokens += len(summary.split())  # Rough estimate

        # Insert summarized results
        summary_section = "\n\nTool Summaries:\n" + "\n".join(summarized_results) if summarized_results else ""
        modified_user = user_prompt + summary_section

        processing_time = time.time() - start_time

        return {
            "modified_prompts": {
                "system": system_prompt,
                "user": modified_user,
            },
            "integration_metadata": {
                "method": "summarized_insertion",
                "tools_processed": len(tool_calls),
                "successful_tools": len([c for c in tool_calls if c.success]),
                "original_tokens": total_original_tokens,
                "summary_tokens": total_summary_tokens,
                "compression_ratio": total_summary_tokens / total_original_tokens if total_original_tokens > 0 else 0,
                "processing_time": processing_time,
            },
            "processed_tools": tool_calls,
        }

    async def _summarize_tool_output(self, tool_call: ToolCall) -> str:
        """Simple tool output summarization."""
        # In a real implementation, this would use an LLM for summarization
        # For now, just truncate and simplify
        output_str = str(tool_call.tool_output)
        if len(output_str) > 200:
            return f"[{tool_call.tool_name} returned {len(output_str)} chars of data - key findings extracted]"
        return f"Result: {output_str[:200]}"


class FilteredInsertionIntegrator(ToolIntegrator):
    """Filter and prioritize tool outputs before insertion."""

    @property
    def integration_method(self) -> ToolIntegrationMethod:
        return ToolIntegrationMethod.FILTERED_INSERTION

    async def integrate_tools(self, tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:

        start_time = time.time()
        system_prompt = conversation_context.get("system_prompt", "")
        user_prompt = conversation_context.get("user_prompt", "")

        # Filter and prioritize tool results
        filtered_results = []
        total_tokens = 0

        # Sort by relevance (simplified - in practice use relevance scoring)
        relevant_calls = [call for call in tool_calls if call.success and self._is_relevant(call, conversation_context)]

        # Limit to most relevant
        max_calls = conversation_context.get("config", {}).get("max_tool_calls", 3)
        prioritized_calls = relevant_calls[:max_calls]

        for call in prioritized_calls:
            # Filter out irrelevant or redundant information
            filtered_output = self._filter_tool_output(call)
            if filtered_output:
                result_text = f"Tool: {call.tool_name}\n{filtered_output}"
                filtered_results.append(result_text)
                total_tokens += len(filtered_output.split())

        # Insert filtered results
        filtered_section = "\n\nFiltered Tool Results:\n" + "\n\n".join(filtered_results) if filtered_results else ""
        modified_user = user_prompt + filtered_section

        processing_time = time.time() - start_time

        return {
            "modified_prompts": {
                "system": system_prompt,
                "user": modified_user,
            },
            "integration_metadata": {
                "method": "filtered_insertion",
                "tools_processed": len(tool_calls),
                "tools_filtered": len(relevant_calls),
                "tools_included": len(prioritized_calls),
                "total_tokens_added": total_tokens,
                "processing_time": processing_time,
            },
            "processed_tools": tool_calls,
        }

    def _is_relevant(self, tool_call: ToolCall, context: Dict[str, Any]) -> bool:
        """Check if tool call is relevant to the conversation."""
        # Simplified relevance check - in practice use semantic similarity
        user_prompt = context.get("user_prompt", "").lower()
        tool_output = str(tool_call.tool_output).lower()

        # Check for keyword matches
        keywords = self._extract_keywords(user_prompt)
        return any(keyword in tool_output for keyword in keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Very simple keyword extraction
        words = text.split()
        return [word for word in words if len(word) > 3][:5]  # Top 5 longer words

    def _filter_tool_output(self, tool_call: ToolCall) -> str:
        """Filter tool output to remove noise."""
        output = str(tool_call.tool_output)

        # Remove common noise patterns
        filtered = re.sub(r'\n\s*\n', '\n', output)  # Multiple newlines
        filtered = re.sub(r'\s+', ' ', filtered)     # Multiple spaces

        # Truncate if too long
        if len(filtered) > 500:
            filtered = filtered[:500] + "..."

        return filtered


class StagedInsertionIntegrator(ToolIntegrator):
    """Insert tool outputs in stages during conversation."""

    @property
    def integration_method(self) -> ToolIntegrationMethod:
        return ToolIntegrationMethod.STAGED_INSERTION

    async def integrate_tools(self, tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:

        # For staged insertion, we prepare multiple conversation stages
        # Each stage adds different tool results
        stages = []
        current_system = conversation_context.get("system_prompt", "")
        current_user = conversation_context.get("user_prompt", "")

        for i, call in enumerate(tool_calls):
            if call.success:
                # Create a new stage with this tool result
                stage_user = current_user + f"\n\nTool Result {i+1}: {call.tool_name}\n{str(call.tool_output)}"

                stages.append({
                    "stage": i + 1,
                    "system_prompt": current_system,
                    "user_prompt": stage_user,
                    "tool_call": call,
                })

        return {
            "modified_prompts": {
                "system": current_system,
                "user": current_user,  # Base prompt without tools
            },
            "integration_metadata": {
                "method": "staged_insertion",
                "total_stages": len(stages),
                "stages": stages,
                "processing_time": 0.0,  # Would track actual processing
            },
            "processed_tools": tool_calls,
        }


class ToolIntegrationManager:
    """Manages different tool integration strategies."""

    def __init__(self):
        self.integrators = {
            ToolIntegrationMethod.NONE: NoToolIntegration(),
            ToolIntegrationMethod.DIRECT_INSERTION: DirectInsertionIntegrator(),
            ToolIntegrationMethod.SUMMARIZED_INSERTION: SummarizedInsertionIntegrator(),
            ToolIntegrationMethod.FILTERED_INSERTION: FilteredInsertionIntegrator(),
            ToolIntegrationMethod.STAGED_INSERTION: StagedInsertionIntegrator(),
        }

    def get_integrator(self, method: ToolIntegrationMethod) -> ToolIntegrator:
        """Get integrator for a specific method."""
        return self.integrators.get(method, self.integrators[ToolIntegrationMethod.NONE])

    async def integrate_tools(self, config: ToolIntegrationConfig,
                            tool_calls: List[ToolCall],
                            conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tool integration based on configuration."""

        integrator = self.get_integrator(config.method)
        result = await integrator.integrate_tools(tool_calls, conversation_context)

        # Add config metadata
        result["integration_metadata"].update({
            "config": {
                "method": config.method.value,
                "max_tool_calls": config.max_tool_calls,
                "max_output_tokens_per_tool": config.max_output_tokens_per_tool,
                "summarization_enabled": config.summarization_enabled,
                "filtering_enabled": config.filtering_enabled,
                "staging_enabled": config.staging_enabled,
            }
        })

        return result

    def get_available_methods(self) -> List[Dict[str, Any]]:
        """Get list of available integration methods."""
        return [
            {
                "method": ToolIntegrationMethod.NONE.value,
                "name": "No Integration",
                "description": "Baseline without tool integration",
                "energy_impact": "baseline",
                "alignment_impact": "baseline (no tool benefits)",
            },
            {
                "method": ToolIntegrationMethod.DIRECT_INSERTION.value,
                "name": "Direct Insertion",
                "description": "Insert raw tool outputs directly",
                "energy_impact": "high (full tool output tokens)",
                "alignment_impact": "high (complete tool information)",
            },
            {
                "method": ToolIntegrationMethod.SUMMARIZED_INSERTION.value,
                "name": "Summarized Insertion",
                "description": "Summarize tool outputs before insertion",
                "energy_impact": "medium (compressed tokens)",
                "alignment_impact": "medium (key information preserved)",
            },
            {
                "method": ToolIntegrationMethod.FILTERED_INSERTION.value,
                "name": "Filtered Insertion",
                "description": "Filter and prioritize tool outputs",
                "energy_impact": "low (relevant info only)",
                "alignment_impact": "high (focused on relevant data)",
            },
            {
                "method": ToolIntegrationMethod.STAGED_INSERTION.value,
                "name": "Staged Insertion",
                "description": "Add tool results incrementally",
                "energy_impact": "variable (per stage)",
                "alignment_impact": "high (contextual integration)",
            },
        ]


# Mock tool implementations for testing
async def mock_search_tool(query: str) -> Dict[str, Any]:
    """Mock search tool that returns simulated results."""
    await asyncio.sleep(0.1)  # Simulate API call
    return {
        "query": query,
        "results": [
            {"title": f"Result 1 for {query}", "snippet": f"Relevant information about {query}..."},
            {"title": f"Result 2 for {query}", "snippet": f"More details on {query}..."},
        ]
    }


async def mock_python_tool(code: str) -> Dict[str, Any]:
    """Mock Python execution tool."""
    await asyncio.sleep(0.05)  # Simulate execution
    # Very basic execution simulation
    if "print(" in code:
        return {"output": "Hello World", "success": True}
    return {"output": "Code executed successfully", "success": True}


# Global tool integration manager
tool_manager = ToolIntegrationManager()
