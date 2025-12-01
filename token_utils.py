#!/usr/bin/env python3
"""
Token Counting Utilities for LLM Behaviour Lab

Provides accurate token counting and middleware overhead tracking.
Critical for measuring real-world LLM system costs vs isolated model metrics.
"""

from typing import List, Dict, Any, Tuple
import re


def count_tokens_approximate(messages: List[Dict[str, str]]) -> int:
    """
    Approximate token count using character-based heuristic.
    
    Heuristic: 1 token ≈ 4 characters (conservative for English text)
    This is a reasonable approximation for most LLMs.
    
    TODO: For exact counts, integrate tiktoken library. However, note that
    tiktoken is GPT-specific and may not match Llama/Mistral/Qwen tokenizers.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
    
    Returns:
        Estimated token count
    
    Examples:
        >>> count_tokens_approximate([{"role": "user", "content": "Hello world"}])
        7  # "Hello world" = 11 chars / 4 + 4 overhead = ~7 tokens
    """
    if not messages:
        return 0
    
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        total_chars += len(content)
    
    # Add overhead for message structure (role, formatting, special tokens)
    # Each message adds ~4 tokens for: <role>, <content>, </content>, <eom>
    message_overhead = len(messages) * 4
    
    # Character-based estimation
    char_tokens = total_chars // 4
    
    return char_tokens + message_overhead


def track_middleware_tokens(
    original_messages: List[Dict],
    messages_after_injection: List[Dict],
    messages_after_tools: List[Dict]
) -> Dict[str, int]:
    """
    Track token additions at each middleware stage.
    
    This is THE CRITICAL FUNCTION for measuring real-world LLM system costs.
    It reveals how much overhead middleware adds to the "250 token query" myth.
    
    Args:
        original_messages: Messages before any middleware
        messages_after_injection: After prompt injection applied
        messages_after_tools: After tool integration applied
    
    Returns:
        {
            "original_tokens": int,        # Baseline user input
            "injection_added": int,        # Tokens added by prompt injection
            "tools_added": int,            # Tokens added by tool integration
            "total_middleware": int,       # Sum of all middleware overhead
            "final_total": int            # Total input tokens before generation
        }
    
    Examples:
        >>> original = [{"role": "user", "content": "Explain AI"}]
        >>> after_inj = [
        ...     {"role": "system", "content": "You are helpful..." * 1000},  # 5000 tokens
        ...     {"role": "user", "content": "Explain AI"}
        ... ]
        >>> after_tools = [...] + [{"role": "system", "content": "Tool results..."}]  # +3000 tokens
        >>> track = track_middleware_tokens(original, after_inj, after_tools)
        >>> track["injection_added"]  # ~5000
        >>> track["tools_added"]       # ~3000
        >>> track["total_middleware"]  # ~8000 (32x the original 250 token query!)
    """
    original = count_tokens_approximate(original_messages)
    after_injection = count_tokens_approximate(messages_after_injection)
    after_tools = count_tokens_approximate(messages_after_tools)
    
    # Calculate deltas (ensure non-negative)
    # Note: original_messages now only contains Base System + Base User
    # messages_after_injection contains Final System (Base+Context+Injections) + Base User
    
    injection_delta = max(0, after_injection - original)
    tools_delta = max(0, after_tools - after_injection)
    
    return {
        "original_tokens": original,
        "injection_added": injection_delta,
        "tools_added": tools_delta,
        "total_middleware": injection_delta + tools_delta,
        "final_total": after_tools
    }


def create_token_breakdown(
    original_messages: List[Dict],
    middleware_tracking: Dict[str, int],
    ollama_prompt_tokens: int,
    ollama_completion_tokens: int
) -> Dict[str, Any]:
    """
    Create comprehensive token breakdown for UI display.
    
    IMPORTANT: Only the user query counts as "original". Everything else
    (system prompts, tool results, etc.) is middleware overhead.
    
    This exposes the FULL token accounting that commercial LLM metrics hide.
    
    Args:
        original_messages: Messages before middleware
        middleware_tracking: Output from track_middleware_tokens()
        ollama_prompt_tokens: Actual count from Ollama (ground truth)
        ollama_completion_tokens: Actual count from Ollama
    
    Returns:
        Comprehensive breakdown dict ready for JSON serialization to UI
    """
    # CRITICAL FIX: Only user messages are "original"
    # System prompts are injection overhead!
    user_tokens = 0
    system_tokens = 0
    
    for msg in original_messages:
        tokens = count_tokens_approximate([msg])
        if msg.get("role") == "user":
            user_tokens += tokens
        elif msg.get("role") == "system":
            system_tokens += tokens  # This is middleware overhead!
    
    # Calculate estimation accuracy
    estimated_prompt = middleware_tracking["final_total"]
    actual_prompt = ollama_prompt_tokens
    
    if actual_prompt > 0:
        accuracy = (min(estimated_prompt, actual_prompt) / max(estimated_prompt, actual_prompt)) * 100
    else:
        accuracy = 0.0
    
    # Calculate middleware multiplier (how much bigger than original)
    # Original = ONLY user query, everything else is overhead
    original_user_only = user_tokens
    total_injection = system_tokens + middleware_tracking["total_middleware"]
    
    if original_user_only > 0:
        middleware_multiplier = (total_injection / original_user_only) * 100
    else:
        middleware_multiplier = 0
    
    # Build comprehensive breakdown
    return {
        "original": {
            "user_tokens": user_tokens,  # Only the user query
            "total_original_tokens": user_tokens  # Only user input
        },
        "injected": {
            "system_prompt_tokens": system_tokens,  # System prompt is overhead!
            "injection_overhead": middleware_tracking["injection_added"],
            "tool_integration_overhead": middleware_tracking["tools_added"],
            "total_injection_tokens": system_tokens + middleware_tracking["total_middleware"]
        },
        "generation": {
            "direct_output_tokens": ollama_completion_tokens
        },
        "verification": {
            "estimated_prompt_tokens": estimated_prompt,
            "ollama_actual_prompt_tokens": actual_prompt,
            "delta": actual_prompt - estimated_prompt,
            "accuracy_percent": round(accuracy, 1)
        },
        "totals": {
            "total_input": actual_prompt,
            "total_output": ollama_completion_tokens,
            "grand_total": actual_prompt + ollama_completion_tokens
        },
        "analysis_notes": [
            f"User query: {user_tokens} tokens (the only 'original' input)",
            f"System prompt: {system_tokens} tokens (middleware overhead)",
            f"Additional middleware: {middleware_tracking['total_middleware']} tokens",
            f"Total overhead: {total_injection} tokens (+{middleware_multiplier:.0f}% vs user query)",
            f"Output generated: {ollama_completion_tokens} tokens",
            f"Grand total: {actual_prompt + ollama_completion_tokens} tokens"
        ]
    }


def estimate_commercial_llm_overhead(query_tokens: int) -> Dict[str, Any]:
    """
    Estimate the hidden token overhead in commercial LLM systems.
    
    This illustrates the "250 token query" problem: what users see as a simple
    query actually consumes 10-100x more tokens due to middleware.
    
    Based on leaked system prompts and documented behavior:
    - System prompts: 5,000-7,000 tokens (repeated on EVERY query)
    - Context window: 10,000+ tokens (conversation history)
    - Tool invocations: 1,000-50,000 tokens (web search results)
    - Memory systems: 1,000-5,000 tokens (personalization)
    - Hidden models: Unknown additional parsing/ranking overhead
    
    Args:
        query_tokens: The visible user query token count
    
    Returns:
        {
            "visible_query": int,
            "system_prompt": int,
            "context": int,
            "tools": int,
            "memory": int,
            "estimated_total": int,
            "multiplier": float
        }
    
    Examples:
        >>> estimate_commercial_llm_overhead(250)  # The "250 token query" myth
        {
            "visible_query": 250,
            "system_prompt": 6000,
            "context": 5000,
            "tools": 10000,
            "memory": 2000,
            "estimated_total": 23250,
            "multiplier": 93.0  # 93x bigger than visible!
        }
    """
    # Conservative estimates based on leaked data
    system_prompt = 6000  # Claude: 5000-7000, GPT-4: 3000-5000
    context_window = 5000  # Conversation history + CoT
    tool_calls = 10000  # 10-50 web pages @ ~200-1000 tokens each
    memory = 2000  # Personalization data
    
    total = query_tokens + system_prompt + context_window + tool_calls + memory
    multiplier = total / max(query_tokens, 1)
    
    return {
        "visible_query": query_tokens,
        "system_prompt": system_prompt,
        "context": context_window,
        "tools": tool_calls,
        "memory": memory,
        "estimated_total": total,
        "multiplier": round(multiplier, 1),
        "warning": "This is a conservative estimate. Real-world overhead can be much higher."
    }


# Utility function for debugging
def print_token_breakdown(breakdown: Dict[str, Any]) -> None:
    """Pretty-print a token breakdown for debugging."""
    print("\n" + "=" * 60)
    print("TOKEN BREAKDOWN")
    print("=" * 60)
    
    print("\n📝 Original Input:")
    print(f"  System: {breakdown['original']['system_tokens']} tokens")
    print(f"  User:   {breakdown['original']['user_tokens']} tokens")
    print(f"  Total:  {breakdown['original']['total_original_tokens']} tokens")
    
    print("\n⚙️ Middleware Overhead:")
    print(f"  Injection:        {breakdown['injected']['injection_overhead']} tokens")
    print(f"  Tool Integration: {breakdown['injected']['tool_integration_overhead']} tokens")
    print(f"  Total Overhead:   {breakdown['injected']['total_injection_tokens']} tokens")
    
    print("\n🤖 Generation:")
    print(f"  Output: {breakdown['generation']['direct_output_tokens']} tokens")
    
    print("\n✅ Verification:")
    print(f"  Estimated: {breakdown['verification']['estimated_prompt_tokens']} tokens")
    print(f"  Actual:    {breakdown['verification']['ollama_actual_prompt_tokens']} tokens")
    print(f"  Accuracy:  {breakdown['verification']['accuracy_percent']}%")
    
    print("\n📊 Totals:")
    print(f"  Input:  {breakdown['totals']['total_input']} tokens")
    print(f"  Output: {breakdown['totals']['total_output']} tokens")
    print(f"  Grand:  {breakdown['totals']['grand_total']} tokens")
    
    print("\n💡 Analysis:")
    for note in breakdown['analysis_notes']:
        print(f"  • {note}")
    
    print("=" * 60 + "\n")
