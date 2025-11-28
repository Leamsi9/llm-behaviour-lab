#!/usr/bin/env python3
"""
Unit tests for token_utils.py

Tests the critical token counting and middleware tracking functionality.
"""

import pytest
from token_utils import (
    count_tokens_approximate,
    track_middleware_tokens,
    create_token_breakdown,
    estimate_commercial_llm_overhead
)


class TestTokenCountingApproximate:
    """Test the basic token counting approximation."""
    
    def test_empty_messages(self):
        """Empty messages should return 0 tokens."""
        assert count_tokens_approximate([]) == 0
    
    def test_single_short_message(self):
        """Short message should have reasonable token count."""
        messages = [{"role": "user", "content": "Hello"}]
        tokens = count_tokens_approximate(messages)
        # "Hello" = 5 chars / 4 = 1 token + 4 overhead = 5 tokens
        assert tokens >= 4  # At least message overhead
        assert tokens <= 10  # Not too high
    
    def test_multi_message_conversation(self):
        """Multi-turn conversation should accumulate tokens."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."}
        ]
        tokens = count_tokens_approximate(messages)
        # 3 messages * 4 overhead = 12 base
        # Plus char content ~20 tokens
        assert tokens >= 12
        assert tokens <= 50
    
    def test_long_content(self):
        """Long content should scale appropriately."""
        # 4000 characters should be ~1000 tokens + overhead
        long_text = "word " * 800  # 4000 chars
        messages = [{"role": "user", "content": long_text}]
        tokens = count_tokens_approximate(messages)
        assert tokens >= 1000
        assert tokens <= 1100
    
    def test_message_structure_overhead(self):
        """More messages should add more overhead."""
        single = count_tokens_approximate([{"role": "user", "content": "hi"}])
        double = count_tokens_approximate([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi"}
        ])
        # Double should have ~4 more tokens (one message overhead)
        assert double >= single + 3
        assert double <= single + 5


class TestMiddlewareTracking:
    """Test middleware token tracking."""
    
    def test_no_middleware(self):
        """No middleware should show zero overhead."""
        original = [{"role": "user", "content": "test"}]
        result = track_middleware_tokens(original, original, original)
        
        assert result["original_tokens"] > 0
        assert result["injection_added"] == 0
        assert result["tools_added"] == 0
        assert result["total_middleware"] == 0
        assert result["final_total"] == result["original_tokens"]
    
    def test_injection_only(self):
        """Injection should add tokens."""
        original = [{"role": "user", "content": "test"}]
        after_injection = [
            {"role": "system", "content": "You are helpful." * 100},  # Big system prompt
            {"role": "user", "content": "test"}
        ]
        result = track_middleware_tokens(original, after_injection, after_injection)
        
        assert result["original_tokens"] > 0
        assert result["injection_added"] > 0
        assert result["tools_added"] == 0
        assert result["total_middleware"] == result["injection_added"]
    
    def test_tools_only(self):
        """Tools should add tokens."""
        original = [{"role": "user", "content": "test"}]
        after_tools = original + [
            {"role": "system", "content": "Tool results: " + "data " * 500}
        ]
        result = track_middleware_tokens(original, original, after_tools)
        
        assert result["original_tokens"] > 0
        assert result["injection_added"] == 0
        assert result["tools_added"] > 0
        assert result["total_middleware"] == result["tools_added"]
    
    def test_both_middleware_types(self):
        """Both injection and tools should accumulate."""
        original = [{"role": "user", "content": "test"}]
        
        after_injection = [
            {"role": "system", "content": "You are helpful." * 100},
            {"role": "user", "content": "test"}
        ]
        
        after_tools = after_injection + [
            {"role": "system", "content": "Tool: " + "result " * 300}
        ]
        
        result = track_middleware_tokens(original, after_injection, after_tools)
        
        assert result["injection_added"] > 0
        assert result["tools_added"] > 0
        assert result["total_middleware"] == result["injection_added"] + result["tools_added"]
        assert result["final_total"] > result["original_tokens"]
    
    def test_massive_middleware_overhead(self):
        """Simulate commercial LLM-scale middleware."""
        # Simple 250 token query
        original = [{"role": "user", "content": "What is AI?" * 50}]  # ~250 tokens
        
        # Add realistic commercial LLM overhead
        after_injection = [
            {"role": "system", "content": "x" * 24000},  # ~6000 token system prompt
            {"role": "user", "content": "What is AI?" * 50}
        ]
        
        after_tools = after_injection + [
            {"role": "system", "content": "x" * 40000}  # ~10000 token tool results
        ]
        
        result = track_middleware_tokens(original, after_injection, after_tools)
        
        # Middleware should be 10-100x the original
        multiplier = result["total_middleware"] / max(result["original_tokens"], 1)
        assert multiplier >= 10, f"Multiplier {multiplier} should be >= 10x (commercial LLM reality)"


class TestTokenBreakdown:
    """Test comprehensive token breakdown generation."""
    
    def test_breakdown_structure(self):
        """Breakdown should have all required fields."""
        original = [{"role": "user", "content": "test"}]
        tracking = track_middleware_tokens(original, original, original)
        
        breakdown = create_token_breakdown(
            original_messages=original,
            middleware_tracking=tracking,
            ollama_prompt_tokens=10,
            ollama_completion_tokens=50
        )
        
        # Check required top-level keys
        assert "original" in breakdown
        assert "injected" in breakdown
        assert "generation" in breakdown
        assert "verification" in breakdown
        assert "totals" in breakdown
        assert "analysis_notes" in breakdown
    
    def test_accuracy_calculation(self):
        """Accuracy should be calculated correctly."""
        original = [{"role": "user", "content": "test"}]
        tracking = track_middleware_tokens(original, original, original)
        
        # Simulate good estimation
        estimated = tracking["final_total"]
        actual = estimated  # Exact match
        
        breakdown = create_token_breakdown(
            original_messages=original,
            middleware_tracking=tracking,
            ollama_prompt_tokens=actual,
            ollama_completion_tokens=50
        )
        
        # Should be ~100% accurate
        assert breakdown["verification"]["accuracy_percent"] >= 99
    
    def test_analysis_notes_present(self):
        """Should generate helpful analysis notes."""
        original = [{"role": "user", "content": "test"}]
        tracking = track_middleware_tokens(original, original, original)
        
        breakdown = create_token_breakdown(
            original_messages=original,
            middleware_tracking=tracking,
            ollama_prompt_tokens=10,
            ollama_completion_tokens=50
        )
        
        notes = breakdown["analysis_notes"]
        assert len(notes) >= 3
        assert any("Original" in note for note in notes)
        assert any("Middleware" in note for note in notes)


class TestCommercialLLMEstimation:
    """Test commercial LLM overhead estimation."""
    
    def test_250_token_query_myth(self):
        """The famous '250 token query' should reveal massive hidden overhead."""
        result = estimate_commercial_llm_overhead(250)
        
        assert result["visible_query"] == 250
        assert result["system_prompt"] > 0
        assert result["tools"] > 0
        assert result["estimated_total"] > 250
        
        # Should be at least 10x bigger
        assert result["multiplier"] >= 10
    
    def test_includes_all_components(self):
        """Should include all middleware components."""
        result = estimate_commercial_llm_overhead(100)
        
        assert "system_prompt" in result
        assert "context" in result
        assert "tools" in result
        assert "memory" in result
        assert "multiplier" in result
        assert "warning" in result


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
