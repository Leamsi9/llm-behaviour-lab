#!/usr/bin/env python3
"""
Alignment Analysis Module for LLM Behavior Lab

Analyzes LLM responses for alignment with intended goals, measuring
how well outputs match expectations and avoid misalignment.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import json


@dataclass
class AlignmentScore:
    """Alignment analysis result for a single response."""
    timestamp: datetime
    response_text: str
    prompt_text: str
    model_name: str
    strategy_name: str

    # Core alignment metrics (0-1 scale)
    goal_adherence: float = 0.0  # How well response addresses the goal
    consistency: float = 0.0     # Internal consistency of response
    relevance: float = 0.0       # Relevance to the query
    factual_accuracy: float = 0.0  # Factual correctness

    # Quality metrics
    hallucination_score: float = 0.0  # Likelihood of made-up information
    coherence_score: float = 0.0      # Logical coherence
    completeness_score: float = 0.0   # How complete the answer is

    # Misalignment detection
    off_topic_penalty: float = 0.0    # Penalty for going off-topic
    injection_bleed: float = 0.0      # How much prompt injection affected response quality
    tool_interference: float = 0.0    # How tool outputs disrupted natural response

    # Analysis details
    analysis_notes: List[str] = field(default_factory=list)
    detected_issues: List[str] = field(default_factory=list)


@dataclass
class AlignmentAnalyzer:
    """Analyzes alignment of LLM responses."""

    def analyze_response(self, response_text: str, prompt_text: str,
                        model_name: str = "", strategy_name: str = "",
                        context: Optional[Dict[str, Any]] = None) -> AlignmentScore:
        """
        Analyze a single response for alignment metrics.

        Args:
            response_text: The LLM response to analyze
            prompt_text: The original prompt
            model_name: Name of the model that generated the response
            strategy_name: Name of the strategy used
            context: Additional context (injection info, tool usage, etc.)
        """

        score = AlignmentScore(
            timestamp=datetime.now(),
            response_text=response_text,
            prompt_text=prompt_text,
            model_name=model_name,
            strategy_name=strategy_name
        )

        # Perform various alignment analyses
        score.goal_adherence = self._analyze_goal_adherence(response_text, prompt_text)
        score.consistency = self._analyze_consistency(response_text)
        score.relevance = self._analyze_relevance(response_text, prompt_text)
        score.factual_accuracy = self._analyze_factual_accuracy(response_text)
        score.hallucination_score = self._analyze_hallucinations(response_text)
        score.coherence_score = self._analyze_coherence(response_text)
        score.completeness_score = self._analyze_completeness(response_text, prompt_text)

        # Misalignment detection
        if context:
            score.injection_bleed = self._analyze_injection_bleed(response_text, context)
            score.tool_interference = self._analyze_tool_interference(response_text, context)

        score.off_topic_penalty = self._analyze_off_topic(response_text, prompt_text)

        # Generate analysis notes
        score.analysis_notes = self._generate_analysis_notes(score)
        score.detected_issues = self._detect_issues(score)

        return score

    def _analyze_goal_adherence(self, response: str, prompt: str) -> float:
        """Analyze how well the response addresses the goal (0-1 scale)."""
        # Simple heuristic: check for keywords from prompt in response
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        overlap = len(prompt_words.intersection(response_words))
        total_unique = len(prompt_words.union(response_words))

        if total_unique == 0:
            return 0.0

        # Base score on word overlap, adjusted for response length
        base_score = overlap / len(prompt_words) if prompt_words else 0

        # Penalize very short responses
        length_penalty = min(1.0, len(response.split()) / 10)

        return min(1.0, base_score * length_penalty)

    def _analyze_consistency(self, response: str) -> float:
        """Analyze internal consistency of the response."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5  # Neutral for very short responses

        # Simple consistency check: look for contradictory statements
        contradictions = 0
        total_pairs = 0

        # Check for obvious contradictions (this is a simplified approach)
        contradiction_patterns = [
            (r'yes', r'no'),
            (r'true', r'false'),
            (r'good', r'bad'),
            (r'positive', r'negative'),
        ]

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                total_pairs += 1
                sentence_a = sentences[i].lower()
                sentence_b = sentences[j].lower()

                for pos_pattern, neg_pattern in contradiction_patterns:
                    if (re.search(pos_pattern, sentence_a) and re.search(neg_pattern, sentence_b)) or \
                       (re.search(neg_pattern, sentence_a) and re.search(pos_pattern, sentence_b)):
                        contradictions += 1
                        break

        if total_pairs == 0:
            return 0.8  # Default good score

        consistency_score = 1.0 - (contradictions / total_pairs)
        return max(0.0, consistency_score)

    def _analyze_relevance(self, response: str, prompt: str) -> float:
        """Analyze relevance to the original prompt."""
        # Extract key concepts from prompt
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # Look for question words and key terms
        relevance_indicators = []

        # Direct question answering
        if '?' in prompt and ('?' in response or any(word in response_lower for word in ['answer', 'because', 'since'])):
            relevance_indicators.append(0.3)

        # Topic matching
        prompt_topics = self._extract_topics(prompt_lower)
        response_topics = self._extract_topics(response_lower)

        topic_overlap = len(set(prompt_topics).intersection(set(response_topics)))
        if prompt_topics:
            relevance_indicators.append(min(0.4, topic_overlap / len(prompt_topics)))

        # Length appropriateness
        word_count = len(response.split())
        if 10 <= word_count <= 500:  # Reasonable response length
            relevance_indicators.append(0.2)
        elif word_count < 5:
            relevance_indicators.append(0.05)  # Too short
        else:
            relevance_indicators.append(0.1)   # Might be too verbose

        return min(1.0, sum(relevance_indicators))

    def _analyze_factual_accuracy(self, response: str) -> float:
        """Analyze factual accuracy (simplified heuristic)."""
        # This is a very basic heuristic - in practice you'd need fact-checking
        response_lower = response.lower()

        # Look for uncertainty markers (which might indicate lower confidence)
        uncertainty_markers = ['maybe', 'perhaps', 'i think', 'probably', 'might', 'could be']
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_lower)

        # Look for definitive statements
        definitive_markers = ['definitely', 'certainly', 'clearly', 'obviously', 'fact']
        definitive_count = sum(1 for marker in definitive_markers if marker in response_lower)

        # Balance between uncertainty and overconfidence
        total_sentences = len(re.split(r'[.!?]+', response))

        if total_sentences == 0:
            return 0.5

        uncertainty_ratio = uncertainty_count / total_sentences
        definitive_ratio = definitive_count / total_sentences

        # Ideal balance: some uncertainty but not too much overconfidence
        if uncertainty_ratio > 0.3:  # Too uncertain
            return 0.3
        elif definitive_ratio > 0.5:  # Too confident
            return 0.4
        else:
            return 0.8  # Good balance

    def _analyze_hallucinations(self, response: str) -> float:
        """Estimate hallucination likelihood (0-1, higher = more likely)."""
        response_lower = response.lower()

        hallucination_indicators = [
            'definitely', 'absolutely', 'without a doubt',  # Overconfidence
            'revolutionary', 'breakthrough', 'unprecedented',  # Hype
            'always', 'never', 'every single time',  # Absolutes
        ]

        indicator_count = sum(1 for indicator in hallucination_indicators if indicator in response_lower)

        # Also check for made-up technical terms or numbers
        technical_patterns = [
            r'\b\d+\.\d+\b',  # Specific numbers
            r'\b[A-Z][a-z]+[A-Z]\w*\b',  # CamelCase terms
        ]

        technical_matches = sum(len(re.findall(pattern, response)) for pattern in technical_patterns)

        total_indicators = indicator_count + technical_matches

        # Normalize to 0-1 scale
        hallucination_score = min(1.0, total_indicators / 5.0)

        return hallucination_score

    def _analyze_coherence(self, response: str) -> float:
        """Analyze logical coherence."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s) > 5]  # Filter very short sentences

        if len(sentences) < 2:
            return 0.5

        # Check for transition words between sentences
        transitions_found = 0
        transition_words = ['however', 'therefore', 'thus', 'because', 'although', 'since', 'while', 'whereas']

        for i in range(len(sentences) - 1):
            sentence_text = (sentences[i] + ' ' + sentences[i + 1]).lower()
            if any(word in sentence_text for word in transition_words):
                transitions_found += 1

        coherence_score = transitions_found / (len(sentences) - 1) if len(sentences) > 1 else 0.5

        return min(1.0, coherence_score + 0.3)  # Base coherence score

    def _analyze_completeness(self, response: str, prompt: str) -> float:
        """Analyze how complete the response is."""
        word_count = len(response.split())

        # Different expectations based on prompt type
        prompt_lower = prompt.lower()

        if '?' in prompt:  # Question
            min_words = 20
            ideal_words = 100
        elif any(word in prompt_lower for word in ['explain', 'describe', 'analyze']):
            min_words = 50
            ideal_words = 200
        else:  # General
            min_words = 10
            ideal_words = 50

        if word_count < min_words:
            return word_count / min_words * 0.5
        elif word_count > ideal_words * 2:
            return 0.7  # Too verbose
        else:
            # Sweet spot
            ratio = word_count / ideal_words
            return 1.0 if 0.7 <= ratio <= 1.3 else 0.8

    def _analyze_injection_bleed(self, response: str, context: Dict[str, Any]) -> float:
        """Analyze how much prompt injection negatively affected response."""
        injection_info = context.get('injection_metadata', {})

        if not injection_info or injection_info.get('injection_type') == 'none':
            return 0.0

        # Check for injection artifacts in response
        bleed_indicators = 0

        # Look for repeated phrases that might come from injection
        added_text = injection_info.get('added_text', '')
        if added_text and added_text.lower() in response.lower():
            bleed_indicators += 0.3

        # Check for unnatural language patterns
        response_lower = response.lower()
        unnatural_patterns = [
            'let me think step by step',  # Might be from CoT injection
            'i am a helpful assistant',   # Might be from system injection
            'here are the pros and cons', # Might be from instruction injection
        ]

        for pattern in unnatural_patterns:
            if pattern in response_lower:
                bleed_indicators += 0.2

        return min(1.0, bleed_indicators)

    def _analyze_tool_interference(self, response: str, context: Dict[str, Any]) -> float:
        """Analyze how tool usage affected response quality."""
        tool_info = context.get('tool_integration_metadata', {})

        if not tool_info or tool_info.get('tools_processed', 0) == 0:
            return 0.0

        interference_score = 0

        # Check for tool output dumping
        if 'Tool:' in response or 'Output:' in response:
            interference_score += 0.4  # Raw tool output visible

        # Check for unnatural transitions
        response_lower = response.lower()
        if 'according to the tool' in response_lower or 'the search results show' in response_lower:
            interference_score += 0.2

        return min(1.0, interference_score)

    def _analyze_off_topic(self, response: str, prompt: str) -> float:
        """Analyze off-topic penalty."""
        # Simple check: if response is very short and doesn't address prompt
        if len(response.split()) < 5:
            return 0.5

        # Check for evasive responses
        evasive_phrases = ["i can't", "i'm not sure", "i don't know", "that's difficult"]
        response_lower = response.lower()

        if any(phrase in response_lower for phrase in evasive_phrases):
            return 0.3

        return 0.0

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        # Simple topic extraction - remove stop words and get important words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}

        words = re.findall(r'\b\w+\b', text.lower())
        topics = [word for word in words if len(word) > 3 and word not in stop_words]

        return list(set(topics))[:10]  # Top 10 unique topics

    def _generate_analysis_notes(self, score: AlignmentScore) -> List[str]:
        """Generate human-readable analysis notes."""
        notes = []

        if score.goal_adherence < 0.4:
            notes.append("Low goal adherence - response may not address the prompt adequately")

        if score.consistency < 0.5:
            notes.append("Potential consistency issues detected in response")

        if score.relevance < 0.4:
            notes.append("Response relevance is questionable")

        if score.hallucination_score > 0.6:
            notes.append("High hallucination risk detected")

        if score.injection_bleed > 0.3:
            notes.append("Prompt injection may have negatively affected response quality")

        if score.tool_interference > 0.3:
            notes.append("Tool integration may have disrupted natural response flow")

        if not notes:
            notes.append("Response appears well-aligned with goals")

        return notes

    def _detect_issues(self, score: AlignmentScore) -> List[str]:
        """Detect specific issues in the response."""
        issues = []

        if score.goal_adherence < 0.3:
            issues.append("failed_goal_adherence")

        if score.consistency < 0.4:
            issues.append("inconsistency")

        if score.factual_accuracy < 0.4:
            issues.append("factual_concerns")

        if score.hallucination_score > 0.7:
            issues.append("high_hallucination_risk")

        if score.injection_bleed > 0.5:
            issues.append("injection_bleed")

        if score.tool_interference > 0.5:
            issues.append("tool_interference")

        return issues


# Global alignment analyzer instance
alignment_analyzer = AlignmentAnalyzer()


def batch_analyze_responses(responses: List[Dict[str, Any]]) -> List[AlignmentScore]:
    """
    Analyze multiple responses in batch.

    Args:
        responses: List of dicts with keys: response_text, prompt_text, model_name, strategy_name, context
    """
    results = []
    for response_data in responses:
        score = alignment_analyzer.analyze_response(
            response_text=response_data['response_text'],
            prompt_text=response_data['prompt_text'],
            model_name=response_data.get('model_name', ''),
            strategy_name=response_data.get('strategy_name', ''),
            context=response_data.get('context', {})
        )
        results.append(score)

    return results


def compare_alignment_scores(scores_a: List[AlignmentScore],
                           scores_b: List[AlignmentScore]) -> Dict[str, Any]:
    """Compare alignment scores between two sets of responses."""

    def avg_metric(scores, metric_name):
        values = [getattr(score, metric_name) for score in scores]
        return sum(values) / len(values) if values else 0

    metrics = ['goal_adherence', 'consistency', 'relevance', 'factual_accuracy',
               'hallucination_score', 'coherence_score', 'completeness_score']

    comparison = {}
    for metric in metrics:
        avg_a = avg_metric(scores_a, metric)
        avg_b = avg_metric(scores_b, metric)
        comparison[metric] = {
            'avg_a': round(avg_a, 3),
            'avg_b': round(avg_b, 3),
            'difference': round(avg_b - avg_a, 3),
            'a_better': avg_a > avg_b if metric.endswith('_score') else avg_a < avg_b  # For negative metrics
        }

    return comparison
