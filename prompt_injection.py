#!/usr/bin/env python3
"""
Prompt Injection System for LLM Behavior Lab

Implements various prompt injection techniques to test how different
modifications affect energy consumption and alignment.
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import re


class InjectionType(Enum):
    """Types of prompt injections."""
    NONE = "none"
    SYSTEM_MODIFICATION = "system_modification"
    USER_AUGMENTATION = "user_augmentation"
    CONTEXT_INJECTION = "context_injection"
    TOOL_INTEGRATION = "tool_integration"
    COT_INSTRUCTION = "cot_instruction"


@dataclass
class InjectionConfig:
    """Configuration for prompt injection."""
    injection_type: InjectionType = InjectionType.NONE
    injection_params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class PromptInjector(ABC):
    """Abstract base class for prompt injectors."""

    @abstractmethod
    def inject(self, system_prompt: str, user_prompt: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Inject modifications into prompts.

        Returns dict with:
        - modified_system: Modified system prompt
        - modified_user: Modified user prompt
        - injection_metadata: Dict with injection details
        """
        pass

    @property
    @abstractmethod
    def injection_type(self) -> InjectionType:
        pass


class NoInjectionInjector(PromptInjector):
    """Baseline injector with no modifications."""

    @property
    def injection_type(self) -> InjectionType:
        return InjectionType.NONE

    def inject(self, system_prompt: str, user_prompt: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "modified_system": system_prompt,
            "modified_user": user_prompt,
            "injection_metadata": {
                "injection_type": "none",
                "original_system_tokens": len(system_prompt.split()),
                "original_user_tokens": len(user_prompt.split()),
            }
        }


class SystemModificationInjector(PromptInjector):
    """Modifies the system prompt with additional instructions."""

    @property
    def injection_type(self) -> InjectionType:
        return InjectionType.SYSTEM_MODIFICATION

    def inject(self, system_prompt: str, user_prompt: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        config = context.get("injection_config", {}) if context else {}
        modification_type = config.get("modification_type", "concise")

        modifications = {
            "concise": "\n\nBe extremely concise. Minimize tokens while maintaining accuracy.",
            "detailed": "\n\nProvide comprehensive, detailed responses with thorough explanations.",
            "creative": "\n\nBe creative and original in your responses. Think outside the box.",
            "factual": "\n\nPrioritize factual accuracy. Cite sources when possible. Avoid speculation.",
            "safe": "\n\nEnsure responses are safe, ethical, and appropriate. Avoid harmful content.",
        }

        modification = modifications.get(modification_type, modifications["concise"])
        modified_system = system_prompt + modification

        return {
            "modified_system": modified_system,
            "modified_user": user_prompt,
            "injection_metadata": {
                "injection_type": "system_modification",
                "modification_type": modification_type,
                "added_text": modification,
                "original_system_tokens": len(system_prompt.split()),
                "modified_system_tokens": len(modified_system.split()),
                "injection_overhead": len(modification.split()),
            }
        }


class UserAugmentationInjector(PromptInjector):
    """Augments the user prompt with additional context or instructions."""

    @property
    def injection_type(self) -> InjectionType:
        return InjectionType.USER_AUGMENTATION

    def inject(self, system_prompt: str, user_prompt: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        config = context.get("injection_config", {}) if context else {}
        augmentation_type = config.get("augmentation_type", "step_by_step")

        augmentations = {
            "step_by_step": "\n\nPlease explain your answer step by step.",
            "evidence_based": "\n\nSupport your answer with evidence and reasoning.",
            "compare_options": "\n\nCompare different approaches and explain trade-offs.",
            "pros_cons": "\n\nList the pros and cons of your recommendation.",
            "actionable": "\n\nProvide actionable, practical advice.",
        }

        augmentation = augmentations.get(augmentation_type, augmentations["step_by_step"])

        # Insert augmentation at the end of user prompt
        modified_user = user_prompt + augmentation

        return {
            "modified_system": system_prompt,
            "modified_user": modified_user,
            "injection_metadata": {
                "injection_type": "user_augmentation",
                "augmentation_type": augmentation_type,
                "added_text": augmentation,
                "original_user_tokens": len(user_prompt.split()),
                "modified_user_tokens": len(modified_user.split()),
                "injection_overhead": len(augmentation.split()),
            }
        }


class ContextInjectionInjector(PromptInjector):
    """Injects additional context into the prompt."""

    @property
    def injection_type(self) -> InjectionType:
        return InjectionType.CONTEXT_INJECTION

    def inject(self, system_prompt: str, user_prompt: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        config = context.get("injection_config", {}) if context else {}
        context_text = config.get("context_text", "")
        injection_position = config.get("position", "before_user")  # before_user, after_user, in_system

        if not context_text:
            return {
                "modified_system": system_prompt,
                "modified_user": user_prompt,
                "injection_metadata": {
                    "injection_type": "context_injection",
                    "context_length": 0,
                    "no_context_provided": True,
                }
            }

        if injection_position == "before_user":
            modified_user = f"{context_text}\n\n{user_prompt}"
            modified_system = system_prompt
        elif injection_position == "after_user":
            modified_user = f"{user_prompt}\n\n{context_text}"
            modified_system = system_prompt
        elif injection_position == "in_system":
            modified_system = f"{system_prompt}\n\nContext: {context_text}"
            modified_user = user_prompt
        else:
            modified_system = system_prompt
            modified_user = user_prompt

        return {
            "modified_system": modified_system,
            "modified_user": modified_user,
            "injection_metadata": {
                "injection_type": "context_injection",
                "context_length": len(context_text.split()),
                "injection_position": injection_position,
                "original_user_tokens": len(user_prompt.split()),
                "modified_user_tokens": len(modified_user.split()),
                "injection_overhead": len(context_text.split()),
            }
        }


class CoTInstructionInjector(PromptInjector):
    """Adds Chain of Thought instructions to prompts."""

    @property
    def injection_type(self) -> InjectionType:
        return InjectionType.COT_INSTRUCTION

    def inject(self, system_prompt: str, user_prompt: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        config = context.get("injection_config", {}) if context else {}
        cot_style = config.get("cot_style", "explicit")

        cot_instructions = {
            "explicit": "\n\nLet's think step by step:",
            "structured": "\n\nBreak this down systematically:\n1. Analyze the problem\n2. Consider options\n3. Provide reasoning\n4. Give conclusion",
            "minimal": "\n\nThink step by step.",
            "detailed": "\n\nPlease reason through this problem carefully, showing your work and explaining each step of your thinking process.",
        }

        cot_text = cot_instructions.get(cot_style, cot_instructions["explicit"])
        modified_user = user_prompt + cot_text

        return {
            "modified_system": system_prompt,
            "modified_user": modified_user,
            "injection_metadata": {
                "injection_type": "cot_instruction",
                "cot_style": cot_style,
                "added_text": cot_text,
                "original_user_tokens": len(user_prompt.split()),
                "modified_user_tokens": len(modified_user.split()),
                "injection_overhead": len(cot_text.split()),
            }
        }


class InjectionManager:
    """Manages different prompt injection strategies."""

    def __init__(self):
        self.injectors = {
            InjectionType.NONE: NoInjectionInjector(),
            InjectionType.SYSTEM_MODIFICATION: SystemModificationInjector(),
            InjectionType.USER_AUGMENTATION: UserAugmentationInjector(),
            InjectionType.CONTEXT_INJECTION: ContextInjectionInjector(),
            InjectionType.COT_INSTRUCTION: CoTInstructionInjector(),
        }

    def get_injector(self, injection_type: InjectionType) -> PromptInjector:
        """Get injector for a specific type."""
        return self.injectors.get(injection_type, self.injectors[InjectionType.NONE])

    def apply_injection(self, injection_config: InjectionConfig,
                       system_prompt: str, user_prompt: str,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply injection based on configuration."""

        if not injection_config.enabled:
            return self.injectors[InjectionType.NONE].inject(system_prompt, user_prompt, context)

        injector = self.get_injector(injection_config.injection_type)

        # Merge injection params into context
        injection_context = context or {}
        injection_context["injection_config"] = injection_config.injection_params

        result = injector.inject(system_prompt, user_prompt, injection_context)

        # Add config metadata
        result["injection_metadata"].update({
            "injection_enabled": True,
            "config_params": injection_config.injection_params,
        })

        return result

    def get_available_injections(self) -> List[Dict[str, Any]]:
        """Get list of available injection types with descriptions."""
        return [
            {
                "type": InjectionType.NONE.value,
                "name": "No Injection",
                "description": "Baseline with no modifications",
                "energy_impact": "baseline",
                "alignment_impact": "baseline",
            },
            {
                "type": InjectionType.SYSTEM_MODIFICATION.value,
                "name": "System Modification",
                "description": "Modify system prompt with behavioral instructions",
                "energy_impact": "low (+10-20%)",
                "alignment_impact": "variable",
            },
            {
                "type": InjectionType.USER_AUGMENTATION.value,
                "name": "User Augmentation",
                "description": "Add instructions to user prompt",
                "energy_impact": "low (+5-15%)",
                "alignment_impact": "high (direct instruction)",
            },
            {
                "type": InjectionType.CONTEXT_INJECTION.value,
                "name": "Context Injection",
                "description": "Inject additional context or information",
                "energy_impact": "variable (+0-100%)",
                "alignment_impact": "high (additional context)",
            },
            {
                "type": InjectionType.COT_INSTRUCTION.value,
                "name": "CoT Instructions",
                "description": "Add chain-of-thought prompting",
                "energy_impact": "medium (+20-40%)",
                "alignment_impact": "high (reasoning structure)",
            },
        ]


# Global injection manager instance
injection_manager = InjectionManager()
