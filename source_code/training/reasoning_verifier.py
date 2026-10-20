# Copyright 2025 The Qwen Team and Alibaba Group.
# Apache License, Version 2.0 (the "License");
#
# Module: Test-Time Compute & Reasoning Verification Engine
# Implements Process Reward Modeling (PRM), step-by-step reasoning trajectory
# verification, and Best-of-N test-time compute scaling for Qwen3 reasoning traces.

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

class StepLevelPRM(nn.Module):
    """
    Process Reward Model (PRM) scoring head.
    Evaluates intermediate reasoning steps by projecting final token states
    of each reasoning step into a correctness confidence score in [0, 1].
    """
    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, step_representations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_representations: (B, num_steps, hidden_size)
        Returns:
            step_rewards: (B, num_steps) scores in [0, 1]
        """
        return self.scorer(step_representations).squeeze(-1)


class ReasoningVerifier:
    """
    Evaluates and verifies mathematical and chain-of-thought (CoT) reasoning traces.
    Combines:
    1. Structural formatting reward (clean <think> tags, non-empty, proper closure).
    2. Length and step granularity penalties.
    3. Trajectory aggregation (geometric mean of step scores).
    """
    def __init__(
        self,
        think_open_tag: str = "<think>",
        think_close_tag: str = "</think>",
        min_steps: int = 2,
        max_repetition_ngram: int = 4
    ):
        self.think_open_tag = think_open_tag
        self.think_close_tag = think_close_tag
        self.min_steps = min_steps
        self.max_repetition_ngram = max_repetition_ngram

    def extract_reasoning_and_answer(self, text: str) -> Dict[str, Any]:
        """Parses reasoning body inside <think>...</think> and final answer."""
        has_open = self.think_open_tag in text
        has_close = self.think_close_tag in text
        
        if has_open and has_close:
            pattern = re.escape(self.think_open_tag) + r"(.*?)" + re.escape(self.think_close_tag)
            match = re.search(pattern, text, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                answer = text[match.end():].strip()
                valid_format = True
            else:
                reasoning = ""
                answer = text.strip()
                valid_format = False
        else:
            reasoning = text.strip()
            answer = ""
            valid_format = False
            
        # Segment into discrete steps by double newline or punctuation
        steps = [s.strip() for s in re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z0-9])", reasoning) if s.strip()]
        
        return {
            "valid_format": valid_format,
            "reasoning": reasoning,
            "answer": answer,
            "steps": steps,
            "num_steps": len(steps)
        }

    def compute_format_reward(self, text: str) -> float:
        """Computes rule-based format reward in [-1.0, 1.0]."""
        parsed = self.extract_reasoning_and_answer(text)
        if not parsed["valid_format"]:
            return -1.0
            
        reward = 0.5
        # Reward having structured steps
        if parsed["num_steps"] >= self.min_steps:
            reward += 0.3
            
        # Reward having a clean final response
        if len(parsed["answer"]) > 0:
            reward += 0.2
            
        return reward

    def detect_reasoning_loops(self, text: str) -> bool:
        """Detects pathological repetitive reasoning degeneration (n-gram loops)."""
        words = text.split()
        if len(words) < 16:
            return False
            
        n = self.max_repetition_ngram
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ratio = len(set(ngrams)) / max(len(ngrams), 1)
        return unique_ratio < 0.45


class BestOfNVerifier:
    """
    Best-of-N Candidate Search Engine with PRM & Structural Verification.
    Scores N candidate completions and selects the highest-confidence reasoning trace.
    """
    def __init__(self, prm_model: Optional[StepLevelPRM] = None):
        self.verifier = ReasoningVerifier()
        self.prm_model = prm_model

    def select_best(
        self,
        candidates: List[str],
        candidate_embeddings: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Ranks candidates and returns the best trajectory along with diagnostic scores.
        """
        scores = []
        parsed_candidates = []
        
        for idx, cand in enumerate(candidates):
            parsed = self.verifier.extract_reasoning_and_answer(cand)
            parsed_candidates.append(parsed)
            
            fmt_reward = self.verifier.compute_format_reward(cand)
            has_loop = self.verifier.detect_reasoning_loops(cand)
            
            prm_score = 1.0
            if self.prm_model is not None and candidate_embeddings is not None:
                with torch.no_grad():
                    emb = candidate_embeddings[idx].unsqueeze(0) # (1, num_steps, H)
                    step_probs = self.prm_model(emb).squeeze(0) # (num_steps,)
                    # Trajectory product score (all steps must be sound)
                    prm_score = float(torch.prod(step_probs).item())
                    
            total_score = fmt_reward * 0.4 + (0.0 if has_loop else 0.3) + prm_score * 0.3
            scores.append(total_score)
            
        best_idx = int(torch.tensor(scores).argmax().item())
        return {
            "best_index": best_idx,
            "best_candidate": candidates[best_idx],
            "best_score": scores[best_idx],
            "all_scores": scores,
            "metadata": parsed_candidates[best_idx]
        }
