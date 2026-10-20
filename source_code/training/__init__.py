# Copyright 2025 The Qwen Team and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Módulos de entrenamiento, alineación y optimización de razonamiento para Qwen3."""

from .reasoning_loss import GRPOLoss, DPOLoss, ReasoningFormatReward
from .reasoning_verifier import StepLevelPRM, ReasoningVerifier, BestOfNVerifier
from .mcts_reasoning import MCTSNode, Qwen3MCTSSearchEngine
from .sequence_packing import SequencePacker, PackedSequenceBatch
from .activation_checkpointing import apply_activation_checkpointing, CheckpointedBlockWrapper

__all__ = [
    "GRPOLoss",
    "DPOLoss",
    "ReasoningFormatReward",
    "StepLevelPRM",
    "ReasoningVerifier",
    "BestOfNVerifier",
    "MCTSNode",
    "Qwen3MCTSSearchEngine",
    "SequencePacker",
    "PackedSequenceBatch",
    "apply_activation_checkpointing",
    "CheckpointedBlockWrapper"
]
