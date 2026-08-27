"""Shared runner and algorithm configuration for AMP tasks."""

# pyright: reportIncompatibleVariableOverride=false

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "mjlab_playground.amp.rl.amp_ppo:AmpPPO"
  amp_cfg: dict = field(default_factory=dict)
  symmetry_cfg: dict | None = None


@dataclass
class AmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  class_name: str = "mjlab_playground.amp.rl.runner:AmpOnPolicyRunner"
  algorithm: AmpPpoAlgorithmCfg = field(default_factory=AmpPpoAlgorithmCfg)
