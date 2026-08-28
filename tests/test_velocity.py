"""Tests for the velocity task configurations."""

from __future__ import annotations

import torch
from mjlab.tasks.registry import list_tasks
from mjlab_playground.tasks.velocity import mdp as velocity_local_mdp
from mjlab_playground.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
from mjlab_playground.tasks.velocity.mdp.curriculums import commands_vel
from mjlab_playground.tasks.velocity.mdp.observations import generated_commands
from mjlab_playground.tasks.velocity.mdp.rewards import _is_forward_command_active


def test_velocity_task_registration() -> None:
  import mjlab_playground  # noqa: F401

  tasks = list_tasks()
  assert "Mjlab-Velocity-Flat-Booster-T1" in tasks
  assert "Mjlab-Velocity-Rough-Booster-T1" not in tasks
  assert "Mjlab-VelocityYaw-Flat-Booster-T1" not in tasks
  assert "Mjlab-VelocityRaw-Flat-Booster-T1" not in tasks


def test_t1_velocity_play_cfg_resamples_visible_commands() -> None:
  cfg = booster_t1_flat_env_cfg(play=True)
  twist = cfg.commands["twist"]
  assert twist.resampling_time_range == (1.0, 1.5)
  assert twist.rel_standing_envs == 0.0
  assert twist.rel_lateral_envs == 0.0
  assert twist.ranges.lin_vel_x == (2.0, 3.0)
  assert twist.ranges.lin_vel_y == (0.0, 0.0)
  assert twist.ranges.heading == (-0.5, 0.5)


def test_t1_velocity_uses_mixed_stand_x_heading_and_lateral_commands() -> None:
  cfg = booster_t1_flat_env_cfg()
  twist = cfg.commands["twist"]
  assert twist.heading_command is True
  assert twist.rel_lateral_envs == 0.2
  assert twist.rel_heading_envs == 1.0
  assert twist.rel_forward_envs == 0.0
  assert twist.ranges.lin_vel_y == (-0.3, 0.3)
  assert twist.ranges.ang_vel_z == (-0.5, 0.5)
  assert twist.ranges.heading == (-torch.pi, torch.pi)

  assert (
    cfg.observations["actor"].terms["command"].func
    is velocity_local_mdp.generated_commands
  )
  assert cfg.rewards["track_heading"].func is velocity_local_mdp.track_heading_command

  assert set(cfg.curriculum.keys()) == {"command_vel"}
  stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert stages[0]["lin_vel_x"] == (-0.5, 1.5)
  assert stages[0]["lin_vel_y"] == (-0.3, 0.3)
  assert stages[0]["heading"] == (-0.5, 0.5)
  assert "ang_vel_z" not in stages[0]
  assert stages[1]["lin_vel_x"] == (-1.0, 2.5)
  assert stages[1]["lin_vel_y"] == (-0.6, 0.6)
  assert stages[1]["heading"] == (-0.7, 0.7)
  assert "ang_vel_z" not in stages[1]
  assert stages[2]["lin_vel_x"] == (0.0, 3.0)
  assert stages[2]["lin_vel_y"] == (-1.0, 1.0)
  assert stages[2]["heading"] == (-torch.pi * 0.5, torch.pi * 0.5)
  assert "ang_vel_z" not in stages[2]


def test_generated_commands_returns_x_y_plus_heading_encoding() -> None:
  class FakeCommandTerm:
    def __init__(self) -> None:
      self.heading_target = torch.tensor(
        [0.0, torch.pi / 2, torch.pi], dtype=torch.float32
      )
      self.is_heading_env = torch.tensor([False, True, False])
      self.is_standing_env = torch.tensor([True, False, False])
      self.robot = type(
        "Robot",
        (),
        {
          "data": type(
            "Data",
            (),
            {"heading_w": torch.tensor([1.2, 0.0, 0.4], dtype=torch.float32)},
          )()
        },
      )()

  class FakeCommandManager:
    def get_command(self, name: str) -> torch.Tensor:
      assert name == "twist"
      return torch.tensor(
        [
          [0.0, 0.0, 0.0],
          [1.1, 0.0, -0.1],
          [0.0, -0.6, 0.0],
        ],
        dtype=torch.float32,
      )

    def get_term(self, name: str) -> FakeCommandTerm:
      assert name == "twist"
      return FakeCommandTerm()

  env = type("Env", (), {"command_manager": FakeCommandManager(), "scene": {}})()
  obs = generated_commands(env, command_name="twist")
  expected = torch.tensor(
    [
      [0.0, 0.0, 0.0, 1.0],
      [1.1, 0.0, 1.0, 0.0],
      [0.0, -0.6, 0.0, 1.0],
    ],
    dtype=torch.float32,
  )
  assert torch.allclose(obs, expected, atol=1e-5)


def test_commands_vel_logs_heading_ranges_only() -> None:
  cfg = booster_t1_flat_env_cfg()
  command_term = type("CommandTerm", (), {"cfg": cfg.commands["twist"]})()
  env = type(
    "Env",
    (),
    {
      "common_step_counter": 120000,
      "command_manager": type(
        "CommandManager", (), {"get_term": lambda self, name: command_term}
      )(),
    },
  )()
  metrics = commands_vel(
    env,
    env_ids=torch.tensor([], dtype=torch.long),
    command_name="twist",
    velocity_stages=cfg.curriculum["command_vel"].params["velocity_stages"],
  )
  assert set(metrics.keys()) == {
    "lin_vel_x_min",
    "lin_vel_x_max",
    "lin_vel_y_min",
    "lin_vel_y_max",
    "heading_min",
    "heading_max",
  }


def test_forward_command_gate_ignores_y_and_heading() -> None:
  command = torch.tensor(
    [
      [0.0, 0.8, 0.0],
      [0.0, 0.0, 0.7],
      [0.2, 0.8, 0.7],
      [-0.2, 0.0, 0.0],
    ],
    dtype=torch.float32,
  )

  active = _is_forward_command_active(command, command_threshold=0.1)

  expected = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
  assert torch.equal(active, expected)
