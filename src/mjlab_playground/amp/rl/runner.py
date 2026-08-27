"""On-policy runner with adversarial motion-prior rewards."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

import os
import time

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan

from mjlab_playground.amp.mdp.observations import amp_robot_obs


class AmpOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that computes AMP observations around env.step()."""

  _VERBOSE_LOG_VALUES = {"1", "true", "yes", "on"}

  def _amp_obs(self) -> torch.Tensor:
    return amp_robot_obs(self.env.unwrapped).to(self.device)

  def _amp_cfg(self) -> dict:
    return dict(
      getattr(self.alg, "amp_cfg", {}) or self.cfg["algorithm"].get("amp_cfg", {})
    )

  def _log_namespace(self) -> str:
    return str(self._amp_cfg().get("log_namespace", "amp"))

  def _verbose_log_enabled(self) -> bool:
    return (
      os.environ.get("MJLAB_AMP_VERBOSE_LOG", "").lower() in self._VERBOSE_LOG_VALUES
    )

  def _stage_weights(self, valid: torch.Tensor) -> dict[str, torch.Tensor] | None:
    if not getattr(self.alg, "uses_stage_amp", False):
      return None
    env = self.env.unwrapped
    stage_weight_map = getattr(env, "_amp_stage_weight_map", None)
    if isinstance(stage_weight_map, dict) and stage_weight_map:
      resolved: dict[str, torch.Tensor] = {}
      for name, weight in stage_weight_map.items():
        resolved[name] = weight.to(self.device).view(-1)[valid.view(-1)]
      return resolved

    raise RuntimeError(
      "Stage AMP is enabled but env._amp_stage_weight_map is not available."
    )

  def _task_reward_rate(self, task_rewards: torch.Tensor) -> tuple[torch.Tensor, float]:
    env = self.env.unwrapped
    scale = 1.0
    if getattr(env.cfg, "scale_rewards_by_dt", True):
      scale = max(float(getattr(env, "step_dt", 1.0)), 1.0e-6)
    return task_rewards / scale, scale

  def _style_reward_ceiling(
    self,
    task_rewards: torch.Tensor,
    stage_weights: dict[str, torch.Tensor] | None,
  ) -> torch.Tensor:
    def discriminator_ceiling(discriminator) -> float:
      return float(discriminator.amp_reward_coef) * float(
        discriminator.amp_reward_scale
      )

    if not getattr(self.alg, "uses_stage_amp", False):
      discriminator = getattr(self.alg, "discriminator", None)
      ceiling = (
        discriminator_ceiling(discriminator) if discriminator is not None else 0.0
      )
      return torch.full_like(task_rewards, ceiling)

    total_weight = torch.zeros_like(task_rewards)
    ceiling = torch.zeros_like(task_rewards)
    for name, discriminator in self.alg.stage_discriminators.items():
      if stage_weights is None:
        weight = torch.full_like(task_rewards, 1.0 / len(self.alg.stage_discriminators))
      else:
        weight = stage_weights.get(name, torch.zeros_like(task_rewards))
        weight = weight.to(device=task_rewards.device, dtype=task_rewards.dtype).view(
          -1
        )
      ceiling += weight * discriminator_ceiling(discriminator)
      total_weight += weight
    return ceiling / total_weight.clamp_min(1.0e-6)

  def _style_reward_gate(
    self,
    style_rewards: torch.Tensor,
    dones: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    gate_cfg = self._amp_cfg().get("style_reward_gate")
    if not gate_cfg:
      return style_rewards, None

    env = self.env.unwrapped
    entity_name = str(gate_cfg.get("entity_name", "robot"))
    asset = env.scene[entity_name]
    healthy = torch.ones_like(style_rewards, dtype=torch.bool, device=self.device)

    if gate_cfg.get("zero_on_done", True):
      healthy = healthy & (~dones.to(self.device).bool().view(-1))
    if gate_cfg.get("min_root_height") is not None:
      root_height = asset.data.root_link_pos_w[:, 2].to(self.device).view(-1)
      healthy = healthy & (root_height >= float(gate_cfg["min_root_height"]))
    if gate_cfg.get("min_up_z") is not None:
      up_z = (-asset.data.projected_gravity_b[:, 2]).to(self.device).view(-1)
      healthy = healthy & (up_z >= float(gate_cfg["min_up_z"]))

    gated_style = torch.where(healthy, style_rewards, torch.zeros_like(style_rewards))
    return gated_style, healthy.float()

  def _apply_amp_reward_schedule(self) -> tuple[float, float]:
    amp_cfg = dict(getattr(self.alg, "amp_cfg", {}))
    if not amp_cfg:
      amp_cfg = self.cfg["algorithm"].get("amp_cfg", {})
    schedule = amp_cfg.get("reward_mix_schedule", ())

    task_reward_lerp = float(getattr(self.alg, "task_reward_lerp", 0.0))
    main_discriminator = getattr(self.alg, "discriminator", None)
    reward_coef = (
      float(getattr(main_discriminator, "amp_reward_coef", 0.0))
      if main_discriminator is not None
      else 0.0
    )

    for stage in schedule:
      if self.env.unwrapped.common_step_counter < int(stage["step"]):
        continue
      if "task_reward_lerp" in stage:
        task_reward_lerp = float(stage["task_reward_lerp"])
      if "reward_coef" in stage:
        reward_coef = float(stage["reward_coef"])

    self.alg.task_reward_lerp = task_reward_lerp
    discriminators = []
    if main_discriminator is not None:
      discriminators.append(main_discriminator)
    discriminators.extend(getattr(self.alg, "stage_discriminators", {}).values())

    seen: set[int] = set()
    for discriminator in discriminators:
      disc_id = id(discriminator)
      if disc_id in seen:
        continue
      seen.add(disc_id)
      discriminator.task_reward_lerp = task_reward_lerp
      discriminator.amp_reward_coef = reward_coef

    return task_reward_lerp, reward_coef

  def _disable_auto_reset_for_terminal_amp(self) -> None:
    """Let the runner capture terminal AMP transitions before resetting envs."""
    self.env.unwrapped.cfg.auto_reset = False

  def _reset_done_envs(self, dones: torch.Tensor) -> torch.Tensor | None:
    done_ids = dones.to(self.env.device).bool().nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() == 0:
      return None
    self.env.unwrapped.reset(env_ids=done_ids)
    return self.env.get_observations().to(self.device)

  def _learning_start_iteration(self) -> int:
    start_it = self.current_learning_iteration
    if getattr(self, "_resume_next_iteration", False):
      start_it += 1
      self._resume_next_iteration = False
    return start_it

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if self._amp_cfg().get("disable_init_at_random_ep_len", False):
      init_at_random_ep_len = False
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self._disable_auto_reset_for_terminal_amp()
    self.logger.init_logging_writer()
    verbose_log = self._verbose_log_enabled()
    log_namespace = self._log_namespace()

    start_it = self._learning_start_iteration()
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          amp_obs = self._amp_obs()
          actions = self.alg.act(obs)
          obs, task_rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, task_rewards, dones)
          next_amp_obs = self._amp_obs()

          obs = obs.to(self.device)
          task_rewards = task_rewards.to(self.device)
          dones = dones.to(self.device)

          valid = torch.ones_like(dones, dtype=torch.bool)
          task_reward_weight, amp_reward_coef = self._apply_amp_reward_schedule()
          amp_reward_weight = 1.0 - task_reward_weight
          stage_weights = self._stage_weights(valid)
          amp_rewards, amp_prior_rewards, amp_disc, amp_stage_logs = (
            self.alg.predict_amp_reward(
              amp_obs,
              next_amp_obs,
              task_rewards,
              stage_weights=stage_weights,
            )
          )
          amp_prior_rewards, style_gate = self._style_reward_gate(
            amp_prior_rewards, dones
          )
          amp_rewards = (
            task_reward_weight * task_rewards + amp_reward_weight * amp_prior_rewards
          )
          self.alg.process_amp_step(
            amp_obs,
            next_amp_obs,
            stage_weights=stage_weights,
          )

          extras.setdefault("log", {})[f"Rewards/{log_namespace}/summary/task"] = (
            task_rewards.mean()
          )
          task_component_mean = task_reward_weight * task_rewards.mean()
          amp_component_mean = amp_reward_weight * amp_prior_rewards.mean()
          extras["log"][f"Rewards/{log_namespace}/summary/style_prior"] = (
            amp_prior_rewards.mean()
          )
          extras["log"][f"Rewards/{log_namespace}/summary/train_mixed"] = (
            amp_rewards.mean()
          )
          extras["log"][f"Rewards/{log_namespace}/summary/style_to_task_abs"] = (
            torch.abs(amp_component_mean)
            / torch.abs(task_component_mean).clamp_min(1.0e-6)
          )
          if style_gate is not None:
            extras["log"][f"Metrics/{log_namespace}/style_reward_gate_fraction"] = (
              style_gate.mean()
            )
          extras["log"][f"Curriculum/{log_namespace}/task_reward_lerp"] = (
            torch.as_tensor(
              task_reward_weight,
              dtype=task_rewards.dtype,
              device=task_rewards.device,
            )
          )
          extras["log"][f"Curriculum/{log_namespace}/reward_coef"] = torch.as_tensor(
            amp_reward_coef,
            dtype=task_rewards.dtype,
            device=task_rewards.device,
          )
          if getattr(self.alg, "uses_stage_amp", False):
            for name, replay in self.alg.stage_replay_buffers.items():
              extras["log"][f"Metrics/{log_namespace}/amp/{name}_replay_size"] = (
                torch.as_tensor(
                  replay.size,
                  dtype=task_rewards.dtype,
                  device=task_rewards.device,
                )
              )
          else:
            replay = getattr(self.alg, "amp_replay", None)
            if replay is not None:
              extras["log"][f"Metrics/{log_namespace}/amp/replay_size"] = (
                torch.as_tensor(
                  replay.size,
                  dtype=task_rewards.dtype,
                  device=task_rewards.device,
                )
              )
          if verbose_log:
            task_reward_rate, task_dt_scale = self._task_reward_rate(task_rewards)
            style_ceiling = self._style_reward_ceiling(task_rewards, stage_weights)
            amp_ceiling_component_mean = amp_reward_weight * style_ceiling.mean()
            extras["log"][
              f"RewardsDebug/{log_namespace}/summary/task_rate_unscaled"
            ] = task_reward_rate.mean()
            extras["log"][f"RewardsDebug/{log_namespace}/components/task"] = (
              task_component_mean
            )
            extras["log"][f"RewardsDebug/{log_namespace}/components/style"] = (
              amp_component_mean
            )
            extras["log"][f"RewardsDebug/{log_namespace}/components/style_ceiling"] = (
              amp_ceiling_component_mean
            )
            extras["log"][
              f"RewardsDebug/{log_namespace}/ratios/style_ceiling_to_task_abs"
            ] = torch.abs(amp_ceiling_component_mean) / torch.abs(
              task_component_mean
            ).clamp_min(1.0e-6)
            extras["log"][
              f"MetricsDebug/{log_namespace}/reward_scale/task_dt_scale"
            ] = torch.as_tensor(
              task_dt_scale,
              dtype=task_rewards.dtype,
              device=task_rewards.device,
            )
            extras["log"][
              f"MetricsDebug/{log_namespace}/reward_scale/style_fraction_of_ceiling"
            ] = amp_prior_rewards.mean() / style_ceiling.mean().clamp_min(1.0e-6)
            extras["log"][f"RewardsDebug/{log_namespace}/amp/scale"] = torch.as_tensor(
              style_ceiling.mean(),
              dtype=task_rewards.dtype,
              device=task_rewards.device,
            )
            extras["log"][f"MetricsDebug/{log_namespace}/amp/disc_pred_mean"] = (
              amp_disc.mean()
            )
            extras["log"][f"MetricsDebug/{log_namespace}/amp/done_transition_ratio"] = (
              dones.float().mean()
            )
            for key, value in amp_stage_logs.items():
              extras["log"][f"MetricsDebug/{log_namespace}/amp/{key}"] = value.mean()

          step_extras = dict(extras)
          if "log" in extras:
            step_extras["log"] = dict(extras["log"])
          obs_after_reset = self._reset_done_envs(dones)
          if obs_after_reset is not None:
            obs = obs_after_reset

          self.alg.process_env_step(obs, amp_rewards, dones, step_extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(
            amp_rewards, dones, step_extras, intrinsic_rewards
          )

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
        print_minimal=not verbose_log,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )  # type: ignore[arg-type]
      self.logger.stop_logging_writer()
