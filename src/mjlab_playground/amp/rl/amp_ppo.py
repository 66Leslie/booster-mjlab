"""PPO variant with an AMP discriminator update."""

# pyright: reportArgumentType=false, reportCallIssue=false, reportIndexIssue=false
# pyright: reportOptionalMemberAccess=false, reportPossiblyUnboundVariable=false

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict

from .discriminator import AmpDiscriminator
from .motion_loader import RetargetedAmpMotionLoader
from .replay_buffer import AmpTransitionBuffer


class AmpPPO(PPO):
  """Current RSL-RL-compatible PPO with AMP discriminator training."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    discriminator: AmpDiscriminator,
    expert_loader: RetargetedAmpMotionLoader,
    replay_buffer: AmpTransitionBuffer,
    stage_discriminators: dict[str, AmpDiscriminator] | None = None,
    stage_expert_loaders: dict[str, RetargetedAmpMotionLoader] | None = None,
    stage_replay_buffers: dict[str, AmpTransitionBuffer] | None = None,
    amp_loss_coef: float = 1.0,
    amp_cfg: dict | None = None,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 0.001,
    discriminator_learning_rate: float | None = None,
    discriminator_weight_decay: float = 1.0e-4,
    max_grad_norm: float = 1.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    device: str = "cpu",
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    super().__init__(
      actor=actor,
      critic=critic,
      storage=storage,
      num_learning_epochs=num_learning_epochs,
      num_mini_batches=num_mini_batches,
      clip_param=clip_param,
      gamma=gamma,
      lam=lam,
      value_loss_coef=value_loss_coef,
      entropy_coef=entropy_coef,
      learning_rate=learning_rate,
      max_grad_norm=max_grad_norm,
      optimizer=optimizer,
      use_clipped_value_loss=use_clipped_value_loss,
      schedule=schedule,
      desired_kl=desired_kl,
      normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
      device=device,
      rnd_cfg=rnd_cfg,
      symmetry_cfg=symmetry_cfg,
      multi_gpu_cfg=multi_gpu_cfg,
    )
    self.discriminator = discriminator.to(self.device)
    self.expert_loader = expert_loader
    self.amp_replay = replay_buffer
    self.amp_cfg = dict(amp_cfg or {})
    self.stage_discriminators = {
      name: module.to(self.device)
      for name, module in (stage_discriminators or {}).items()
    }
    self.stage_expert_loaders = dict(stage_expert_loaders or {})
    self.stage_replay_buffers = dict(stage_replay_buffers or {})
    self.uses_stage_amp = bool(self.stage_discriminators)
    self.stage_policy_replay: AmpTransitionBuffer | None = None
    if self.uses_stage_amp:
      self.stage_policy_replay = AmpTransitionBuffer(
        capacity=replay_buffer.capacity,
        amp_obs_dim=replay_buffer.amp_obs_dim,
        device=device,
      )
    self.task_reward_lerp = float(self.discriminator.task_reward_lerp)
    self.amp_loss_coef = float(amp_loss_coef)
    self.discriminator_learning_rate = (
      float(learning_rate)
      if discriminator_learning_rate is None
      else float(discriminator_learning_rate)
    )

    self.optimizer = resolve_optimizer(optimizer)(
      [
        {
          "params": chain(self.actor.parameters(), self.critic.parameters()),
          "name": "policy",
          "lr": learning_rate,
        },
        {
          "params": list(self._discriminator_parameters()),
          "name": "discriminator",
          "lr": self.discriminator_learning_rate,
          "weight_decay": float(discriminator_weight_decay),
        },
      ],
      lr=learning_rate,
    )

  def _discriminator_modules(self) -> list[AmpDiscriminator]:
    if self.uses_stage_amp:
      return list(self.stage_discriminators.values())
    return [self.discriminator]

  def _discriminator_parameters(self):
    return chain.from_iterable(
      module.parameters() for module in self._discriminator_modules()
    )

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> "AmpPPO":
    alg_class: type[AmpPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    amp_cfg = cfg["algorithm"].pop("amp_cfg")
    actor: MLPModel = actor_class(
      obs,
      cfg["obs_groups"],
      "actor",
      env.num_actions,
      **cfg["actor"],
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )

    def make_loader(local_cfg: dict) -> RetargetedAmpMotionLoader:
      return RetargetedAmpMotionLoader(
        tuple(local_cfg.get("motion_files", ())),
        device=device,
        expected_dof=local_cfg.get("expected_dof", amp_cfg.get("expected_dof", 22)),
        motion_groups=local_cfg.get("motion_groups"),
        target_fps=local_cfg.get("target_fps", amp_cfg.get("target_fps")),
        joint_names=tuple(local_cfg.get("joint_names", amp_cfg.get("joint_names", ())))
        or None,
        key_body_names=tuple(
          local_cfg.get("key_body_names", amp_cfg.get("key_body_names", ()))
        )
        or None,
        emphasis_frame_offset_range=local_cfg.get(
          "emphasis_frame_offset_range",
          amp_cfg.get("emphasis_frame_offset_range"),
        ),
        emphasis_weight=local_cfg.get(
          "emphasis_weight",
          amp_cfg.get("emphasis_weight"),
        ),
        velocity_emphasis_weight=local_cfg.get(
          "velocity_emphasis_weight",
          amp_cfg.get("velocity_emphasis_weight"),
        ),
      )

    def make_discriminator(
      loader: RetargetedAmpMotionLoader,
      local_cfg: dict,
    ) -> AmpDiscriminator:
      return AmpDiscriminator(
        amp_obs_dim=loader.amp_obs_dim,
        hidden_dims=tuple(
          local_cfg.get(
            "discriminator_hidden_dims",
            amp_cfg.get("discriminator_hidden_dims", (512, 256)),
          )
        ),
        activation=local_cfg.get(
          "discriminator_activation",
          amp_cfg.get("discriminator_activation", "relu"),
        ),
        amp_reward_coef=local_cfg.get("reward_coef", amp_cfg.get("reward_coef", 0.3)),
        amp_reward_scale=local_cfg.get(
          "reward_scale", amp_cfg.get("reward_scale", 1.0)
        ),
        task_reward_lerp=amp_cfg.get("task_reward_lerp", 0.7),
        grad_penalty_coef=local_cfg.get(
          "grad_penalty_coef",
          amp_cfg.get("grad_penalty_coef", 10.0),
        ),
        normalize_input=local_cfg.get(
          "normalize_input", amp_cfg.get("normalize_input", True)
        ),
        loss_type=local_cfg.get(
          "discriminator_loss_type",
          amp_cfg.get("discriminator_loss_type", "bce"),
        ),
        reward_type=local_cfg.get(
          "discriminator_reward_type",
          amp_cfg.get("discriminator_reward_type", "softplus"),
        ),
        expert_label=local_cfg.get(
          "discriminator_expert_label",
          amp_cfg.get("discriminator_expert_label", 0.9),
        ),
        policy_label=local_cfg.get(
          "discriminator_policy_label",
          amp_cfg.get("discriminator_policy_label", 0.1),
        ),
        reward_clip_max=local_cfg.get(
          "discriminator_reward_clip_max",
          amp_cfg.get("discriminator_reward_clip_max", 1.5),
        ),
      ).to(device)

    stage_cfg = dict(amp_cfg.get("stage_amp_cfg", {}))
    stage_discriminators: dict[str, AmpDiscriminator] = {}
    stage_expert_loaders: dict[str, RetargetedAmpMotionLoader] = {}
    stage_replay_buffers: dict[str, AmpTransitionBuffer] = {}
    if stage_cfg.get("enabled", False):
      reserved_stage_cfg_keys = {"enabled", "stage_names", "stages", "primary_stage"}
      if isinstance(stage_cfg.get("stages"), dict):
        stage_items = list(stage_cfg["stages"].items())
      else:
        stage_names = stage_cfg.get("stage_names")
        if stage_names is None:
          inferred_names = [
            name for name in stage_cfg.keys() if name not in reserved_stage_cfg_keys
          ]
          if inferred_names:
            stage_names = tuple(inferred_names)
          else:
            raise ValueError(
              "stage_amp_cfg is enabled but no stage definitions were provided."
            )
        stage_items = [
          (str(name), stage_cfg.get(str(name), {})) for name in stage_names
        ]

      if not stage_items:
        raise ValueError("stage_amp_cfg is enabled but resolved to zero AMP stages.")

      for name, raw_local_cfg in stage_items:
        local_cfg = dict(raw_local_cfg)
        local_cfg.setdefault("motion_groups", stage_cfg.get(f"{name}_motion_groups"))
        local_cfg.setdefault("motion_files", stage_cfg.get(f"{name}_motion_files", ()))
        loader = make_loader(local_cfg)
        stage_expert_loaders[name] = loader
        stage_discriminators[name] = make_discriminator(loader, local_cfg)
        stage_replay_buffers[name] = AmpTransitionBuffer(
          capacity=local_cfg.get(
            "replay_buffer_size", amp_cfg.get("replay_buffer_size", 100_000)
          ),
          amp_obs_dim=loader.amp_obs_dim,
          device=device,
        )
      primary_stage_name = str(stage_cfg.get("primary_stage", ""))
      if primary_stage_name not in stage_discriminators:
        primary_stage_name = next(iter(stage_discriminators))
      expert_loader = stage_expert_loaders[primary_stage_name]
      discriminator = stage_discriminators[primary_stage_name]
      replay_buffer = stage_replay_buffers[primary_stage_name]
      for name, loader in stage_expert_loaders.items():
        print(
          f"AMP {name} expert transitions: "
          f"{loader.num_transitions}, obs dim: {loader.amp_obs_dim}"
        )
    else:
      expert_loader = make_loader(amp_cfg)
      discriminator = make_discriminator(expert_loader, amp_cfg)
      replay_buffer = AmpTransitionBuffer(
        capacity=amp_cfg.get("replay_buffer_size", 100_000),
        amp_obs_dim=expert_loader.amp_obs_dim,
        device=device,
      )
      print(
        "AMP expert transitions: "
        f"{expert_loader.num_transitions}, obs dim: {expert_loader.amp_obs_dim}"
      )

    return alg_class(
      actor=actor,
      critic=critic,
      storage=storage,
      discriminator=discriminator,
      expert_loader=expert_loader,
      replay_buffer=replay_buffer,
      stage_discriminators=stage_discriminators or None,
      stage_expert_loaders=stage_expert_loaders or None,
      stage_replay_buffers=stage_replay_buffers or None,
      amp_loss_coef=amp_cfg.get("loss_coef", 1.0),
      amp_cfg=amp_cfg,
      discriminator_learning_rate=amp_cfg.get("discriminator_learning_rate"),
      discriminator_weight_decay=amp_cfg.get("discriminator_weight_decay", 1.0e-4),
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )

  def train_mode(self) -> None:
    super().train_mode()
    for discriminator in self._discriminator_modules():
      discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    for discriminator in self._discriminator_modules():
      discriminator.eval()

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    stage_weights: dict[str, torch.Tensor] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return mixed reward, raw style reward, discriminator score, and stage logs."""
    if not self.uses_stage_amp:
      mixed, disc = self.discriminator.predict_reward(state, next_state, task_reward)
      amp_weight = 1.0 - self.task_reward_lerp
      style_reward = torch.zeros_like(task_reward)
      if amp_weight > 0.0:
        style_reward = (mixed - self.task_reward_lerp * task_reward) / amp_weight
      return mixed, style_reward, disc, {}

    if stage_weights is None:
      stage_weights = {
        name: torch.full_like(task_reward, 1.0 / len(self.stage_discriminators))
        for name in self.stage_discriminators
      }
    total_weight = torch.zeros_like(task_reward)
    style_reward = torch.zeros_like(task_reward)
    disc_pred = torch.zeros_like(task_reward)
    logs: dict[str, torch.Tensor] = {}
    for name, discriminator in self.stage_discriminators.items():
      weight = stage_weights.get(name)
      if weight is None:
        weight = torch.zeros_like(task_reward)
      weight = weight.to(device=task_reward.device, dtype=task_reward.dtype).view(-1)
      stage_style, stage_disc = discriminator.predict_style_reward(state, next_state)
      style_reward += weight * stage_style
      disc_pred += weight * stage_disc
      total_weight += weight
      logs[f"{name}_weight"] = weight.detach()
      logs[f"{name}_style"] = stage_style.detach()
      logs[f"{name}_disc"] = stage_disc.detach()
    safe_weight = total_weight.clamp_min(1.0e-6)
    style_reward = style_reward / safe_weight
    disc_pred = disc_pred / safe_weight
    mixed = (
      self.task_reward_lerp * task_reward + (1.0 - self.task_reward_lerp) * style_reward
    )
    return mixed, style_reward, disc_pred, logs

  def process_amp_step(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    stage_weights: dict[str, torch.Tensor] | None = None,
  ) -> None:
    if not self.uses_stage_amp:
      self.amp_replay.insert(state, next_state)
      return
    if self.stage_policy_replay is not None:
      self.stage_policy_replay.insert(state, next_state)
    if stage_weights is None:
      return
    for name, replay_buffer in self.stage_replay_buffers.items():
      weight = stage_weights.get(name)
      if weight is None:
        continue
      mask = weight.to(device=state.device).view(-1) >= 0.5
      if torch.any(mask):
        replay_buffer.insert(state[mask], next_state[mask])

  def _sample_stage_policy(
    self,
    name: str,
    batch_size: int,
  ) -> tuple[torch.Tensor, torch.Tensor, bool]:
    replay = self.stage_replay_buffers[name]
    if replay.size > 0:
      policy_state, policy_next_state = replay.sample(batch_size)
      return policy_state, policy_next_state, False
    if self.stage_policy_replay is not None and self.stage_policy_replay.size > 0:
      policy_state, policy_next_state = self.stage_policy_replay.sample(batch_size)
      return policy_state, policy_next_state, True
    raise RuntimeError(
      f"Stage AMP replay buffer {name!r} is empty and no policy fallback "
      "transitions are available. This means no policy AMP transitions were "
      "recorded before the discriminator update."
    )

  def update(self) -> dict[str, float]:  # noqa: C901
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_amp_loss = 0.0
    mean_grad_pen = 0.0
    mean_policy_pred = 0.0
    mean_expert_pred = 0.0
    mean_stage_stats = {
      name: {
        "amp": 0.0,
        "policy_pred": 0.0,
        "expert_pred": 0.0,
        "fallback": 0.0,
      }
      for name in self.stage_discriminators
    }
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (  # type: ignore[union-attr]
            batch.advantages.std() + 1e-8  # type: ignore[union-attr]
          )

      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"],
          obs=batch.observations,
          actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,
            distribution_params,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            if param_group.get("name") == "discriminator":
              param_group["lr"] = self.discriminator_learning_rate
            else:
              param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param,
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(  # type: ignore[operator]
          -self.clip_param,
          self.clip_param,
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(
            obs=batch.observations,
            actions=None,
            env=self.symmetry["_env"],
          )
        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None,
          actions=action_mean_orig,
          env=self.symmetry["_env"],
        )
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      rnd_loss = (
        self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore[index]
        if self.rnd
        else None
      )

      mini_batch_size = int(original_batch_size)
      if self.uses_stage_amp:
        amp_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        stage_stats: list[dict[str, torch.Tensor]] = []
        for name, discriminator in self.stage_discriminators.items():
          loader = self.stage_expert_loaders[name]
          policy_state, policy_next_state, used_fallback = self._sample_stage_policy(
            name,
            mini_batch_size,
          )
          expert_state, expert_next_state = loader.sample(mini_batch_size)
          stage_loss, stats = discriminator.discriminator_loss(
            policy_state,
            policy_next_state,
            expert_state,
            expert_next_state,
          )
          amp_loss = amp_loss + stage_loss
          stage_stats.append(stats)
          mean_stage_stats[name]["amp"] += stats["amp"].item()
          mean_stage_stats[name]["policy_pred"] += stats["amp_policy_pred"].item()
          mean_stage_stats[name]["expert_pred"] += stats["amp_expert_pred"].item()
          mean_stage_stats[name]["fallback"] += float(used_fallback)
        amp_loss = amp_loss / max(len(stage_stats), 1)
        amp_stats = {
          key: torch.stack([stats[key] for stats in stage_stats]).mean()
          for key in stage_stats[0]
        }
      else:
        if self.amp_replay.size > 0:
          policy_state, policy_next_state = self.amp_replay.sample(mini_batch_size)
        else:
          policy_state, policy_next_state = self.expert_loader.sample(mini_batch_size)
        expert_state, expert_next_state = self.expert_loader.sample(mini_batch_size)
        amp_loss, amp_stats = self.discriminator.discriminator_loss(
          policy_state,
          policy_next_state,
          expert_state,
          expert_next_state,
        )
      loss += self.amp_loss_coef * amp_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        assert rnd_loss is not None
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(
        list(self._discriminator_parameters()), self.max_grad_norm
      )
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_amp_loss += amp_stats["amp"].item()
      mean_grad_pen += amp_stats["amp_grad_pen"].item()
      mean_policy_pred += amp_stats["amp_policy_pred"].item()
      mean_expert_pred += amp_stats["amp_expert_pred"].item()
      if mean_rnd_loss is not None:
        assert rnd_loss is not None
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "amp": mean_amp_loss / num_updates,
      "amp_grad_pen": mean_grad_pen / num_updates,
      "amp_policy_pred": mean_policy_pred / num_updates,
      "amp_expert_pred": mean_expert_pred / num_updates,
    }
    for name, stats in mean_stage_stats.items():
      loss_dict[f"amp_{name}"] = stats["amp"] / num_updates
      loss_dict[f"amp_{name}_policy_pred"] = stats["policy_pred"] / num_updates
      loss_dict[f"amp_{name}_expert_pred"] = stats["expert_pred"] / num_updates
      loss_dict[f"amp_{name}_policy_fallback"] = stats["fallback"] / num_updates
    if mean_rnd_loss is not None:
      loss_dict["rnd"] = mean_rnd_loss / num_updates
    if mean_symmetry_loss is not None:
      loss_dict["symmetry"] = mean_symmetry_loss / num_updates

    self.storage.clear()
    return loss_dict

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
    if self.uses_stage_amp:
      saved_dict["stage_discriminator_state_dicts"] = {
        name: discriminator.state_dict()
        for name, discriminator in self.stage_discriminators.items()
      }
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_disc = True if load_cfg is None else load_cfg.get("discriminator", True)
    try:
      load_iteration = super().load(loaded_dict, load_cfg, strict)
    except (RuntimeError, ValueError) as exc:
      wants_optimizer = True if load_cfg is None else load_cfg.get("optimizer", True)
      if not wants_optimizer:
        raise
      retry_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": True,
      }
      if load_cfg is not None:
        retry_cfg.update(load_cfg)
        retry_cfg["optimizer"] = False
      print(
        "Skipping optimizer state while loading AMP checkpoint because the "
        f"current optimizer layout changed: {exc}"
      )
      load_iteration = super().load(loaded_dict, retry_cfg, strict)
    if load_disc and self.uses_stage_amp:
      stage_state_dicts = loaded_dict.get("stage_discriminator_state_dicts")
      if isinstance(stage_state_dicts, dict):
        for name, state_dict in stage_state_dicts.items():
          if name in self.stage_discriminators:
            self.stage_discriminators[name].load_state_dict(state_dict, strict=strict)
      elif "discriminator_state_dict" in loaded_dict:
        target_name = next(iter(self.stage_discriminators))
        self.stage_discriminators[target_name].load_state_dict(
          loaded_dict["discriminator_state_dict"],
          strict=strict,
        )
    elif (
      load_disc
      and (not self.uses_stage_amp)
      and "stage_discriminator_state_dicts" in loaded_dict
      and not (load_cfg or {}).get("allow_stage_to_single_discriminator", False)
    ):
      print(
        "Skipping discriminator state while loading a stage-AMP checkpoint into "
        "a single-discriminator AMP configuration. Set "
        "load_cfg['allow_stage_to_single_discriminator']=True to override."
      )
    elif load_disc and "discriminator_state_dict" in loaded_dict:
      self.discriminator.load_state_dict(
        loaded_dict["discriminator_state_dict"],
        strict=strict,
      )
    return load_iteration

  def broadcast_parameters(self) -> None:
    model_params = [
      self.actor.state_dict(),
      self.critic.state_dict(),
      (
        {
          name: module.state_dict()
          for name, module in self.stage_discriminators.items()
        }
        if self.uses_stage_amp
        else self.discriminator.state_dict()
      ),
    ]
    if self.rnd:
      model_params.append(self.rnd.predictor.state_dict())
    torch.distributed.broadcast_object_list(model_params, src=0)
    self.actor.load_state_dict(model_params[0])
    self.critic.load_state_dict(model_params[1])
    if self.uses_stage_amp:
      for name, state_dict in model_params[2].items():
        self.stage_discriminators[name].load_state_dict(state_dict)
    else:
      self.discriminator.load_state_dict(model_params[2])
    if self.rnd:
      self.rnd.predictor.load_state_dict(model_params[3])

  def reduce_parameters(self) -> None:
    all_params = chain(
      self.actor.parameters(),
      self.critic.parameters(),
      self._discriminator_parameters(),
    )
    if self.rnd:
      all_params = chain(all_params, self.rnd.parameters())
    all_params = list(all_params)
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(
          all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel
