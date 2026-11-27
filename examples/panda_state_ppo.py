"""State-only PPO on the real dual-Panda/free-pin MUSA scene.

First qualification task is joint-target tracking, NOT learned grasp/insertion.
No phase, time, reset label or scripted arm trajectory is a policy input. Physics
is never differentiated: PPO uses rewards from actual accepted successor states.
All rollout tensors, PD, reward, GAE, PPO and Adam are on torch.musa. Only audit
boundaries download data; the resident physics bridge checks W status integers.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np


SCHEMA = "panda-state-ppo-reset-boundary-v1"
OBSERVATION_SIZE = 23 + 22 + 16 + 16 + 16
ACTION_SIZE = 16
REWARD_DESCRIPTION = (
    "1 - mean(((accepted_q16-goal16)/joint_scale16)^2) "
    "- .002*mean((accepted_v16/velocity_scale16)^2) "
    "- .001*mean_substeps_and_motors((actual_ctrl/effort_limit)^2); "
    "joint_scale arms=.20rad/fingers=.035m, velocity_scale arms=2rad/s/fingers=.20m/s; "
    "not a grasp, load-bearing, seating or insertion success reward"
)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint_data(torch, value):
    """Persistence is restricted to plain containers/scalars and tensors."""
    if isinstance(value, torch.Tensor) or value is None or type(value) in (str, bool, int, float):
        return
    if type(value) in (tuple, list):
        for item in value:
            validate_checkpoint_data(torch, item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                raise TypeError("checkpoint permits only basic types and tensors (string/integer keys)")
            validate_checkpoint_data(torch, item)
        return
    raise TypeError("checkpoint permits only basic types and tensors")


def load_checkpoint(torch, path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_data(torch, value)
    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise ValueError("unknown checkpoint schema")
    return value


def initial_baseline(torch, path, trained_checkpoint):
    """Bind update-zero pre-optimizer weights to the final TRAINING source.

    Evaluation's independent seed/reset are deliberately not compared here.
    The two training checkpoints must retain identical model/task/contracts.
    """
    baseline = load_checkpoint(torch, path)
    if (type(baseline.get("next_update")) is not int or baseline["next_update"] != 1
            or baseline.get("completed_update", 0) != 0
            or baseline.get("rollout_file") != "update-00000-rollout.npz"):
        raise ValueError("baseline must be the update-00000 pre-update actor, not a later policy")
    if (trained_checkpoint.get("schema") != SCHEMA
            or type(baseline.get("contract")) is not dict
            or baseline["contract"] != trained_checkpoint.get("contract")):
        raise ValueError("baseline training model/task/contract differs from the loaded policy")
    frozen = baseline.get("frozen_model_sha256")
    if not isinstance(frozen, str) or len(frozen) != 64 or frozen != trained_checkpoint.get("frozen_model_sha256"):
        raise ValueError("baseline compiled physical model differs")
    a, b = baseline.get("goal"), trained_checkpoint.get("goal")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not torch.equal(a, b):
        raise ValueError("baseline training goals differ")
    probe = baseline.get("action_probe")
    if type(probe) is not dict or type(probe.get("before_actor")) is not dict or not probe["before_actor"]:
        raise ValueError("baseline has no saved pre-update actor")
    if not all(isinstance(v, torch.Tensor) for v in probe["before_actor"].values()):
        raise ValueError("baseline actor state must consist of tensors")
    return probe["before_actor"], dict(
        checkpoint_path=str(Path(path).resolve()), checkpoint_sha256=_sha(path), source_update=0,
        actor_field="action_probe.before_actor", role="initial policy before the first optimizer update",
        training_seed=baseline["contract"]["seed"],
        training_reset_sha256=baseline["contract"]["physical_reset_sha256"],
        model_sha256=baseline["contract"]["model_sha256"], frozen_model_sha256=frozen)


def model_digest(model):
    """Freeze the physical model, not mutable simulator data or control state."""
    digest = hashlib.sha256()
    for name in ("body_parentid", "body_jntadr", "body_jntnum", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                 "jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "dof_damping", "dof_armature",
                 "geom_bodyid", "geom_type", "geom_pos", "geom_quat", "geom_size", "geom_condim", "geom_contype", "geom_conaffinity",
                 "geom_friction", "geom_solref", "geom_solimp", "actuator_gear", "actuator_ctrlrange",
                 "actuator_trntype", "actuator_gaintype", "actuator_biastype", "actuator_dyntype",
                 "actuator_gainprm", "actuator_biasprm", "qpos0"):
        value = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode()+value.dtype.str.encode()+str(value.shape).encode()+value.tobytes())
    digest.update(np.asarray([model.opt.timestep, *model.opt.gravity], dtype=np.float64).tobytes())
    digest.update(np.asarray([model.opt.integrator, model.opt.solver, model.opt.cone,
                              model.opt.iterations, model.opt.tolerance], dtype=np.float64).tobytes())
    return digest.hexdigest()


def networks(torch, device, observation_size=OBSERVATION_SIZE):
    """Small Gaussian actor with tanh action transport; no recurrent clock."""
    nn = torch.nn

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, ACTION_SIZE))
            self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -1.5))
            nn.init.zeros_(self.body[-1].weight)
            nn.init.zeros_(self.body[-1].bias)

        def forward(self, obs):
            return self.body(obs), self.log_std.clamp(-4., .5).expand(obs.shape[0], -1)

    actor = Actor().to(device)
    critic = nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(),
                           nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
    return actor, critic


def log_probability(torch, raw, mean, log_std):
    # Store the pre-tanh sample. Its Gaussian density is sufficient for PPO's
    # likelihood ratio: the same bijective tanh Jacobian cancels in that ratio.
    return (-.5*((raw-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)).sum(-1)


def generalized_advantage(torch, rewards, values, bootstrap, gamma=.99, lam=.95):
    # Rollout boundaries truncate a continuing tracking task, so bootstrap the
    # actual final state. No episode reset transition is included in storage.
    result = torch.empty_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in range(rewards.shape[0]-1, -1, -1):
        delta = rewards[t] + gamma*next_value-values[t]
        carry = delta + gamma*lam*carry
        result[t] = carry
        next_value = values[t]
    return result, result+values


def joint_reward(torch, q, v, goal, controls, joint_scale, velocity_scale, effort_limit):
    tracking = ((q[:, :16]-goal)/joint_scale).square().mean(-1)
    speed = (v[:, :16]/velocity_scale).square().mean(-1)
    effort = (controls/effort_limit).square().mean(dim=(0, 2))
    return 1.-tracking-.002*speed-.001*effort


class DeviceTask:
    """Physics bank transfers use D2D copies; no per-step NumPy state bridge."""
    def __init__(self, torch, device, bank, model, q0, args, *, reach=None):
        self.torch, self.device, self.bank, self.args = torch, device, bank, args
        self.reach = reach
        self.worlds = args.worlds
        self.q0 = torch.tensor(q0, dtype=torch.float32, device=device)
        self.q = torch.empty((args.worlds, max(8, model.nq)), device=device, dtype=torch.float32)
        self.v = torch.empty((args.worlds, max(8, model.nv)), device=device, dtype=torch.float32)
        self.bias = torch.zeros_like(self.v)
        self.ctrl = torch.empty((args.worlds, max(1, model.nu)), device=device, dtype=torch.float32)
        from _panda_grasp_insert_control import ARM_KP, ARM_KD
        self.kp = torch.tensor(np.r_[ARM_KP, 1400., 1400.], dtype=torch.float32, device=device)
        self.kd = torch.tensor(np.r_[ARM_KD, 12., 12.], dtype=torch.float32, device=device)
        bounds = np.array(model.actuator_ctrlrange, copy=True)
        bounds[14:, 0] = np.maximum(bounds[14:, 0], -20.)
        bounds[14:, 1] = np.minimum(bounds[14:, 1], 20.)
        self.low_ctrl = torch.tensor(bounds[:, 0], device=device, dtype=torch.float32)
        self.high_ctrl = torch.tensor(bounds[:, 1], device=device, dtype=torch.float32)
        self.effort_limit = torch.maximum(self.low_ctrl.abs(), self.high_ctrl.abs())
        joints = [int(model.actuator_trnid[i, 0]) for i in range(16)]
        if (model.nq, model.nv, model.nu) != (23, 22, 16):
            raise ValueError("frozen full dual-Panda free-pin dimensions required")
        for i, joint in enumerate(joints):
            if (int(model.jnt_qposadr[joint]) != i or int(model.jnt_dofadr[joint]) != i
                    or not np.array_equal(model.actuator_gear[i], [1., 0., 0., 0., 0., 0.])):
                raise ValueError("ordered unit-gear joint effort motors required")
        limits = np.array(model.jnt_range[joints], copy=True)
        limits[:14, 0] += .001
        limits[:14, 1] -= .001
        self.low_q = torch.tensor(limits[:, 0], device=device, dtype=torch.float32)
        self.high_q = torch.tensor(limits[:, 1], device=device, dtype=torch.float32)
        self.joint_scale = torch.tensor([.20]*14+[.035]*2, device=device)
        self.velocity_scale = torch.tensor([2.]*14+[.20]*2, device=device)
        self.goal = self.q0[:16].expand(args.worlds, -1).clone()
        # Static observed joint goals, never a trajectory or time-dependent
        # phase: small independent offsets in both real Panda chains.
        if reach is None:
            self.goal[:, :14] += .08*(2*torch.rand((args.worlds, 14), device=device)-1)
            self.goal[:, 14:] = .012 + .018*torch.rand((args.worlds, 2), device=device)
        self.goal = torch.maximum(self.low_q, torch.minimum(self.high_q, self.goal))
        self.total_physics_steps = 0
        self.last_report = bank.report()
        self.reset()

    def _read(self, buffers):
        buffers["qpos"].copy_to_torch(self.q)
        buffers["qvel"].copy_to_torch(self.v)
        self.bias.zero_()
        if self.args.bias == "lagged" and "entry_bias" in buffers:
            # These are the *previous* physics input's bias forces, not the
            # current output's gravity. First/reset substep uses zero.
            buffers["entry_bias"].copy_to_torch(self.bias)

    def reset(self):
        self.bank.reset()
        self.last_report = self.bank.report()
        self._read(self.bank.buffers())
        self.reset_qpos = self.q.clone()
        if self.reach is not None:
            self.reach.reset(self.q[:, :23])
            self.reach_measurement = self.reach.measure(self.q[:, :23])
            self.reach_previous_qualified = self.reach_measurement["qualified"].clone()
            self.reach_dwell_steps = self.torch.zeros(self.worlds, dtype=self.torch.int32, device=self.device)
            self.reach_max_dwell_steps = self.reach_dwell_steps.clone()
            self.reach_dwell = self.torch.zeros(self.worlds, device=self.device)
            self.reach_max_dwell = self.reach_dwell.clone()
        self.target = self.q0[:16].expand(self.worlds, -1).clone()
        self.previous_action = self.torch.zeros((self.worlds, 16), device=self.device)
        self.finite = self.torch.ones((), dtype=self.torch.bool, device=self.device)
        return self.observe()

    def observe(self):
        # Fixed scales rather than rollout-dependent normalization. No hidden
        # tick, episode fraction, phase, curriculum label or random reset seed.
        task_features = ((self.goal-self.q[:, :16])/self.joint_scale if self.reach is None
                         else self.reach_measurement["features"])
        return self.torch.cat((self.q[:, :23], self.v[:, :22]/5., task_features,
                               (self.target-self.q[:, :16])/self.joint_scale,
                               self.previous_action), -1).detach()

    def step(self, action, *, motors_enabled=True):
        t = self.torch
        self.target[:, :14] += .02*action[:, :14]
        # Finger actions are absolute joint opening targets, not fake grasp
        # flags. Both actual independent finger motors receive PD efforts.
        self.target[:, 14:] = .0175*(action[:, 14:]+1.)
        self.target = t.maximum(self.low_q, t.minimum(self.high_q, self.target))
        controls, substep_q, substep_v, qualifications = [], [], [], []
        for _ in range(self.args.action_repeat):
            effort = self.kp*(self.target-self.q[:, :16])-self.kd*self.v[:, :16]
            if self.args.bias == "lagged":
                effort = effort + self.bias[:, :16]
            self.ctrl.copy_(t.maximum(self.low_ctrl, t.minimum(self.high_ctrl, effort)))
            if not motors_enabled:
                self.ctrl.zero_()
            self.finite.logical_and_(t.isfinite(self.ctrl).all())
            controls.append(self.ctrl.clone())
            self.bank.buffers()["ctrl"].copy_from_torch(self.ctrl)
            next_buffers = self.bank.step()
            self.total_physics_steps += 1
            self._read(next_buffers)
            self.finite.logical_and_(t.isfinite(self.q).all() & t.isfinite(self.v).all())
            if self.reach is not None:
                self.reach_measurement = self.reach.measure(self.q[:, :23])
                qualified = self.reach_measurement["qualified"]
                self.reach_dwell_steps = t.where(qualified & self.reach_previous_qualified,
                                                self.reach_dwell_steps+1, t.zeros_like(self.reach_dwell_steps))
                self.reach_previous_qualified = qualified.clone()
                self.reach_max_dwell_steps = t.maximum(self.reach_max_dwell_steps, self.reach_dwell_steps)
                substep_q.append(self.q.clone()); substep_v.append(self.v.clone())
                qualifications.append(qualified.clone())
        self.previous_action = action.detach().clone()
        controls = t.stack(controls)
        if self.reach is None:
            reward = joint_reward(t, self.q, self.v, self.goal, controls,
                                  self.joint_scale, self.velocity_scale, self.effort_limit)
        else:
            reward = self.reach.reward(self.q[:, :23], self.v[:, :22], controls,
                                      self.effort_limit, measurement=self.reach_measurement)
            self.reach_dwell = self.reach_dwell_steps.to(t.float32)*self.bank.timestep
            self.reach_max_dwell = self.reach_max_dwell_steps.to(t.float32)*self.bank.timestep
            self.substep_qpos, self.substep_qvel = t.stack(substep_q), t.stack(substep_v)
            self.substep_qualified = t.stack(qualifications)
        self.finite.logical_and_(t.isfinite(reward).all())
        self.last_report = self.bank.report()
        return self.observe(), reward.detach(), controls.detach()


def optimize(torch, actor, critic, optimizer, storage, args):
    advantages, returns = generalized_advantage(torch, storage["reward"], storage["value"], storage["bootstrap"])
    obs, raw = storage["obs"].flatten(0, 1), storage["raw_action"].flatten(0, 1)
    old_logp = storage["logp"].flatten()
    advantages = advantages.flatten()
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    returns = returns.flatten()
    metrics = []
    parameters = list(actor.parameters())+list(critic.parameters())
    for _ in range(args.epochs):
        order = torch.randperm(obs.shape[0], device=obs.device)
        for ids in order.split(args.minibatch):
            mean, log_std = actor(obs[ids])
            new_logp = log_probability(torch, raw[ids], mean, log_std)
            ratio = (new_logp-old_logp[ids]).exp()
            surrogate = torch.minimum(ratio*advantages[ids], ratio.clamp(.8, 1.2)*advantages[ids])
            value_loss = .5*(critic(obs[ids]).squeeze(-1)-returns[ids]).square().mean()
            entropy = (log_std+.5*math.log(2*math.pi*math.e)).sum(-1).mean()
            loss = -surrogate.mean()+value_loss-.001*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            metrics.append(torch.stack((loss.detach(), value_loss.detach(), entropy.detach(), norm.detach())))
    return torch.stack(metrics).detach(), advantages.detach(), returns.detach()


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _trace_npz(torch, path, values):
    arrays = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in values.items()}
    np.savez_compressed(path, **arrays)
    return _sha(path)


def paired_physics(torch, task, before_actor, after_actor, steps, output, label):
    """Equal actual initial state/goals; three independently reset trajectories."""
    traces = {}
    for name, actor, enabled in (("before", before_actor, True), ("after", after_actor, True),
                                  ("motors_off", after_actor, False)):
        obs = task.reset()
        q, v = [task.q.clone()], [task.v.clone()]
        actions, rewards, controls, substeps, qualified_steps = [], [], [], [], []
        with torch.no_grad():
            for _ in range(steps):
                mean, _ = actor(obs)
                action = mean.tanh()
                obs, reward, control = task.step(action, motors_enabled=enabled)
                q.append(task.q.clone()); v.append(task.v.clone())
                actions.append(action); rewards.append(reward); controls.append(control)
                if task.reach is not None:
                    substeps.append(task.substep_qpos)
                    qualified_steps.append(task.substep_qualified)
        traces[name] = dict(qpos=torch.stack(q), qvel=torch.stack(v), action=torch.stack(actions),
                            reward=torch.stack(rewards), controls=torch.stack(controls), goal=task.goal.clone())
        if task.reach is not None:
            traces[name].update({"final_reach_"+k: value.clone() for k, value in task.reach_measurement.items()})
            traces[name]["reach_max_qualified_dwell_s"] = task.reach_max_dwell.clone()
            traces[name]["reach_max_qualified_dwell_steps"] = task.reach_max_dwell_steps.clone()
            traces[name]["substep_qpos"] = torch.stack(substeps)
            traces[name]["substep_qualified"] = torch.stack(qualified_steps)
        if not bool(task.finite.cpu()):
            raise RuntimeError(f"nonfinite {name} paired physical trajectory")
    arrays = {f"{name}_{field}": tensor for name, trace in traces.items() for field, tensor in trace.items()}
    path = output / f"{label}-paired-physics.npz"
    digest = _trace_npz(torch, path, arrays)
    first = traces["before"]["qpos"][0]
    for name in ("after", "motors_off"):
        if not torch.equal(first, traces[name]["qpos"][0]) or not torch.equal(traces["before"]["qvel"][0], traces[name]["qvel"][0]):
            raise RuntimeError("paired evaluation reset did not reproduce the exact physical initial state")
    result = dict(file=path.name, sha256=digest,
                mean_return={name: float(trace["reward"].sum(0).mean().cpu()) for name, trace in traces.items()},
                after_arm_motion_rad=float((traces["after"]["qpos"][-1, :, :14]-first[:, :14]).abs().max().cpu()),
                motor_negative_q_difference=float((traces["after"]["qpos"][-1]-traces["motors_off"]["qpos"][-1]).abs().max().cpu()),
                physical_task_success=None)
    result["arm_motion_observed"] = result["after_arm_motion_rad"] > 1e-6
    result["motor_negative_distinguishable"] = result["motor_negative_q_difference"] > 1e-6
    if task.reach is not None:
        result["reach_only_qualified_worlds"] = {
            name: int((trace["reach_max_qualified_dwell_steps"] >= math.ceil(.10/task.bank.timestep)).sum().cpu()) for name, trace in traces.items()}
        result["reach_scope"] = "consecutive actual physical-step endpoint qualifications for .10s; not continuous collision proof, physical grasp or insertion"
    return result


def config(args, scene):
    return dict(model_sha256=scene.xml_sha256, task=args.task, worlds=args.worlds, seed=args.seed,
                action_repeat=args.action_repeat, steps=args.steps, epochs=args.epochs,
                minibatch=args.minibatch, learning_rate=args.learning_rate, bias=args.bias,
                timestep=scene.timestep, observation_size=98 if args.task == "reach" else OBSERVATION_SIZE,
                action_size=ACTION_SIZE,
                adam_execution="torch.optim.Adam noncapturable: parameter/moment math on MUSA; scalar step counters may be CPU and are reported separately",
                reward=REWARD_DESCRIPTION if args.task == "joint-track" else "see bound reach contract", gamma=.99, gae_lambda=.95,
                reset_semantics="whole-batch original physical initial state before every update; continuation bootstrap before reset",
                policy_input=("q23,v22/5,goal-minus-q16/scales,target-minus-q16/scales,previous-action16; no clock"
                              if args.task == "joint-track" else
                              "q23,v22/5,actual-reach-features21,target-minus-q16/scales,previous-action16; no clock"),
                action="tanh Gaussian; arm target +=.02rad*a; per-finger target=.0175m*(a+1)")


def require_musa():
    import torch
    import torch_musa  # noqa: F401 -- registers the actual MUSA torch backend
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("real torch.musa is required; no CUDA/CPU fallback")
    return torch


def physical_initial_states(scene, args):
    """Explicit physical declarations; none of their source metadata enters obs."""
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    q = np.repeat(np.asarray(scene.initial_qpos, dtype=np.float64)[None, :], args.worlds, axis=0)
    v = np.zeros((args.worlds, 22), dtype=np.float64)
    if args.reset_npz is not None:
        with np.load(args.reset_npz, allow_pickle=False) as data:
            if str(data["model_sha256"].item()) != scene.xml_sha256:
                raise ValueError("physical reset declarations belong to another model")
            q, v = np.array(data["qpos"], dtype=np.float64), np.array(data["qvel"], dtype=np.float64)
    elif args.stage == "evaluate":
        # A real, newly declared initial configuration, not observation noise.
        # The free part remains on its authored cradle; both seven-axis arms
        # get small independent joint offsets. Physics settles it itself.
        rng = np.random.default_rng(args.eval_seed)
        q[:, :14] += rng.uniform(-.025, .025, size=(args.worlds, 14))
    if q.shape != (args.worlds, 23) or v.shape != (args.worlds, 22) or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("reset declarations require finite W23/W22 states")
    digest = hashlib.sha256(q.tobytes()+v.tobytes()).hexdigest()
    return ({k: MuJoCoInitialState(q=q[k], v=v[k], time=0.) for k in range(args.worlds)}, digest,
            dict(qpos=q, qvel=v, model_sha256=scene.xml_sha256))


def audit_checkpoint(path, output):
    """CPU-only independent forward replay of both saved policy parameter sets."""
    import torch
    checkpoint = load_checkpoint(torch, path)
    rollout = Path(path).parent / checkpoint["rollout_file"]
    if _sha(rollout) != checkpoint["rollout_sha256"]:
        raise ValueError("physical rollout checksum differs")
    # Independent CPU math checks actual successor q/v/effort -> reward -> GAE.
    # No synthetic target loss can substitute for this recorded transition path.
    with np.load(rollout, allow_pickle=False) as data:
        scale = np.asarray([.20]*14+[.035]*2)
        velocity_scale = np.asarray([2.]*14+[.20]*2)
        if checkpoint["contract"]["task"] == "joint-track":
            expected_reward = (1.-np.mean(((data["next_qpos"][..., :16]-data["goal"])/scale)**2, axis=-1)
                               -.002*np.mean((data["next_qvel"][..., :16]/velocity_scale)**2, axis=-1)
                               -.001*np.mean((data["controls"]/data["effort_limit"])**2, axis=(1, 3)))
        else:
            import mujoco
            from _panda_ppo_reach import audit_numpy
            xml_path = Path(path).parent / "scene.xml"
            if _sha(xml_path) != checkpoint["contract"]["model_sha256"]:
                raise ValueError("CPU reach audit model XML hash differs")
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            if model_digest(model) != checkpoint["frozen_model_sha256"]:
                raise ValueError("CPU reach model arrays differ")
            measured = audit_numpy(model, data["reset_qpos"], data["next_qpos"],
                                   data["next_qvel"], data["controls"], data["effort_limit"])
            expected_reward = measured["reward"]
            for key, value in measured.items():
                if "reach_"+key in data:
                    np.testing.assert_allclose(data["reach_"+key], value, rtol=2e-4, atol=2e-5)
            qs, vs, controls = data["substep_qpos"], data["substep_qvel"], data["controls"]
            count, repeat, worlds = qs.shape[:3]
            every_step = audit_numpy(model, data["reset_qpos"], qs.reshape(count*repeat, worlds, 23),
                                     vs.reshape(count*repeat, worlds, 22),
                                     controls.reshape(count*repeat, 1, worlds, 16), data["effort_limit"])
            np.testing.assert_array_equal(data["substep_qualified"], every_step["qualified"].reshape(count, repeat, worlds))
        np.testing.assert_allclose(data["reward"], expected_reward, rtol=2e-5, atol=2e-5)
        recorded_reward = data["reward"].astype(np.float64)
        values = data["value"].astype(np.float64)
        advantage = np.empty_like(recorded_reward)
        physical_advantage = np.empty_like(recorded_reward)
        reward_error_bound = np.empty_like(recorded_reward)
        next_value = data["bootstrap"].astype(np.float64)
        carry = np.zeros_like(next_value)
        physical_carry = np.zeros_like(next_value)
        error_carry = np.zeros_like(next_value)
        for i in range(len(expected_reward)-1, -1, -1):
            carry = recorded_reward[i]+.99*next_value-values[i]+.99*.95*carry
            physical_carry = expected_reward[i]+.99*next_value-values[i]+.99*.95*physical_carry
            # Both recurrences use the same saved critic values and bootstrap.
            # Propagate measured CPU/GPU reward differences along the actual
            # horizon, after checking each physical reward independently.
            error_carry = np.abs(recorded_reward[i]-expected_reward[i])+.99*.95*error_carry
            advantage[i] = carry
            physical_advantage[i] = physical_carry
            reward_error_bound[i] = error_carry
            next_value = values[i]
        expected_returns = advantage+values
        gpu_returns = data["returns"].reshape(expected_returns.shape)
        np.testing.assert_allclose(gpu_returns, expected_returns, rtol=3e-5, atol=3e-5)
        physical_return_error = np.abs(gpu_returns-(physical_advantage+values))
        return_tolerance = 3e-5+3e-5*np.abs(expected_returns)
        if np.any(physical_return_error > reward_error_bound+return_tolerance):
            raise ValueError("physical return error exceeds propagated reward differences and GAE tolerance")
        flat = advantage.ravel()
        np.testing.assert_allclose(data["advantages"], (flat-flat.mean())/(flat.std()+1e-8), rtol=5e-5, atol=5e-5)
        physical_flat = physical_advantage.ravel()
        np.testing.assert_allclose(data["advantages"], (physical_flat-physical_flat.mean())/(physical_flat.std()+1e-8),
                                   rtol=5e-5, atol=5e-5)
        replay_errors = dict(max_successor_reward=float(np.max(np.abs(recorded_reward-expected_reward))),
                             max_recorded_reward_return=float(np.max(np.abs(gpu_returns-expected_returns))),
                             max_physical_return=float(physical_return_error.max()),
                             max_propagated_reward_difference=float(reward_error_bound.max()))
        if not np.isfinite(data["optimizer_metrics"]).all() or not np.any(data["optimizer_metrics"][:, 3] > 0.):
            raise ValueError("no finite nonzero optimizer gradient evidence")
    probe = checkpoint["action_probe"]
    results = {}
    for name in ("before", "after"):
        actor, _ = networks(torch, "cpu", checkpoint["contract"]["observation_size"])
        actor.load_state_dict(probe[name+"_actor"])
        with torch.no_grad():
            mean, log_std = actor(probe["observation"])
            action = (mean + log_std.exp()*probe["noise"]).tanh()
        delta = (action-probe[name+"_action"]).abs().max().item()
        if not torch.allclose(action, probe[name+"_action"], rtol=2e-5, atol=2e-5):
            raise ValueError(f"{name} CPU checkpoint forward disagrees with device probe: {delta}")
        results[name] = dict(max_action_difference=delta)
    if any(not torch.equal(checkpoint["actor"][k], probe["after_actor"][k]) for k in checkpoint["actor"]):
        raise ValueError("checkpoint actor differs from the post-update probe")
    result = dict(schema="panda-state-ppo-cpu-audit-v1", checkpoint_sha256=_sha(path),
                  comparisons=results, successor_reward_and_gae_verified=True, replay_errors=replay_errors,
                  scope="CPU forward/weights and saved successor reward/GAE; no CPU physical re-simulation")
    _write(output, result)
    return result


def run(args):
    if args.baseline_checkpoint is not None and args.stage != "evaluate":
        raise ValueError("--baseline-checkpoint applies only to independent evaluate")
    if args.stage == "audit":
        if args.checkpoint is None:
            raise ValueError("audit requires --checkpoint")
        return audit_checkpoint(args.checkpoint, args.out)
    if args.stage == "evaluate" and (args.eval_seed is None or args.eval_seed < 0 or args.eval_seed == args.seed):
        raise ValueError("evaluate requires a nonnegative --eval-seed distinct from the training --seed")
    torch = require_musa()
    from _panda_grasp_insert_scene import build_scene
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime

    torch.musa.set_device(args.device_index)
    if args.device_index != 0:
        raise ValueError("the physical runtime currently owns visible MUSA device 0; select hardware using MUSA_VISIBLE_DEVICES")
    device = torch.device(f"musa:{args.device_index}")
    name = torch.musa.get_device_name(args.device_index)
    if "S4000" not in name:
        raise RuntimeError(f"this qualification requires an actual S4000, found {name}")
    torch.manual_seed(args.seed)
    torch.musa.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    scene = build_scene(urdf_path=args.urdf)
    (args.out / "scene.xml").write_text(scene.xml, encoding="utf-8")
    source = scene.compile()
    frozen_model = model_digest(source.mj_model)
    contract = config(args, scene)
    initial_states, reset_sha, reset_arrays = physical_initial_states(scene, args)
    initial_path = args.out / "initial-states.npz"
    np.savez_compressed(initial_path, **reset_arrays)
    contract["physical_reset_sha256"] = reset_sha
    reach = None
    if args.task == "reach":
        from _panda_ppo_reach import ReachTask
        reach = ReachTask(torch, source.mj_model, device)
        contract["reach"] = reach.contract
        contract["reach_source_sha256"] = _sha(Path(__file__).with_name("_panda_ppo_reach.py"))
    _write(args.out / "config.json", dict(schema=SCHEMA, contract=contract, device=name,
                                          script_sha256=_sha(__file__), physical_task_success=None))
    actor, critic = networks(torch, device, contract["observation_size"])
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=args.learning_rate)
    checkpoint, start_update = None, 0
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(torch, args.checkpoint)
        bound_contract = copy.deepcopy(checkpoint.get("contract"))
        current_contract = copy.deepcopy(contract)
        if args.stage == "evaluate" and isinstance(bound_contract, dict):
            # Evaluation may explicitly supply a held-out *physical* initial
            # state; optimizer continuation may not silently change resets.
            bound_contract.pop("physical_reset_sha256", None)
            current_contract.pop("physical_reset_sha256", None)
        if checkpoint.get("schema") != SCHEMA or bound_contract != current_contract:
            raise ValueError("checkpoint task/model/optimizer/observation contract differs")
        if args.stage == "evaluate" and checkpoint["contract"]["physical_reset_sha256"] == reset_sha:
            raise ValueError("held-out evaluate must not reuse the training physical reset")
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["next_update"])
    if args.stage == "evaluate" and checkpoint is None:
        raise ValueError("evaluate requires --checkpoint")
    baseline_state, baseline_receipt = None, None
    if args.baseline_checkpoint is not None:
        baseline_state, baseline_receipt = initial_baseline(torch, args.baseline_checkpoint, checkpoint)
    keys = tuple(range(args.worlds))
    summary = dict(schema=SCHEMA, status="running", stage=args.stage, contract=contract, updates=[],
                   claim=f"real physical {args.task} PPO qualification; not learned grasp/insertion",
                   initial_states_file=initial_path.name, initial_states_sha256=_sha(initial_path))
    bank, task, rows = None, None, None
    try:
        with MuJoCoRuntime(source, mode="musa", world_keys=keys,
                           initial_states=initial_states, cache_root=args.cache,
                           contact_capacity=64, constraint_capacity=256) as runtime:
            with runtime.musa_resident() as bank:
                if bank.buffers()["qpos"].shape != (args.worlds, 23):
                    raise RuntimeError("resident bank dimensions differ from the declared task")
                if args.stage == "evaluate":
                    torch.manual_seed(args.eval_seed)
                    torch.musa.manual_seed_all(args.eval_seed)
                task = DeviceTask(torch, device, bank, source.mj_model, scene.initial_qpos, args, reach=reach)
                if checkpoint is not None and args.stage != "evaluate":
                    task.goal.copy_(checkpoint["goal"].to(device))
                    torch.set_rng_state(checkpoint["rng_cpu"])
                    torch.musa.set_rng_state_all(checkpoint["rng_musa"])
                summary["device_start"] = bank.report()
                if args.stage == "evaluate":
                    before_actor = copy.deepcopy(actor)
                    before_actor.load_state_dict(checkpoint["action_probe"]["before_actor"] if baseline_state is None else baseline_state)
                    summary["evaluation"] = paired_physics(torch, task, before_actor, actor, args.eval_steps, args.out, "heldout")
                    summary["evaluation"].update(
                        eval_seed=args.eval_seed, training_seed=checkpoint["contract"]["seed"],
                        training_reset_sha256=checkpoint["contract"]["physical_reset_sha256"],
                        evaluation_reset_sha256=reset_sha,
                        policy_roles=dict(before=("checkpoint last-update-before, not an untrained policy" if baseline_state is None else
                                                  "bound update-00000 pre-update initial policy"),
                                          after="loaded checkpoint policy", motors_off="loaded policy with all physical motors zero"),
                        baseline_source=baseline_receipt,
                        goals=task.goal.detach().cpu().tolist(),
                        goal_scope="new eval-seed joint targets" if args.task == "joint-track" else "actual measured reset-relative reach target; goal16 unused by reach reward/observation",
                        reset_recipe="explicit held-out reset NPZ" if args.reset_npz is not None else "home arm coordinates + independent Uniform[-.025,.025] rad; free part and fingers authored; zero velocity")
                else:
                    for update in range(start_update, start_update+args.updates):
                        before_actor = copy.deepcopy(actor).eval()
                        obs = task.reset()
                        rows = {k: [] for k in ("obs", "raw_action", "action", "logp", "value", "reward",
                                               "qpos", "qvel", "next_qpos", "next_qvel", "controls")}
                        if task.reach is not None:
                            rows.update({"reach_"+k: [] for k in task.reach_measurement})
                            rows.update(substep_qpos=[], substep_qvel=[], substep_qualified=[])
                        with torch.no_grad():
                            for _ in range(args.steps):
                                mean, log_std = actor(obs)
                                raw = mean + log_std.exp()*torch.randn_like(mean)
                                action = raw.tanh()
                                logp = log_probability(torch, raw, mean, log_std)
                                value = critic(obs).squeeze(-1)
                                q, v = task.q.clone(), task.v.clone()
                                next_obs, reward, controls = task.step(action)
                                for key, item in dict(obs=obs, raw_action=raw, action=action, logp=logp,
                                                      value=value, reward=reward, qpos=q, qvel=v,
                                                      next_qpos=task.q.clone(), next_qvel=task.v.clone(), controls=controls).items():
                                    rows[key].append(item.detach())
                                if task.reach is not None:
                                    for key, item in task.reach_measurement.items():
                                        rows["reach_"+key].append(item.detach())
                                    rows["substep_qpos"].append(task.substep_qpos)
                                    rows["substep_qvel"].append(task.substep_qvel)
                                    rows["substep_qualified"].append(task.substep_qualified)
                                obs = next_obs
                            storage = {key: torch.stack(value) for key, value in rows.items()}
                            storage["bootstrap"] = critic(obs).squeeze(-1)
                            probe_obs = obs.clone()
                            noise = torch.randn((args.worlds, 16), device=device)
                            old_mean, old_std = actor(probe_obs)
                            old_action = (old_mean + old_std.exp()*noise).tanh()
                        if not bool(task.finite.cpu()):
                            raise RuntimeError("nonfinite physical rollout; no optimizer update performed")
                        metrics, advantages, returns = optimize(torch, actor, critic, optimizer, storage, args)
                        gpu_stages = dict(actor=sorted({str(p.device) for p in actor.parameters()}),
                                          critic=sorted({str(p.device) for p in critic.parameters()}),
                                          rollout=str(storage["reward"].device), gae=str(returns.device),
                                          adam_moments=sorted({str(value.device) for state in optimizer.state.values()
                                                               for key, value in state.items() if key in ("exp_avg", "exp_avg_sq")}),
                                          adam_counters=sorted({str(state["step"].device) for state in optimizer.state.values()}))
                        if (any(not name.startswith("musa") for name in gpu_stages["actor"]+gpu_stages["critic"]+gpu_stages["adam_moments"])
                                or not gpu_stages["rollout"].startswith("musa") or not gpu_stages["gae"].startswith("musa")):
                            raise RuntimeError("a numerical learning stage left the real MUSA device")
                        with torch.no_grad():
                            new_mean, new_std = actor(probe_obs)
                            new_action = (new_mean + new_std.exp()*noise).tanh()
                            if not bool(torch.isfinite(metrics).all().cpu()):
                                raise RuntimeError("nonfinite PPO update")
                            # Same current physical input as probe_obs: the new
                            # action is actually sent to the motor-only scene.
                            probe_q, probe_v = task.q.clone(), task.v.clone()
                            _, probe_reward, probe_controls = task.step(new_action)
                        delta = float((new_action-old_action).abs().max().cpu())
                        changed = any(not torch.equal(v, before_actor.state_dict()[k]) for k, v in actor.state_dict().items())
                        if not changed or delta == 0.:
                            raise RuntimeError("PPO update did not change policy parameters and same-observation action")
                        label = f"update-{update:05d}"
                        storage.update(goal=task.goal.clone(), reset_qpos=task.reset_qpos.clone(), effort_limit=task.effort_limit.clone(),
                                       advantages=advantages, returns=returns,
                                       optimizer_metrics=metrics, probe_observation=probe_obs, probe_noise=noise,
                                       probe_before_action=old_action, probe_after_action=new_action,
                                       probe_input_qpos=probe_q, probe_input_qvel=probe_v,
                                       probe_next_qpos=task.q.clone(), probe_next_qvel=task.v.clone(),
                                       probe_reward=probe_reward, probe_controls=probe_controls)
                        trace_path = args.out / f"{label}-rollout.npz"
                        trace_sha = _trace_npz(torch, trace_path, storage)
                        paired = paired_physics(torch, task, before_actor, actor, args.eval_steps, args.out, label)
                        probe = dict(observation=probe_obs.cpu(), noise=noise.cpu(),
                                     before_action=old_action.cpu(), after_action=new_action.cpu(),
                                     before_actor=_cpu_state(before_actor), after_actor=_cpu_state(actor))
                        saved = dict(schema=SCHEMA, contract=contract, completed_update=update, next_update=update+1,
                                     actor=_cpu_state(actor), critic=_cpu_state(critic), optimizer=optimizer.state_dict(),
                                     rng_cpu=torch.get_rng_state(), rng_musa=torch.musa.get_rng_state_all(),
                                     goal=task.goal.cpu(), action_probe=probe, rollout_sha256=trace_sha,
                                     rollout_file=trace_path.name, frozen_model_sha256=frozen_model,
                                     physical_resume="next update begins with real whole-batch reset, not arbitrary mid-step restore")
                        checkpoint_path = args.out / f"{label}.pt"
                        validate_checkpoint_data(torch, saved)
                        torch.save(saved, checkpoint_path)
                        item = dict(update=update, rollout=trace_path.name, rollout_sha256=trace_sha,
                                    checkpoint=checkpoint_path.name, checkpoint_sha256=_sha(checkpoint_path),
                                    same_observation_action_max_delta=delta, physical_probe_sent=True,
                                    paired=paired, numerical_stage_devices=gpu_stages, device=bank.report())
                        summary["updates"].append(item)
                        _write(args.out / "progress.json", summary)
                        print(json.dumps(dict(update=update, action_delta=delta, paired=paired["mean_return"])), flush=True)
                summary["device_end"] = bank.report()
                begin, end = summary["device_start"], summary["device_end"]
                if (end["explicit_readback_bytes"] != begin["explicit_readback_bytes"]
                        or end["downloaded_bytes"]-begin["downloaded_bytes"] != end["status_downloaded_bytes"]-begin["status_downloaded_bytes"]):
                    raise RuntimeError("physics resident path downloaded data other than status; audit tensor copies are separately boundary-only")
                if (end["status_downloaded_bytes"]-begin["status_downloaded_bytes"] != 4*args.worlds*task.total_physics_steps
                        or end["resident_steps"]-begin["resident_steps"] != task.total_physics_steps
                        or end["accepted_world_steps"]-begin["accepted_world_steps"] != args.worlds*task.total_physics_steps):
                    raise RuntimeError("actual accepted physics/status counts differ from the control loop")
                if end["completed_kernels"] <= begin["completed_kernels"] or end["completed_kernels"] != end["launched_kernels"]:
                    raise RuntimeError("physical kernel launch/completion evidence is incomplete")
                summary["physical_steps_per_world"] = task.total_physics_steps
                summary["frozen_model_sha256"] = frozen_model
                if model_digest(source.mj_model) != frozen_model:
                    raise RuntimeError("physical model mutated during training/evaluation")
                summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if task is not None:
            summary["last_successful_boundary_report"] = task.last_report
            summary["physical_steps_per_world"] = task.total_physics_steps
            # Includes an accepted substep prefix even when the next substep
            # fails before a full control transition could enter PPO storage.
            failure = dict(last_observed_qpos=task.q, last_observed_qvel=task.v,
                           last_submitted_ctrl=task.ctrl, joint_target=task.target, goal=task.goal)
            if rows:
                failure.update({key: torch.stack(values) for key, values in rows.items() if values})
            try:
                summary["failure_trace_sha256"] = _trace_npz(torch, args.out / "failure-prefix.npz", failure)
            except BaseException as capture_error:
                summary["failure_trace_error"] = str(capture_error)
        _write(args.out / "summary.json", summary)
        raise
    _write(args.out / "summary.json", summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoke", "train", "evaluate", "audit"), default="smoke")
    p.add_argument("--task", choices=("joint-track", "reach"), default="joint-track")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--baseline-checkpoint", type=Path,
                   help="evaluate against same-training update-00000.pt action_probe.before_actor instead of last-update-before")
    p.add_argument("--cache", type=Path)
    p.add_argument("--urdf", type=Path)
    p.add_argument("--reset-npz", type=Path, help="optional physical W23 qpos/W22 qvel and model_sha256; held-out evaluation allowed")
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--eval-steps", type=int, default=16)
    p.add_argument("--action-repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=91001)
    p.add_argument("--eval-seed", type=int, help="required separate seed for held-out physical reset and joint goals; never restores training RNG")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--bias", choices=("none", "lagged"), default="lagged")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if any(getattr(args, key) <= 0 for key in ("worlds", "steps", "updates", "epochs", "minibatch", "eval_steps", "action_repeat")):
        raise SystemExit("positive rollout/update sizes required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.seed < 0:
        raise SystemExit("positive finite learning rate and nonnegative seed required")
    run(args)
