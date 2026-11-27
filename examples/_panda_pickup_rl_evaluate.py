"""Buffer deterministic resident evaluation, then score complete same-solve pickup traces."""
import numpy as np

from _panda_pickup_loads import PickupLoads


CHANNEL_SHAPES = dict(qpos=(23,), qvel=(22,), ctrl=(16,), motor_target=(16,), action=(16,),
                      normal_n=(5, 3), up_n=(5, 3), force_world_n=(5, 3, 3),
                      part_xyz=(3,), wrist_xyz=(3,), part_in_hand_m=(3,))
CONTACT_SOURCE = "musa_device_same_solve"


def _check_motor_model(model, loads):
    import mujoco as mj

    if (model.nq, model.nv, model.nu) != (23, 22, 16) or loads.part_dof != 16:
        raise ValueError("Expected dual Panda, sixteen motors and six trailing free-pin DOFs")
    joints = np.asarray(model.actuator_trnid)[:, 0]
    if (not np.all(model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT)
            or not np.array_equal(model.jnt_dofadr[joints], np.arange(16))
            or not np.all(np.isin(model.jnt_type[joints],
                                  [mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE]))):
        raise ValueError("The resident motor-only evaluation requires each motor on its own non-part DOF")


class ResidentPickupTrace:
    """One uninterrupted rollout; instantiate after reset and finish inside its resident context.

    append follows a successful bank.step(). Geometry, time, q/v and submitted
    controls belong to that step's input; all three load arrays come from the
    just-accepted contact solve. Input tensors are copied before views are reused.
    """
    def __init__(self, model, bank, steps):
        from phymeo.backends.mujoco_musa_resident import MusaResidentSession

        if not isinstance(bank, MusaResidentSession) or steps < 2 or bank.timestep != model.opt.timestep:
            raise ValueError("A live MUSA resident session and at least two physics steps are required")
        self.loads = PickupLoads(model)
        _check_motor_model(model, self.loads)
        self.bank = bank
        self.begin = bank.report()
        self.world_keys = tuple(bank.world_keys)
        self.steps, self.index = steps, 0
        self.channels = {}

    def append(self, input_tick, entry_time, qpos, qvel, ctrl, target, action, normal_n, up_n,
               force_world_n, part_xyz, wrist_xyz, part_in_hand_m):
        import torch

        if self.index >= self.steps or input_tick != self.begin["episode_steps"] + self.index:
            raise ValueError("Incomplete or out-of-order resident physics input ticks")
        n = len(self.world_keys)
        channels = dict(qpos=qpos, qvel=qvel, ctrl=ctrl, motor_target=target, action=action,
                        normal_n=normal_n, up_n=up_n, force_world_n=force_world_n,
                        part_xyz=part_xyz, wrist_xyz=wrist_xyz, part_in_hand_m=part_in_hand_m)
        if (not isinstance(entry_time, torch.Tensor) or entry_time.shape not in ((n,), (n, 2))
                or entry_time.device != qpos.device):
            raise ValueError("Expected the input clock tensor or the native compensated clock on the state device")
        channels["clock"] = entry_time
        for name, value in channels.items():
            if (not isinstance(value, torch.Tensor) or value.device != qpos.device
                    or value.device.type != "musa" or value.dtype != torch.float32
                    or (name != "clock" and tuple(value.shape) != (n, *CHANNEL_SHAPES[name]))):
                raise ValueError(f"Invalid resident evaluation channel {name}")
            if self.index == 0:
                self.channels[name] = value.new_empty((self.steps, *value.shape))
            if self.channels[name].shape[1:] != value.shape:
                raise ValueError(f"Resident channel shape changed at {name}")
            self.channels[name][self.index].copy_(value.detach())
        self.index += 1

    def finish(self):
        end = self.bank.report()
        n = len(self.world_keys)
        expected = dict(resident_steps=self.steps, accepted_world_steps=self.steps*n,
                        resident_resets=0, failed_batches=0)
        if self.index != self.steps or any(end[k] - self.begin[k] != v for k, v in expected.items()):
            raise ValueError("Evaluation trace does not cover exactly the accepted resident rollout")
        arrays = {k: v.cpu().numpy() for k, v in self.channels.items()}
        return _score_resident_arrays(self.loads, arrays, self.begin["episode_steps"])


def _score_resident_arrays(loads, arrays, first_tick):
    """Score the contiguous inputs collected by the verified resident motor-only session."""
    clock = arrays["clock"].astype(np.float64)
    time = clock[..., 0] - clock[..., 1] if clock.ndim == 3 else clock
    steps, worlds = time.shape
    ticks = first_tick + np.arange(steps, dtype=np.int64)
    traces, scores = [], []
    for w in range(worlds):
        trace = {k: arrays[k][:, w] for k in CHANNEL_SHAPES}
        if (not np.isfinite(trace["motor_target"]).all() or not np.isfinite(trace["action"]).all()
                or np.any(np.abs(trace["action"]) > 1.)):
            raise ValueError("Invalid resident target or normalized motor action")
        trace.update(time_s=time[:, w], input_tick=ticks.copy(), contact_solver_code=np.full(steps, 2, np.int8),
                     finger_opening_m=trace["qpos"][:, loads.jaws],
                     # MusaResidentSession._start zeros the body-force and generalized applied-force banks.
                     # It only exposes writable ctrl; _check_motor_model excludes any actuator on the pin.
                     part_noncontact_applied=np.zeros((steps, 18), dtype=np.float32))
        score = loads.score(trace, contact_source=CONTACT_SOURCE)
        score["noncontact_evidence"] = ("Resident session zeroed applied generalized effort and body wrenches; "
                                       "compiled joint transmissions act only on the sixteen non-part DOFs.")
        traces.append(trace)
        scores.append(score)
    return traces, scores
