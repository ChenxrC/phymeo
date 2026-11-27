"""Pickup rewards on measured geometry and contact loads from the same physics input.

Every step consumes one accepted physical solve. An action-repeat collector
sums these rewards. The actor never needs the reward's force or episode state.
The final independent evaluator is PickupLoads.score on the full trace.
"""

import math

import numpy as np

from _panda_pickup_loads import (
    BEARING_WEIGHT, DWELL_S, LIFT_M, MIN_NORMAL_WEIGHT, PARTS, SLIP_M, SOURCES, UNLOADED_WEIGHT,
)


class ResidentPickupLoads:
    """Device-to-device reads of accepted MUSA mixed-constraint contact buffers.

    Construct before or after reset. Call read only after a successful step,
    paired with that step's *input* geometry. Returns reusable [W,5,3]
    tensors: normal magnitude and signed force opposite gravity, ordered as in
    PickupLoads. Clone output to keep it across reads. No force is inferred
    from motion, collision depth, or motor torques.
    """

    def __init__(self, torch, model, bank, device):
        self.t, self.device, self.bank = torch, device, bank
        gravity = np.asarray(model.opt.gravity)
        magnitude = np.linalg.norm(gravity)
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("finite nonzero gravity required")
        self.up = torch.as_tensor(-gravity / magnitude, dtype=torch.float32, device=device)
        self.sources = {model.geom(name).id: idx for idx, name in enumerate(SOURCES[:-1])}
        self.parts = {model.geom(name).id: idx for idx, name in enumerate(PARTS)}
        if set(np.flatnonzero(model.geom_bodyid == model.body("assembly_part").id)) != set(self.parts):
            raise ValueError("missing free-part geometry from contact attribution")
        self.raw = None

    def _allocate(self, views):
        t, device = self.t, self.device
        worlds, capacity, _ = views["contact_map"].shape
        slots = np.asarray(views["slot_ids"])
        nslot = len(slots)
        if min(worlds, capacity, nslot) < 1 or slots.shape != (nslot, 4) or slots.dtype.kind not in "iu":
            raise ValueError("invalid compiled contact slot identities")
        specs = {"counts": ((worlds, 3), t.int32), "contact_map": ((worlds, capacity, 3), t.int32),
                 "contact_wrench": ((worlds, capacity, 6), t.float32), "normal_rows": ((worlds, nslot, 32), t.float32)}
        for name, (shape, _) in specs.items():
            expected = np.int32 if name in ("counts", "contact_map") else np.float32
            if tuple(views[name].shape) != shape or np.dtype(views[name].dtype) != np.dtype(expected):
                raise ValueError(f"unexpected accepted device contact buffer: {name}")
        # A zero sign marks non-part and internal contacts. The first frame
        # axis points geom1 -> geom2 and the local wrench acts on geom2.
        groups = np.zeros(nslot, dtype=np.int64)
        signs = np.zeros(nslot, dtype=np.float32)
        for i, (_, _, a, b) in enumerate(slots):
            if (a in self.parts) == (b in self.parts):
                continue
            part, other, sign = (b, a, 1.) if b in self.parts else (a, b, -1.)
            signs[i] = sign
            groups[i] = 3 * self.sources.get(other, 4) + self.parts[part]
        self.raw = {n: t.empty(shape, dtype=dtype, device=device) for n, (shape, dtype) in specs.items()}
        self.groups = t.as_tensor(groups, device=device)
        self.signs = t.as_tensor(signs, device=device)
        self.columns = t.arange(capacity, dtype=t.int32, device=device)[None]
        self.bins = t.arange(15, dtype=t.int64, device=device)
        self.slot = t.empty((worlds, capacity), dtype=t.int64, device=device)
        self.frame = t.empty((worlds, capacity, 9), dtype=t.float32, device=device)
        self.members = t.empty((worlds, capacity, 15), dtype=t.bool, device=device)
        self.weighted = t.empty_like(self.members, dtype=t.float32)
        self.normal = t.empty((worlds, 15), dtype=t.float32, device=device)
        self.vertical = t.empty_like(self.normal)
        self.force = t.empty((worlds, 15, 3), dtype=t.float32, device=device)
        self.force_axis = t.empty_like(self.normal)

    def read(self):
        views = self.bank.contact_buffers()
        if self.raw is None:
            self._allocate(views)
        for name, dest in self.raw.items():
            views[name].copy_to_torch(dest)
        t = self.t
        mapping, wrench = self.raw["contact_map"], self.raw["contact_wrench"]
        self.slot.copy_(mapping[:, :, 0]).clamp_(0, len(self.groups)-1)
        count = self.raw["counts"][:, 0]
        active = (self.columns < count[:, None]) & (mapping[:, :, 2] > 0)
        slot_ok = (mapping[:, :, 0] >= 0) & (mapping[:, :, 0] < len(self.groups))
        invalid = ((count < 0) | (count > self.slot.shape[1]) | (active & ~slot_ok).any(1)
                   | (self.raw["counts"][:, 1] != active.sum(1)))
        sign = self.signs[self.slot]
        used = active & slot_ok & (sign != 0)
        t.eq(self.groups[self.slot].unsqueeze(-1), self.bins, out=self.members)
        idx = self.slot.unsqueeze(-1).expand_as(self.frame)
        t.gather(self.raw["normal_rows"][:, :, 12:21], 1, idx, out=self.frame)
        world_up_in_contact = (self.frame.reshape(*self.slot.shape, 3, 3) * self.up).sum(-1)
        force_up = t.where(used, sign * (world_up_in_contact * wrench[:, :, :3]).sum(-1), 0.)
        normal = t.where(used, wrench[:, :, 0], 0.)
        # A fixed reduction over C avoids unordered floating-point atomics on
        # duplicate slots. Output layout is sources first, then part geometry.
        t.mul(normal.unsqueeze(-1), self.members, out=self.weighted)
        t.sum(self.weighted, dim=1, out=self.normal)
        t.mul(force_up.unsqueeze(-1), self.members, out=self.weighted)
        t.sum(self.weighted, dim=1, out=self.vertical)
        self.normal.masked_fill_(invalid[:, None], float("nan"))
        self.vertical.masked_fill_(invalid[:, None], float("nan"))
        self.used, self.sign, self.invalid = used, sign, invalid
        return self.normal.view(-1, 5, 3), self.vertical.view(-1, 5, 3)

    def world_force(self):
        """Full signed world force [W,5,3,3] from the last read, without another solver copy."""
        if self.raw is None or not hasattr(self, "used"):
            raise ValueError("read one accepted contact solve first")
        t = self.t
        frame = self.frame.view(*self.slot.shape, 3, 3)
        wrench = self.raw["contact_wrench"][:, :, :3]
        for axis in range(3):
            component = self.sign * (frame[..., axis] * wrench).sum(-1)
            component = t.where(self.used, component, 0.)
            t.mul(component.unsqueeze(-1), self.members, out=self.weighted)
            t.sum(self.weighted, dim=1, out=self.force_axis)
            self.force[:, :, axis].copy_(self.force_axis)
        self.force.masked_fill_(self.invalid[:, None, None], float("nan"))
        return self.force.view(-1, 5, 3, 3)


class PickupRLReward:
    """One batch of episodes on CPU or MUSA; reset all world rows together.

    reset accepts [W,3] as Tensor or NumPy. By default its device follows the
    input (NumPy selects CPU). step accepts data on that device or host arrays.
    Native rows use PickupLoads.sample, MUSA rows use the integrated receipt
    or ResidentPickupLoads. Both require unforced free-part dynamics.
    """

    def __init__(self, torch, weight_n, dt, up=(0., 0., 1.), *, device=None):
        self.t, self.requested_device = torch, device
        self.weight, self.dt = float(weight_n), float(dt)
        direction = np.asarray(up, dtype=float)
        if (not math.isfinite(self.weight) or self.weight <= 0. or not math.isfinite(self.dt) or self.dt <= 0.
                or direction.shape != (3,) or not np.isclose(np.linalg.norm(direction), 1., rtol=1e-6)):
            raise ValueError("positive weight/dt and an upward unit vector required")
        self.direction = direction.copy()
        self.needed_intervals = math.ceil(DWELL_S / self.dt - 1e-9)
        self.origin = None

    def reset(self, part_xyz):
        t = self.t
        p = t.as_tensor(part_xyz, device=self.requested_device, dtype=t.float32)
        if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] != 3:
            raise ValueError("expected [world,3] initial part positions")
        self.origin = p.detach().clone()
        self.up = t.as_tensor(self.direction, device=p.device, dtype=p.dtype)
        mask = t.zeros(len(p), device=p.device, dtype=t.bool)
        self.supported = mask.clone()
        self.grip = mask.clone()
        self.slipped = mask.clone()
        self.previous_carried = mask.clone()
        self.succeeded = mask.clone()
        self.valid_episode = ~mask
        self.intervals = t.zeros_like(mask, dtype=t.int32)
        self.grasp_local = t.zeros_like(p)
        self.grasp_wrist = t.zeros_like(p)

    def _channel(self, value, dims):
        x = self.t.as_tensor(value, device=self.origin.device, dtype=self.origin.dtype).detach()
        if tuple(x.shape) != (len(self.origin), *dims):
            raise ValueError(f"expected pickup measurement [world,{dims}], received {tuple(x.shape)}")
        return x

    def step(self, normal_n, up_n, part_xyz, wrist_xyz, part_in_hand_m, orientation_rad):
        """Return reward [W] and independent per-world terms at this solved input."""
        if self.origin is None:
            raise ValueError("reset the reward with the actual starting part positions")
        t, weight = self.t, self.weight
        n, up = self._channel(normal_n, (5, 3)), self._channel(up_n, (5, 3))
        p, wrist, local = (self._channel(value, (3,)) for value in (part_xyz, wrist_xyz, part_in_hand_m))
        orient = self._channel(orientation_rad, ())
        valid = (t.isfinite(n).all(dim=2).all(dim=1) & (n >= 0.).all(dim=2).all(dim=1)
                 & t.isfinite(up).all(dim=2).all(dim=1) & t.isfinite(p).all(dim=1)
                 & t.isfinite(wrist).all(dim=1) & t.isfinite(local).all(dim=1)
                 & t.isfinite(orient) & (orient >= 0.) & (orient <= math.pi+1e-5))
        self.valid_episode = self.valid_episode & valid & t.isfinite(self.origin).all(dim=1)
        delta = p - self.origin
        height = (delta * self.up).sum(-1)
        displacement = delta.square().sum(-1).sqrt()
        rail_up = up[:, 2:4].sum(dim=(1, 2))
        fork_support = rail_up >= BEARING_WEIGHT * weight
        self.supported = self.supported | fork_support
        handle_min = t.minimum(n[:, 0, 0], n[:, 1, 0])
        both = handle_min >= MIN_NORMAL_WEIGHT * weight
        acquire = (~self.grip & both & self.supported & (height.abs() <= SLIP_M))
        self.grasp_local = t.where(acquire[:, None], local, self.grasp_local)
        self.grasp_wrist = t.where(acquire[:, None], wrist, self.grasp_wrist)
        self.grip = (self.grip | acquire) & both
        slip = (local - self.grasp_local).square().sum(-1).sqrt()
        self.slipped = self.grip & ((self.slipped & ~acquire) | (slip > SLIP_M))
        hand_rise = ((wrist - self.grasp_wrist) * self.up).sum(-1)
        rails_clear = n[:, 2:4].sum(dim=(1, 2)) <= UNLOADED_WEIGHT * weight
        others_clear = n[:, 4].sum(-1) <= UNLOADED_WEIGHT * weight
        pads_up = up[:, :2].sum(dim=(1, 2))
        loading = (self.valid_episode & self.grip & ~self.slipped & rails_clear & others_clear
                   & (pads_up >= BEARING_WEIGHT * weight))
        carried = loading & (height >= LIFT_M) & (hand_rise >= LIFT_M-SLIP_M)
        self.intervals = t.where(carried & self.previous_carried, self.intervals + 1, 0)
        self.previous_carried = carried
        new_success = (self.intervals >= self.needed_intervals) & ~self.succeeded
        self.succeeded = (self.succeeded | new_success) & self.valid_episode

        # Grounded alignment and actual squeeze teach the precursor skills.
        # Neither term depends positively on part height. Unsupported motion
        # can only incur the disturbance cost while the carrying gate is shut.
        grounded = fork_support & (height.abs() <= SLIP_M) & (displacement <= .010)
        distance = local.square().sum(-1).sqrt()
        align_rate = .30 * t.exp(-distance/.025 - orient/.25) * grounded
        grip_rate = .25 * (handle_min/(.5*weight)).clamp(0., 1.) * grounded
        rise = (t.minimum(height, hand_rise)/LIFT_M).clamp(0., 1.)
        lift_rate = loading * (2.*rise + carried.to(p.dtype))
        disturbance_rate = (~loading) * (displacement/LIFT_M).clamp(0., 1.) * 2.
        align_reward, grip_reward, lift_reward = (rate * self.dt for rate in (align_rate, grip_rate, lift_rate))
        penalty = disturbance_rate * self.dt
        reward = align_reward + grip_reward + lift_reward - penalty + new_success.to(p.dtype)
        reward = t.where(self.valid_episode, reward, t.full_like(reward, float("nan")))
        info = dict(align_reward=align_reward, grip_reward=grip_reward, lift_reward=lift_reward,
                    disturbance_penalty=penalty, new_success=new_success, success=self.succeeded, carried=carried,
                    dwell_intervals=self.intervals, grip=self.grip, slipped=self.slipped, loading=loading,
                    height_m=height, wrist_rise_m=hand_rise, slip_m=slip, handle_min_normal_n=handle_min,
                    fork_up_n=rail_up, pad_up_n=pads_up, valid=self.valid_episode)
        return reward, info
