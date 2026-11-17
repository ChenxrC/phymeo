"""Solve-time body measurements for mounted IMU and internal force/torque truth."""

import tyx as tx


@tx.kernel(gradient="unsupported")
def _publish(com: tx.Array[tx.f32], subtree: tx.Array[tx.f32],
             velocity: tx.Array[tx.f32], acceleration: tx.Array[tx.f32],
             internal: tx.Array[tx.f32], roots: tx.Array[tx.i32],
             old: tx.ndarray(tx.f32, 2), output: tx.ndarray(tx.f32, 2),
             old_time: tx.Array[tx.f32], time: tx.Array[tx.f32],
             mask: tx.Array[tx.bool1], flags: tx.Array[tx.i32]):
    for i in tx.range(output.shape[0]):
        world = i // roots.shape[0]
        body = i % roots.shape[0]
        if mask[world]:
            root = world * roots.shape[0] + roots[body]
            point = tx.vec3f(com[i*3], com[i*3+1], com[i*3+2])
            offset = point - tx.vec3f(subtree[root*3], subtree[root*3+1], subtree[root*3+2])
            omega = tx.vec3f(velocity[i*6], velocity[i*6+1], velocity[i*6+2])
            linear = tx.vec3f(velocity[i*6+3], velocity[i*6+4], velocity[i*6+5]) + tx.cross(omega, offset)
            alpha = tx.vec3f(acceleration[i*6], acceleration[i*6+1], acceleration[i*6+2])
            specific = tx.vec3f(acceleration[i*6+3], acceleration[i*6+4], acceleration[i*6+5]) + tx.cross(alpha, offset) + tx.cross(omega, linear)
            force = tx.vec3f(internal[i*6+3], internal[i*6+4], internal[i*6+5])
            torque = tx.vec3f(internal[i*6], internal[i*6+1], internal[i*6+2]) - tx.cross(offset, force)
            for c in range(3):
                output[i, c] = point[c]
                output[i, c+3] = specific[c]
                output[i, c+6] = alpha[c]
                output[i, c+9] = linear[c]
                output[i, c+12] = omega[c]
                output[i, c+15] = force[c]
                output[i, c+18] = torque[c]
            for c in range(21):
                if not tx.isfinite(output[i, c]):
                    tx.atomic_or(flags[0], 1048576)
            if body == 0 and not tx.isfinite(time[world]):
                tx.atomic_or(flags[0], 1048576)
        else:
            for c in range(21):
                output[i, c] = old[i, c]
            if body == 0:
                time[world] = old_time[world]


class PreparedMeasurements:
    """One RNE workspace and two derived output banks; no solver state mirror."""

    # start/end, unit, frame are the single source for RuntimeField metadata.
    fields = {
        'measured_body_com_position': (0, 3, 'm', 'world; body COM'),
        'measured_body_specific_acceleration': (3, 6, 'm/s^2', 'world; linear specific acceleration at body COM'),
        'measured_body_angular_acceleration': (6, 9, 'rad/s^2', 'world'),
        'measured_body_com_twist': (9, 15, 'm/s,rad/s', 'world; linear at COM then angular'),
        'measured_body_internal_wrench': (15, 21, 'N,N*m', 'world; force then torque at body COM; parent acting on child'),
    }

    def __init__(self, model, banks, device, flags):
        from physolve_tyx import PostConstraintWorkspace

        if PostConstraintWorkspace.contract_version != 1:
            raise RuntimeError('PhySolve postconstraint workspace contract v1 required')
        self.workspace = PostConstraintWorkspace(model, banks)
        self.native, self.flags = banks, flags
        self.nbody, self.nworld = model.nbody, banks[0].nworld
        self.raw = tuple(tx.empty((self.nworld, self.nbody, 21), tx.f32, device=device) for _ in banks)
        self.flat = tuple(a.reshape((self.nworld*self.nbody, 21)) for a in self.raw)
        self.times = tuple(tx.empty(self.nworld, tx.f32, device=device) for _ in banks)
        self.banks = tuple({**{name: a[:, :, start:end] for name, (start, end, _, _) in self.fields.items()},
                            'measurement_sample_time': time} for a, time in zip(self.raw, self.times))
        self.inputs = tuple(tuple(getattr(data, name).view(tx.f32).flatten()
                                  for name in ('xipos', 'subtree_com', 'cvel')) for data in banks)
        self.derived = tuple(getattr(self.workspace, name).view(tx.f32).flatten()
                             for name in ('cacc', 'cfrc_int'))
        self.roots = model.body_rootid.view(tx.i32).flatten()
        self.evaluations = self.copies = 0

    def capture(self, index, stream):
        self.workspace.capture_velocity(index)
        self.times[index].assign(self.native[index].time, stream=stream)

    def prepare(self, stream, all_mask):
        self.all_mask = all_mask
        for index in (0, 1):
            self.update(index, stream)
        self.update(0, stream, mask=all_mask)
        self.evaluations = self.copies = 0

    def update(self, index, stream, *, integrated=False, mask=None, record=True):
        if not integrated:
            self.capture(index, stream)
        self.workspace.evaluate(index)
        _publish(*self.inputs[index], *self.derived, self.roots,
                 self.flat[1-index], self.flat[index], self.times[1-index], self.times[index],
                 self.all_mask if mask is None else mask, self.flags, mode='cuda', stream=stream)
        if record:
            self.evaluations += 1

    def copy(self, source, target, stream):
        self.raw[target].assign(self.raw[source], stream=stream)
        self.times[target].assign(self.times[source], stream=stream)
        self.copies += 1

    @property
    def allocated_bytes(self):
        return self.workspace.allocated_bytes + sum(a.size*4 for a in (*self.raw, *self.times))
