"""Endpoint body observations using the pinned solver's TYX kinematics.

One shared derived workspace; two published output banks. No model or
integration-state mirror, contact solve, CPU forward, or per-world dispatch.
"""

from types import SimpleNamespace

import tyx as tx


@tx.kernel(gradient="unsupported")
def _publish(
    pos: tx.Array[tx.f32], quat: tx.Array[tx.f32], com: tx.Array[tx.f32],
    subtree: tx.Array[tx.f32], velocity: tx.Array[tx.f32], roots: tx.Array[tx.i32],
    out_pos: tx.Array[tx.f32], out_quat: tx.Array[tx.f32], out_twist: tx.Array[tx.f32],
):
    for i in tx.range(out_pos.shape[0] // 3):
        world = i // roots.shape[0]
        body = i % roots.shape[0]
        root = world * roots.shape[0] + roots[body]
        dx = com[i * 3] - subtree[root * 3]
        dy = com[i * 3 + 1] - subtree[root * 3 + 1]
        dz = com[i * 3 + 2] - subtree[root * 3 + 2]
        wx = velocity[i * 6]
        wy = velocity[i * 6 + 1]
        wz = velocity[i * 6 + 2]
        out_twist[i * 6] = velocity[i * 6 + 3] + wy * dz - wz * dy
        out_twist[i * 6 + 1] = velocity[i * 6 + 4] + wz * dx - wx * dz
        out_twist[i * 6 + 2] = velocity[i * 6 + 5] + wx * dy - wy * dx
        out_twist[i * 6 + 3] = wx
        out_twist[i * 6 + 4] = wy
        out_twist[i * 6 + 5] = wz
        for c in range(3):
            out_pos[i * 3 + c] = pos[i * 3 + c]
        for c in range(4):
            out_quat[i * 4 + c] = quat[i * 4 + c]


@tx.kernel(gradient="unsupported")
def _preserve(mask: tx.Array[tx.bool1], old: tx.Array[tx.f32],
              output: tx.Array[tx.f32], width: tx.i32):
    for i in tx.range(output.shape[0]):
        if not mask[i // width]:
            output[i] = old[i]


class PreparedBodyObservations:
    """Version-pinned PhySolve frontend adaptation, not a new FK implementation."""

    field_names = ("body_position", "body_rotation_wxyz", "body_com_twist")

    def __init__(self, model, banks, device):
        from physolve_tyx._src import smooth

        self.model = model
        self.operations = (smooth.kinematics, smooth.com_pos, smooth.com_vel)
        # All written fields of the three pinned frontend operations are private
        # workspace. The only borrowed state is qpos/qvel/mocap, never writable here.
        outputs = (
            "xpos", "xquat", "xanchor", "xaxis", "xmat", "xipos", "ximat",
            "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "subtree_com",
            "cinert", "cdof", "cvel", "cdof_dot",
        )
        self.workspace = {name: tx.empty_like(getattr(banks[0], name)) for name in outputs}
        self.initial_workspace = tuple((self.workspace[name], getattr(banks[0], name)) for name in outputs)
        self.inputs = tuple(SimpleNamespace(
            nworld=bank.nworld, **self.workspace,
            **{name: getattr(bank, name) for name in ("qpos", "qvel", "mocap_pos", "mocap_quat")},
        ) for bank in banks)
        self.banks = tuple({name: tx.empty((banks[0].nworld, model.nbody, width), tx.f32, device=device)
                            for name, width in zip(self.field_names, (3, 4, 6))} for _ in banks)
        roots = model.body_rootid.view(tx.i32).flatten()

        def arguments(data):
            return tuple(getattr(data, name).view(tx.f32).flatten()
                         for name in ("xpos", "xquat", "xipos", "subtree_com", "cvel")) + (roots,)

        self.native_inputs = tuple(arguments(bank) for bank in banks)
        self.derived_inputs = arguments(self.inputs[0])
        self.flat_outputs = tuple(tuple(bank[name].flatten() for name in self.field_names)
                                  for bank in self.banks)
        self.kinematics_passes = self.publications = 0

    def prepare(self, stream, all_mask):
        # Warm derived operations with no writes to native Data; publish native
        # initial observations afterward. Runtime owns completion and failure cleanup.
        for target, source in self.initial_workspace:
            target.assign(source, stream=stream)
        self.initial_workspace = ()
        for index in range(2):
            self.update(index, stream, integrated=True)
            self.update(index, stream, integrated=False)
            self.preserve(index, all_mask, stream)
        self.kinematics_passes = self.publications = 0

    def update(self, index, stream, *, integrated):
        inputs = self.native_inputs[index]
        if integrated:
            for operation in self.operations:
                operation(self.model, self.inputs[index])
            self.kinematics_passes += 1
            inputs = self.derived_inputs
        _publish(*inputs, *self.flat_outputs[index], mode="cuda", stream=stream)
        self.publications += 1

    def copy(self, source, target, stream):
        for name in self.field_names:
            self.banks[target][name].assign(self.banks[source][name], stream=stream)

    def preserve(self, index, mask, stream):
        for name, target in zip(self.field_names, self.flat_outputs[index]):
            old = self.flat_outputs[1 - index][self.field_names.index(name)]
            width = target.size // self.inputs[index].nworld
            if target.size:
                _preserve(mask, old, target, width, mode="cuda", stream=stream)

    @property
    def allocated_bytes(self):
        return sum(array.size * 4 for bank in self.banks for array in bank.values()) + sum(
            array.view(tx.f32).size * 4 for array in self.workspace.values()
        )
