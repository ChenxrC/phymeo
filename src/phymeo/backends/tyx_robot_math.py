"""FP32 tyx robot mathematics with explicit FP64 host storage only."""

from dataclasses import replace

import numpy as np

from ..core.robot import RobotInputs, RobotModel, RobotState, validate_robot_step
from ..dynamics.robot_kinematics import RobotKinematics
from ..dynamics.rotation import quaternion_matrix
from .tyx_math import TyxMath, as_compute, as_index


class TyxRobotMath(TyxMath):
    def __init__(self, model, mode, *, cache_root=None):
        if not isinstance(model, RobotModel):
            raise TypeError("TyxRobotMath requires RobotModel")
        super().__init__(mode, cache_root=cache_root)
        try:
            self.model = replace(model)
            self.n = model.dof_count
            self.parents = as_index(model.parents, "parents")
            self.kinds = as_index(
                [0 if t == "revolute" else 1 for t in model.joint_types], "joint_types"
            )
            self.offsets = as_compute(model.joint_position, "joint_position")
            # Static construction data only. No candidate-dependent FK/J/M is
            # computed on the host or supplied to a device kernel as an oracle.
            self.zero = as_compute(quaternion_matrix(model.joint_orientation), "zero rotation")
            self.axes = as_compute(
                model.joint_axis / np.hypot.reduce(model.joint_axis, axis=1, keepdims=True),
                "joint_axis",
            )
            self.com = as_compute(model.com, "com")
            self.masses = as_compute(model.mass, "mass")
            self.inertia = as_compute(model.inertia_diagonal, "inertia")
            self.gravity = as_compute(model.gravity, "gravity")
        except BaseException:
            self.close()
            raise

    def _q(self, q):
        result = as_compute(q, "q")
        if result.shape != (self.n,):
            raise ValueError("q must match robot DOF count")
        return result

    def _geometry(self, q):
        from ._tyx_robot_kernels import robot_geometry

        q = self._q(q)
        return self.call(
            robot_geometry,
            self.n,
            self.parents,
            self.kinds,
            self.offsets,
            self.zero,
            self.axes,
            self.com,
            q,
            np.zeros((self.n, 3), np.float32),
            np.zeros((self.n, 3, 3), np.float32),
            np.zeros((self.n, 3), np.float32),
            np.zeros((self.n, 3), np.float32),
            np.zeros((self.n, 3, 3), np.float32),
            outputs=(8, 9, 10, 11),
        )

    def geometry(self, q):
        return RobotKinematics(*(array.astype(np.float64) for array in self._geometry(q)))

    def _jacobians(self, geometry, indices, local_points):
        from ._tyx_robot_kernels import robot_jacobians

        indices, local = (
            as_index(indices, "point indices"),
            as_compute(local_points, "local_points"),
        )
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.n):
            raise ValueError("point indices must be valid robot links")
        if local.shape != (len(indices), 3):
            raise ValueError("point attachments require one local 3D point per link index")
        self._check_open()
        linear = np.zeros((len(indices), 3, self.n), np.float32)
        if not len(indices):
            return linear, linear.copy()
        return self.call(
            robot_jacobians,
            self.n,
            len(indices),
            self.parents,
            self.kinds,
            *geometry[:3],
            indices,
            local,
            linear,
            np.zeros_like(linear),
            outputs=(9, 10),
        )

    def jacobians(self, q, indices, local_points):
        return tuple(
            array.astype(np.float64)
            for array in self._jacobians(self._geometry(q), indices, local_points)
        )

    def _com_jacobians(self, geometry):
        return self._jacobians(geometry, np.arange(self.n, dtype=np.int32), self.com)

    def mass(self, q):
        from ._tyx_robot_kernels import robot_mass

        geometry = self._geometry(q)
        linear, angular = self._com_jacobians(geometry)
        (result,) = self.call(
            robot_mass,
            self.n,
            self.masses,
            self.inertia,
            geometry[1],
            linear,
            angular,
            np.zeros((self.n, self.n), np.float32),
            outputs=(6,),
        )
        return result.astype(np.float64)

    def _project(self, jacobians, effort, force, torque):
        from ._tyx_robot_kernels import robot_project_loads

        effort, force, torque = (
            as_compute(value, name)
            for value, name in ((effort, "effort"), (force, "force"), (torque, "torque"))
        )
        if effort.shape != (self.n,) or force.shape != (self.n, 3) or torque.shape != force.shape:
            raise ValueError("generalized input shapes must match the robot")
        (result,) = self.call(
            robot_project_loads,
            self.n,
            *jacobians,
            effort,
            force,
            torque,
            np.zeros(self.n, np.float32),
            outputs=(6,),
        )
        return result.astype(np.float64)

    def bias(self, q, v):
        from ._tyx_robot_kernels import robot_inertial_loads

        velocity = as_compute(v, "v")
        if velocity.shape != (self.n,):
            raise ValueError("v must match robot DOF count")
        geometry = self._geometry(q)
        force, torque = self.call(
            robot_inertial_loads,
            self.n,
            self.parents,
            self.kinds,
            *geometry[:3],
            self.com,
            self.masses,
            self.inertia,
            self.gravity,
            velocity,
            *(np.zeros((self.n, 3), np.float32) for _ in range(6)),
            outputs=(15, 16),
        )
        return self._project(self._com_jacobians(geometry), np.zeros(self.n), force, torque)

    def generalized(self, q, effort, force, torque):
        return self._project(self._com_jacobians(self._geometry(q)), effort, force, torque)

    def solve(self, q, load):
        load = as_compute(load, "generalized load")
        if load.shape != (self.n,):
            raise ValueError("generalized load must match robot DOF count")
        return self.solve_spd(self.mass(q), load).astype(np.float64)

    def acceleration(self, q, v, inputs):
        if not isinstance(inputs, RobotInputs):
            raise TypeError("robot acceleration requires RobotInputs")
        validate_robot_step(self.model, RobotState(q, v), inputs)
        applied = self.generalized(q, inputs.effort, inputs.force, inputs.torque)
        load = self.axpy(applied, self.bias(q, v), -1.0)
        return self.solve(q, load)
