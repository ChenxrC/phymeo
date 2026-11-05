"""Coupled tangent/history use on two actual affine translating masses.

The normal loads are prescribed inputs. This does not replace a scene's normal
contact solve, discover material anchors, or enable robot-scene static friction.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import phymeo as ph


def run_window():
    h = 1 / 256
    masses = np.array([1.0, 2.0])
    model = ph.ParticleModel(1 / masses, np.zeros(3))
    state = ph.ParticleState(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]), np.zeros((2, 3)))
    solver = ph.SemiImplicitSolver()
    book = ph.TangentHistory(2, max_frame_turn_rad=0.1)
    history = book.initial_state()
    addresses = tuple(
        ph.MaterialAddress(
            ph.ContactOwner(0, body, "particle", "material_vertices"),
            ph.CollisionKey(0, body, 0),
            0,
            (node,),
            ((node, 1.0),),
            None,
        )
        for body, node in ((1, 0), (2, 1))
    )
    samples = tuple(
        ph.StationSample(
            ph.StationKey(0, token),
            *addresses,
            normal=(0.0, 0.0, 1.0),
            frame_world=np.eye(3),
            normal_force_n=2.0,
        )
        for token in (10, 20)
    )
    mobility = 1.5 * np.kron(np.ones((2, 2)), np.eye(2))
    stiffness = np.full(2, 100.0)
    friction = np.full(2, 0.5)
    frames = []
    max_momentum = max_motion = max_energy = 0.0
    reordered = sliding = 0
    rejected = stale = retry = False
    first_delta = None
    for index, external_force in enumerate([1.0] * 64 + [3.0] * 64 + [-1.0] * 64):
        before_history, before_state = history, state
        ordered = samples if index % 2 == 0 else samples[::-1]
        reordered += index % 2
        trial = book.prepare(history, ordered, sample_time_s=history.commit_time_s, dt_s=h)
        external = np.array([[-external_force, 0.0, 0.0], [external_force, 0.0, 0.0]])
        free_velocity = state.velocity + h * external / masses[:, None]
        free_relative = h * (free_velocity[1, :2] - free_velocity[0, :2])
        result = ph.solve_coupled_tangent(
            trial.elastic_tangent_m,
            np.tile(free_relative, (2, 1)),
            mobility,
            h,
            trial.normal_force_n,
            stiffness_n_per_m=stiffness,
            friction=friction,
        )
        if index == 10:
            bad = result.verify_motion(
                result.predicted_displacement_m + np.array([[0.001, 0.0], [0.001, 0.0]]),
                tolerance_m=1e-12,
            )
            rejected = not bad.accepted and history is before_history and state is before_state
            assert rejected
        pair_force = np.r_[result.force_n.sum(axis=0), 0.0]
        applied = external + np.array([-pair_force, pair_force])
        moved = solver.step(model, state, ph.StepInputs(applied), h)
        assert moved.report.accepted
        candidate = moved.state
        delta = candidate.position - state.position
        relative = delta[1, :2] - delta[0, :2]
        actual = np.tile(relative, (2, 1))
        verification = result.verify_motion(actual, tolerance_m=1e-12)
        assert verification.accepted
        updates = {}
        for row, station in enumerate(trial.stations):
            prior = station.prior
            updates[trial.keys[row]] = ph.TangentUpdate(
                elastic_displacement_m=trial.frames_world[row, :, :2]
                @ result.elastic_displacement_m[row],
                stored_energy_j=float(result.stored_energy_j[row]),
                plastic_slip_m=(0.0 if prior is None else prior.plastic_slip_m)
                + float(np.linalg.norm(result.plastic_increment_m[row])),
                tangent_work_j=(0.0 if prior is None else prior.tangent_work_j)
                + float(verification.work_j[row]),
                regime=result.regime[row],
                stiffness_n_per_m=float(stiffness[row]),
            )
        next_history = book.commit(history, trial, updates)
        if index == 10:
            repeated = book.prepare(history, ordered, sample_time_s=history.commit_time_s, dt_s=h)
            retry = next_history == book.commit(history, repeated, updates)
            assert retry
        if index == 0:
            first_delta = float(relative[0])
            try:
                book.commit(next_history, trial, updates)
            except ValueError:
                stale = True
            assert stale
        old_energy = sum(row.update.stored_energy_j for row in history.active)
        new_energy = sum(row.update.stored_energy_j for row in next_history.active)
        kinetic_before = float(0.5 * np.sum(masses[:, None] * state.velocity**2))
        kinetic_after = float(0.5 * np.sum(masses[:, None] * candidate.velocity**2))
        external_work = float(np.sum(external * delta))
        kinetic_loss = float(
            0.5 * np.sum(masses[:, None] * (candidate.velocity - state.velocity) ** 2)
        )
        balance = (
            kinetic_after
            - kinetic_before
            + new_energy
            - old_energy
            - external_work
            + float(result.plastic_dissipation_j.sum())
            + float(result.implicit_dissipation_j.sum())
            + float(result.released_energy_j.sum())
            + trial.dropped_spring_energy_j
            + kinetic_loss
        )
        momentum_error = float(
            np.linalg.norm(
                (masses[:, None] * (candidate.velocity - state.velocity)).sum(0)
                - h * external.sum(0)
            )
        )
        max_momentum = max(max_momentum, momentum_error)
        max_motion = max(max_motion, verification.maximum_error_m)
        max_energy = max(max_energy, abs(balance))
        sliding += any(regime == "slide" for regime in result.regime)
        frames.append(
            dict(
                step=index + 1,
                time_s=(index + 1) * h,
                external_pair_force_n=external_force,
                position_m=candidate.position.tolist(),
                velocity_m_s=candidate.velocity.tolist(),
                force_n=result.force_n.tolist(),
                displacement_m=actual.tolist(),
                elastic_displacement_m=result.elastic_displacement_m.tolist(),
                regime=list(result.regime),
                stored_energy_j=result.stored_energy_j.tolist(),
                plastic_dissipation_j=result.plastic_dissipation_j.tolist(),
                implicit_dissipation_j=result.implicit_dissipation_j.tolist(),
                tangent_work_j=verification.work_j.tolist(),
                energy_balance_error_j=balance,
                global_kkt_residual=result.residual,
            )
        )
        # Only after the real motion and history candidate both validate.
        state, history = candidate, next_history
    ending = book.prepare(history, [], sample_time_s=history.commit_time_s, dt_s=h)
    exited = solver.step(model, state, ph.StepInputs.zeros(2), h)
    assert exited.report.accepted
    next_history = book.commit(history, ending, {})
    # The contact-free end event shares a real accepted drift, never a clock-only commit.
    exit_delta = exited.state.position - state.position
    exit_velocity_error = exited.state.velocity - state.velocity
    exit_motion_error = float(np.max(np.linalg.norm(exit_delta - h * state.velocity, axis=1)))
    exit_momentum_error = float(np.linalg.norm((masses[:, None] * exit_velocity_error).sum(0)))
    exit_kinetic_change = float(
        0.5 * np.sum(masses[:, None] * (exited.state.velocity**2 - state.velocity**2))
    )
    exit_balance = (
        exit_kinetic_change
        + ending.dropped_spring_energy_j
        - sum(row.update.stored_energy_j for row in history.active)
    )
    assert np.max(np.abs(exit_velocity_error)) < 1e-12
    max_motion = max(max_motion, exit_motion_error)
    max_momentum = max(max_momentum, exit_momentum_error)
    max_energy = max(max_energy, abs(exit_balance))
    frames.append(
        dict(
            step=193,
            time_s=193 * h,
            external_pair_force_n=0.0,
            position_m=exited.state.position.tolist(),
            velocity_m_s=exited.state.velocity.tolist(),
            force_n=[],
            displacement_m=[],
            elastic_displacement_m=[],
            regime=[],
            stored_energy_j=[],
            plastic_dissipation_j=[],
            implicit_dissipation_j=[],
            tangent_work_j=[],
            energy_balance_error_j=exit_balance,
            global_kkt_residual=0.0,
            free_drift_error_m=exit_motion_error,
            momentum_error_ns=exit_momentum_error,
            kinetic_change_j=exit_kinetic_change,
        )
    )
    state, history = exited.state, next_history
    return dict(
        motion_domain="affine_translation_given_normal_load",
        tangent_backend="host",
        accepted_motion_steps=len(frames),
        history_steps=history.step_index,
        station_count=2,
        final_active_stations=len(history.active),
        stale_trial_rejected=stale,
        bad_motion_rejected_without_commit=rejected,
        same_input_retry_identical=retry,
        reordered_station_steps=reordered,
        sliding_steps=sliding,
        released_spring_energy_j=ending.dropped_spring_energy_j,
        maximum_momentum_error_ns=max_momentum,
        maximum_motion_error_m=max_motion,
        maximum_energy_balance_error_j=max_energy,
        first_relative_displacement_m=first_delta,
        frames=frames,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run_window()
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))
    if not (
        result["maximum_energy_balance_error_j"] < 1e-11
        and result["maximum_motion_error_m"] < 1e-12
        and result["maximum_momentum_error_ns"] < 1e-12
        and result["sliding_steps"] > 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
