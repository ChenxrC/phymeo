"""Real CUDA graph execution, result equivalence, and resource lifetime boundaries."""

import numpy as np
import pytest

from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    Plane,
    StepInputs,
    TyxXPBDSolver,
)


@pytest.mark.parametrize("value", [None, 0, 1, "cuda", np.bool_(True)])
def test_graph_flag_is_strict_bool_before_optional_runtime_import(value):
    with pytest.raises(ValueError, match="cuda_graph.*bool"):
        TyxXPBDSolver(cuda_graph=value)


@pytest.mark.parametrize("mode", ["interpret", "cpu_cpp", "cpu_llvm"])
def test_graph_requires_explicit_cuda_mode(mode):
    with pytest.raises(ValueError, match="cuda_graph.*cuda"):
        TyxXPBDSolver(mode=mode, cuda_graph=True)


@pytest.fixture(scope="module")
def graph_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("cuda-graph-xpbd")


@pytest.fixture
def cuda_backend(tyx_mode):
    if tyx_mode != "cuda":
        pytest.skip("select --tyx-mode cuda for actual CUDA graph execution")
    import tyx

    return tyx


def scene():
    model = ParticleModel(np.array([1, 3, 1], np.float32), np.zeros(3))
    state = ParticleState(
        np.array([[0, 0.2, 0], [2, 0.2, 0], [4, 0.2, 0]], np.float32),
        np.zeros((3, 3), np.float32),
    )
    edges = DistanceConstraints(
        np.array([[0, 1], [1, 2]]), np.ones(2, np.float32), np.full(2, 0.01, np.float32)
    )
    return model, state, StepInputs.zeros(3, dtype=np.float32), edges


def track_resources(tx, monkeypatch):
    """Wrap real public resources; never substitute a fake GPU execution result."""
    events, arrays, graphs, streams = [], [], [], []
    create_array, create_graph, create_stream = tx.cuda_array, tx.Graph, tx.cuda_stream

    def array(*args, **kwargs):
        result = create_array(*args, **kwargs)
        free = result.free

        def release():
            events.append(("free", id(result)))
            free()

        result.free = release
        arrays.append(result)
        return result

    def stream(*args, **kwargs):
        result = create_stream(*args, **kwargs)
        sync, destroy = result.synchronize, result.destroy

        def synchronize():
            sync()
            events.append(("synchronize", id(result)))

        def release():
            events.append(("stream_destroy", id(result)))
            destroy()

        result.synchronize, result.destroy = synchronize, release
        streams.append(result)
        return result

    def graph(*args, **kwargs):
        assert kwargs["mode"] == "cuda" and kwargs["optimize"] is False
        result = create_graph(*args, **kwargs)
        replay, destroy = result.replay, result.destroy

        def launch(*args, **kwargs):
            assert result.backend == "cuda" and result.node_count > 0
            events.append(("replay", id(result)))
            return replay(*args, **kwargs)

        def release():
            events.append(("graph_destroy", id(result)))
            destroy()

        result.replay, result.destroy = launch, release
        graphs.append(result)
        return result

    monkeypatch.setattr(tx, "cuda_array", array)
    monkeypatch.setattr(tx, "cuda_stream", stream)
    monkeypatch.setattr(tx, "Graph", graph)
    return events, arrays, graphs, streams


def assert_released(tracked):
    events, arrays, graphs, streams = tracked
    assert arrays
    assert {id(array) for array in arrays} == {
        identity for event, identity in events if event == "free"
    }
    assert {id(graph) for graph in graphs} == {
        identity for event, identity in events if event == "graph_destroy"
    }
    assert {id(stream) for stream in streams} == {
        identity for event, identity in events if event == "stream_destroy"
    }
    names = [event for event, _ in events]
    if graphs:
        assert names.index("synchronize") < names.index("graph_destroy") < names.index("free")


@pytest.mark.tyx
@pytest.mark.cuda
class TestCudaGraph:
    @pytest.mark.parametrize("kind", ["distance", "plane", "both"])
    @pytest.mark.parametrize("iterations", [1, 100])
    def test_results_match_direct_launch_with_changed_inputs(
        self, cuda_backend, graph_cache, monkeypatch, kind, iterations
    ):
        model, state, inputs, edges = scene()
        edges = None if kind == "plane" else edges
        plane = None if kind == "distance" else Plane()
        options = dict(mode="cuda", iterations=iterations, plane=plane, cache_root=graph_cache)
        with (
            TyxXPBDSolver(edges, **options) as direct,
            TyxXPBDSolver(edges, cuda_graph=True, **options) as captured,
        ):
            assert direct.problem_signature() == captured.problem_signature()
            assert direct.capabilities == captured.capabilities
            for index, dt in enumerate([0.1, 0.02, 0.05]):
                model = ParticleModel(model.inverse_mass, np.array([0, -index - 1, 0], np.float32))
                inputs.force[:, 0] = index * 0.5
                expected = direct.step(model, state, inputs, dt)
                before = state.copy()
                with monkeypatch.context() as scope:
                    tracked = track_resources(cuda_backend, scope)
                    actual = captured.step(model, state, inputs, dt)
                    assert (
                        sum(event == "replay" for event, _ in tracked[0])
                        == actual.report.iterations
                    )
                    assert_released(tracked)
                assert actual.report == expected.report
                np.testing.assert_array_equal(actual.state.position, expected.state.position)
                np.testing.assert_array_equal(actual.state.velocity, expected.state.velocity)
                np.testing.assert_array_equal(state.position, before.position)
                np.testing.assert_array_equal(state.velocity, before.velocity)
                state = actual.state

    def test_single_hard_edge_has_analytic_solution(self, cuda_backend, graph_cache):
        model = ParticleModel(np.array([1, 3], np.float32), np.zeros(3))
        state = ParticleState(
            np.array([[0, 0, 0], [2, 0, 0]], np.float32), np.zeros((2, 3), np.float32)
        )
        edges = DistanceConstraints(
            np.array([[0, 1]]), np.ones(1, np.float32), np.zeros(1, np.float32)
        )
        with TyxXPBDSolver(edges, mode="cuda", cuda_graph=True, cache_root=graph_cache) as solver:
            result = solver.step(model, state, StepInputs.zeros(2, dtype=np.float32), 0.25)
        assert result.report.accepted and result.report.iterations == 1
        np.testing.assert_allclose(result.state.position[:, 0], [0.25, 1.25], atol=1e-6)
        np.testing.assert_allclose(result.state.velocity[:, 0], [1, -3], atol=1e-6)

    @pytest.mark.parametrize("count", [0, 3])
    def test_free_or_empty_scene_never_captures_empty_graph(
        self, cuda_backend, graph_cache, monkeypatch, count
    ):
        def unexpected(**kwargs):
            raise AssertionError("empty graph capture")

        monkeypatch.setattr(cuda_backend, "Graph", unexpected)
        model = ParticleModel(np.ones(count, np.float32), np.zeros(3))
        state = ParticleState(np.zeros((count, 3), np.float32), np.ones((count, 3), np.float32))
        with TyxXPBDSolver(mode="cuda", cuda_graph=True, cache_root=graph_cache) as solver:
            result = solver.step(model, state, StepInputs.zeros(count, dtype=np.float32), 0.1)
        assert result.report.accepted
        np.testing.assert_allclose(result.state.position, 0.1)

    @pytest.mark.parametrize("failure", ["capture", "replay", "readback"])
    def test_exception_releases_graph_before_arrays(
        self, cuda_backend, graph_cache, monkeypatch, failure
    ):
        model, state, inputs, edges = scene()
        before = state.copy()
        tracked = track_resources(cuda_backend, monkeypatch)
        with TyxXPBDSolver(edges, mode="cuda", cuda_graph=True, cache_root=graph_cache) as solver:
            if failure == "capture":
                project = solver._project

                def fail_capture(*args, **kwargs):
                    project(*args, **kwargs)
                    raise RuntimeError("injected capture failure")

                monkeypatch.setattr(solver, "_project", fail_capture)
            elif failure == "replay":
                factory = cuda_backend.Graph

                def fail_replay(**kwargs):
                    graph = factory(**kwargs)
                    replay = graph.replay

                    def launch():
                        replay()
                        raise RuntimeError("injected replay failure")

                    graph.replay = launch
                    return graph

                monkeypatch.setattr(cuda_backend, "Graph", fail_replay)
            else:
                read = cuda_backend.CudaArray.to_numpy
                calls = 0

                def fail_read(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    result = read(*args, **kwargs)
                    if calls == 3:
                        raise RuntimeError("injected readback failure")
                    return result

                monkeypatch.setattr(cuda_backend.CudaArray, "to_numpy", fail_read)
            with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
                solver.step(model, state, inputs, 0.1)
        assert_released(tracked)
        np.testing.assert_array_equal(state.position, before.position)
        np.testing.assert_array_equal(state.velocity, before.velocity)

    def test_fixed_inconsistent_constraint_is_rejected_and_resources_released(
        self, cuda_backend, graph_cache, monkeypatch
    ):
        _, state, inputs, edges = scene()
        edges = DistanceConstraints(edges.indices, edges.rest_length, np.zeros(2, np.float32))
        model = ParticleModel(np.zeros(3, np.float32), np.zeros(3))
        tracked = track_resources(cuda_backend, monkeypatch)
        with TyxXPBDSolver(
            edges, mode="cuda", cuda_graph=True, iterations=1, cache_root=graph_cache
        ) as solver:
            result = solver.step(model, state, inputs, 0.1)
        assert not result.report.accepted
        assert result.report.max_residual > 0
        assert_released(tracked)

    @pytest.mark.parametrize("stage", ["stream", "graph"])
    def test_resource_construction_failure_releases_prior_resources(
        self, cuda_backend, graph_cache, monkeypatch, stage
    ):
        model, state, inputs, edges = scene()
        tracked = track_resources(cuda_backend, monkeypatch)

        def unavailable(**kwargs):
            raise RuntimeError(f"injected {stage} unavailable")

        monkeypatch.setattr(
            cuda_backend, "cuda_stream" if stage == "stream" else "Graph", unavailable
        )
        with TyxXPBDSolver(edges, mode="cuda", cuda_graph=True, cache_root=graph_cache) as solver:
            with pytest.raises(RuntimeError, match=f"injected {stage} unavailable"):
                solver.step(model, state, inputs, 0.1)
        assert_released(tracked)

    def test_nonfinite_projection_rejects_owned_snapshot(
        self, cuda_backend, graph_cache, monkeypatch
    ):
        model, state, inputs, edges = scene()
        state.position[:, 0] = [-2e38, 2e38, 2e38]
        before = state.copy()
        tracked = track_resources(cuda_backend, monkeypatch)
        with TyxXPBDSolver(edges, mode="cuda", cuda_graph=True, cache_root=graph_cache) as solver:
            result = solver.step(model, state, inputs, 0.1)
        assert not result.report.accepted
        assert "non-finite" in result.report.reason
        np.testing.assert_array_equal(result.state.position, before.position)
        assert not np.shares_memory(result.state.position, state.position)
        assert_released(tracked)

    def test_nonfinite_prediction_is_rejected_before_graph_capture(
        self, cuda_backend, graph_cache, monkeypatch
    ):
        model, state, inputs, edges = scene()
        state.velocity[:] = -2e38

        def unexpected(**kwargs):
            raise AssertionError("invalid prediction must not enter graph capture")

        monkeypatch.setattr(cuda_backend, "Graph", unexpected)
        with TyxXPBDSolver(
            edges, mode="cuda", cuda_graph=True, plane=Plane(), cache_root=graph_cache
        ) as solver:
            result = solver.step(model, state, inputs, 10.0)
        assert not result.report.accepted
        assert result.report.reason == "non-finite prediction"
        np.testing.assert_array_equal(result.state.position, state.position)
