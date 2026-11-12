"""Two fixed native CUDA Graphs borrow the runtime's existing physical banks."""


class PreparedStepGraphs:
    def __init__(self, runtime, steps):
        self.runtime = runtime
        self.steps = steps
        self.graphs = []
        self.ready = False

    def prepare(self):
        runtime = self.runtime
        for index in (0, 1):
            graph = runtime._tx.Graph(stream=runtime._stream, mode='cuda_stream')
            self.graphs.append(graph)  # Retain even when capture or cleanup fails.
            with graph:
                runtime._enqueue_step_work(index, self.steps, record=False)
        self.ready = True

    def replay(self, index):
        if not self.ready:
            raise RuntimeError('fixed step Graph is not prepared')
        self.graphs[index].replay()
        self.runtime._graph_replays += 1
        bodies = self.runtime._body_observations
        if bodies is not None:
            bodies.kinematics_passes += 1
            bodies.publications += 1
        contacts = self.runtime._contact_observations
        if contacts is not None:
            contacts.extractions += 1
        if self.runtime._body_measurements is not None:
            self.runtime._body_measurements.evaluations += 1

    def close(self):
        self.ready = False
        while self.graphs:
            self.graphs[-1].destroy()
            self.graphs.pop()
        self.runtime = None
