"""Accumulate ordered candidate edits; materialize observations once per boundary."""

import tyx as tx


@tx.kernel(gradient="unsupported")
def _union(target: tx.Array[tx.bool1], source: tx.Array[tx.bool1]):
    for i in tx.range(target.shape[0]):
        target[i] = target[i] or source[i]


class PreparedUpdates:
    def __init__(self, nworld, device):
        self.worlds = tx.empty(nworld, tx.bool1, device=device)
        self.batches = self.edits = self.materializations = 0

    def prepare(self, stream):
        self.worlds.zero_(stream=stream)
        _union(self.worlds, self.worlds, mode='cuda', stream=stream)

    def add(self, runtime, *, refresh_state, compose_controls, selected_worlds,
            restore_contacts=False):
        pending = runtime._pending
        if 'updates' not in pending:
            pending['updates'] = dict(refresh=False, compose=False, restore_contacts=True)
            self.worlds.zero_(stream=runtime._stream)
            self.batches += 1
        options = pending['updates']
        if refresh_state:
            _union(self.worlds,
                   runtime._reset.all_mask if selected_worlds is None else selected_worlds,
                   mode='cuda', stream=runtime._stream)
        options['refresh'] |= bool(refresh_state)
        options['compose'] |= bool(compose_controls)
        options['restore_contacts'] &= restore_contacts
        self.edits += 1

    def flush(self, runtime, ticket):
        options = runtime._pending.pop('updates', None)
        if options is None:
            return
        runtime.enqueue_update(ticket, refresh_state=options['refresh'],
                               compose_controls=options['compose'],
                               restore_contacts=options['restore_contacts'],
                               selected_worlds=self.worlds if options['refresh'] else None)
        self.materializations += 1

    def compose_for_execution(self, runtime, index):
        options = runtime._pending.pop('updates', None)
        if options and options['compose']:
            runtime._controls_binding.compose(index, runtime._stream)
