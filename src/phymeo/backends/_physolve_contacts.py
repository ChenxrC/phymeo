"""Prepared solve-time measurements; never rerun collision or constraint solve."""

import tyx as tx


@tx.kernel(gradient="unsupported")
def _ids(count: tx.Array[tx.i32], world: tx.Array[tx.i32], geom: tx.Array[tx.i32],
         dim: tx.Array[tx.i32], address: tx.Array[tx.i32], nefc: tx.Array[tx.i32], flags: tx.Array[tx.i32],
         ids: tx.Array[tx.i32], nworld: tx.i32, ngeom: tx.i32, njmax: tx.i32,
         stride: tx.i32, cone: tx.i32):
    for i in tx.range(ids.shape[0]):
        # The frontend skips contact IDs >= nacon before dereferencing anything.
        ids[i] = count[0]
        if flags[0] == 0 and i < count[0] and count[0] <= ids.shape[0]:
            valid = world[i] >= 0 and world[i] < nworld
            valid = valid and geom[2*i] >= 0 and geom[2*i] < ngeom
            valid = valid and geom[2*i+1] >= 0 and geom[2*i+1] < ngeom
            d = dim[i]
            valid = valid and (d == 1 or d == 3 or d == 4 or d == 6)
            base = address[i*stride]
            valid = valid and base >= -1
            active_rows = 0
            if world[i] >= 0 and world[i] < nworld:
                active_rows = nefc[world[i]]
            if base >= 0:
                if cone == 0:
                    width = 1
                    if d > 1:
                        width = 2*(d-1)
                    valid = valid and base + width <= njmax and base + width <= active_rows
                else:
                    valid = valid and d <= stride
                    for c in range(6):
                        if c < d and c < stride:
                            a = address[i*stride+c]
                            valid = valid and a >= 0 and a < njmax and a < active_rows
            if valid:
                ids[i] = i
            else:
                tx.atomic_or(flags[0], 65536)


@tx.kernel(gradient="unsupported")
def _measure(count: tx.Array[tx.i32], ids: tx.Array[tx.i32], world: tx.Array[tx.i32],
             geom: tx.Array[tx.i32], dim: tx.Array[tx.i32], pos: tx.Array[tx.f32],
             frame: tx.Array[tx.f32], distance: tx.Array[tx.f32], force: tx.Array[tx.f32],
             values: tx.Array[tx.f32], identity: tx.Array[tx.i32],
             output_count: tx.Array[tx.i32], by_world: tx.Array[tx.i32],
             flags: tx.Array[tx.i32]):
    for i in tx.range(ids.shape[0]):
        if i == 0:
            output_count[0] = count[0]
        if i < count[0] and count[0] <= ids.shape[0] and ids[i] == i:
            identity[4*i] = world[i]
            identity[4*i+1] = geom[2*i]
            identity[4*i+2] = geom[2*i+1]
            identity[4*i+3] = dim[i]
            tx.atomic_add(by_world[world[i]], 1)
            for c in range(3):
                values[25*i+c] = pos[3*i+c]
            for c in range(9):
                values[25*i+3+c] = frame[9*i+c]
            values[25*i+12] = distance[i]
            for c in range(6):
                values[25*i+13+c] = force[6*i+c]
                part = c // 3
                axis = c % 3
                values[25*i+19+c] = (force[6*i+part*3] * frame[9*i+axis]
                    + force[6*i+part*3+1] * frame[9*i+3+axis]
                    + force[6*i+part*3+2] * frame[9*i+6+axis])
            for c in range(25):
                if not tx.isfinite(values[25*i+c]):
                    tx.atomic_or(flags[0], 131072)


@tx.kernel(gradient="unsupported")
def _merge(old_values: tx.Array[tx.f32], old_ids: tx.Array[tx.i32], old_count: tx.Array[tx.i32],
           new_values: tx.Array[tx.f32], new_ids: tx.Array[tx.i32], new_count: tx.Array[tx.i32],
           mask: tx.Array[tx.bool1], values: tx.Array[tx.f32], identity: tx.Array[tx.i32],
           count: tx.Array[tx.i32], by_world: tx.Array[tx.i32], flags: tx.Array[tx.i32]):
    for row in tx.range(identity.shape[0] // 2):
        capacity = identity.shape[0] // 4
        i = row % capacity
        old = row < capacity
        n = new_count[0]
        if old:
            n = old_count[0]
        if i < n and flags[0] == 0:
            world = new_ids[4*i]
            if old:
                world = old_ids[4*i]
            keep = mask[world]
            if old:
                keep = not keep
            if keep:
                slot = tx.atomic_add(count[0], 1)
                if slot >= capacity:
                    tx.atomic_or(flags[0], 262144)
                else:
                    tx.atomic_add(by_world[world], 1)
                    for c in range(4):
                        value = new_ids[4*i+c]
                        if old:
                            value = old_ids[4*i+c]
                        identity[4*slot+c] = value
                    for c in range(25):
                        value = new_values[25*i+c]
                        if old:
                            value = old_values[25*i+c]
                        values[25*slot+c] = value


class PreparedContacts:
    """Two published banks plus reset scratch; O(shared contact capacity + N*B).

    Contact rows have no persistent identity or ordering contract. Atomic merge
    retains the exact unselected records, not their old global slot numbers.
    """

    float_fields = {
        "contact_position": (0, 3, "m", "world midpoint"),
        "contact_frame": (3, 12, "unit basis", "row-major normal,tangent1,tangent2 in world"),
        "contact_distance": (12, 13, "m", "signed separation"),
        "contact_wrench_local": (13, 19, "N,N*m", "contact frame; force then torque at midpoint, on second geom"),
        "contact_wrench": (19, 25, "N,N*m", "world; force then torque at midpoint, on second geom"),
    }
    int_fields = {"contact_world": (0, 1), "contact_geom": (1, 3), "contact_dim": (3, 4)}

    def __init__(self, model, banks, device, flags):
        import physolve_tyx as frontend
        self.model, self.native, self.flags = model, banks, flags
        self.force_operation = frontend.contact_force
        self.capacity, self.nworld = banks[0].naconmax, banks[0].nworld
        self.sample_times = tuple(tx.empty(self.nworld, tx.f32, device=device) for _ in banks)
        self.body_com = tuple(tx.empty((self.nworld, model.nbody, 3), tx.f32, device=device)
                              for _ in banks)
        self.native_com = tuple(bank.xipos.view(tx.f32).flatten() for bank in banks)
        self.flat_com = tuple(array.flatten() for array in self.body_com)
        self.ids = tx.empty(self.capacity, tx.i32, device=device)
        self.force = tx.empty(self.capacity, tx.spatial_vector, device=device)
        self.force_flat = self.force.view(tx.f32).flatten()
        self.raw = tuple((tx.empty((self.capacity, 25), tx.f32, device=device),
                          tx.empty((self.capacity, 4), tx.i32, device=device),
                          tx.empty(1, tx.i32, device=device),
                          tx.empty(self.nworld, tx.i32, device=device)) for _ in range(3))
        self.flat = tuple(tuple(a.flatten() for a in bank) for bank in self.raw)
        self.banks = []
        for bank_index, (values, identity, count, by_world) in enumerate(self.raw[:2]):
            fields = {name: values[:, start:end] for name, (start, end, _, _) in self.float_fields.items()}
            fields.update({name: identity[:, start:end] for name, (start, end) in self.int_fields.items()})
            fields.update(contact_count=count, contact_count_by_world=by_world)
            fields['contact_sample_time'] = self.sample_times[bank_index]
            fields['contact_body_com_position'] = self.body_com[bank_index]
            self.banks.append(fields)
        self.inputs = tuple(tuple(getattr(bank.contact, name).view(dtype).flatten()
                                  for name, dtype in (("worldid", tx.i32), ("geom", tx.i32),
                                                      ("dim", tx.i32), ("pos", tx.f32),
                                                      ("frame", tx.f32), ("dist", tx.f32))) for bank in banks)
        self.addresses = tuple(bank.contact.efc_address.view(tx.i32).flatten() for bank in banks)
        self.extractions = self.merges = self.copies = 0

    def prepare(self, stream, all_mask):
        self.all_mask = all_mask
        for index in range(2):
            self.update(index, stream)
        # Warm merge without changing either published bank's records.
        self.update(0, stream, mask=all_mask)
        self.extractions = self.merges = self.copies = 0

    def update(self, index, stream, *, mask=None, integrated=False, record=True):
        native = self.native[index]
        target = index if mask is None else 2
        inputs = self.inputs[index]
        _ids(native.nacon, *inputs[:3], self.addresses[index], native.nefc, self.flags, self.ids,
             self.nworld, self.model.ngeom, native.njmax,
             native.contact.efc_address.shape[1], int(self.model.opt.cone), mode="cuda", stream=stream)
        self.force_operation(self.model, native, self.ids, False, self.force)
        self.raw[target][3].zero_(stream=stream)
        _measure(native.nacon, self.ids, *inputs, self.force_flat, *self.flat[target],
                 self.flags, mode="cuda", stream=stream)
        if record:
            self.extractions += 1
        _sample_reference(native.time, self.sample_times[1-index], self.sample_times[index],
                          self.native_com[index], self.flat_com[1-index], self.flat_com[index],
                          self.all_mask if mask is None else mask, self.flags, self.model.nbody * 3,
                          not integrated, mode="cuda", stream=stream)
        if mask is not None:
            self.raw[index][2].zero_(stream=stream)
            self.raw[index][3].zero_(stream=stream)
            _merge(*self.flat[1-index][:3], *self.flat[2][:3], mask, *self.flat[index],
                   self.flags, mode="cuda", stream=stream)
            if record:
                self.merges += 1

    def capture_time(self, index, stream):
        # Snapshot the actual last substep start; subtraction after integration
        # would not recover its exact float32 value.
        self.sample_times[index].assign(self.native[index].time, stream=stream)

    def copy(self, source, target, stream):
        for out, old in zip(self.raw[target], self.raw[source]):
            out.assign(old, stream=stream)
        self.sample_times[target].assign(self.sample_times[source], stream=stream)
        self.body_com[target].assign(self.body_com[source], stream=stream)
        self.copies += 1

    @property
    def allocated_bytes(self):
        return (sum(a.size*4 for bank in self.raw for a in bank) + self.capacity*28
                + self.nworld*8 + sum(a.size*4 for a in self.body_com))


@tx.kernel(gradient="unsupported")
def _sample_reference(time: tx.Array[tx.f32], previous: tx.Array[tx.f32], output: tx.Array[tx.f32],
                      com: tx.Array[tx.f32], previous_com: tx.Array[tx.f32],
                      output_com: tx.Array[tx.f32], mask: tx.Array[tx.bool1],
                      flags: tx.Array[tx.i32], width: tx.i32, refresh_time: tx.bool1):
    for i in tx.range(output_com.shape[0]):
        world = i // width
        if mask[world]:
            output_com[i] = com[i]
            if not tx.isfinite(com[i]):
                tx.atomic_or(flags[0], 131072)
        else:
            output_com[i] = previous_com[i]
        if refresh_time and i % width == 0:
            if mask[world]:
                output[world] = time[world]
            else:
                output[world] = previous[world]
