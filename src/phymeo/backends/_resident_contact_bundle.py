"""World-complete contact pages sharing one resident mechanics/publication step.

Pages bound broad-phase storage; they never partition a world's forces or step
physics independently. All registrations are validated before device allocation.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionModel
from ._resident_affine_binding import ResidentAffineTreeBinding
from ._resident_collision_plan import ResidentCollisionPlan, ResidentStaticBinding, _capacity
from ._resident_contact_plan import _ResidentContactMechanics, _site_limit
from ._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_tree_binding import ResidentTreeBinding

MAX_CONTACT_PAGES = 8
_CONTACT_BINDING_TYPES = (
    ResidentTreeBinding,
    ResidentAffineTreeBinding,
    ResidentFloatingTreeBinding,
    ResidentFloatingAffineTreeBinding,
    ResidentStaticBinding,
)


@dataclass(frozen=True, eq=False)
class ResidentContactPageSpec:
    """Owned immutable registration data; original per-page model/caps apply."""

    model: CollisionModel
    bindings: tuple
    materials_by_pair: Mapping
    candidate_capacity: int = 4096

    def __post_init__(self):
        if not isinstance(self.model, CollisionModel):
            raise TypeError("contact page requires a CollisionModel")
        if not isinstance(self.materials_by_pair, Mapping):
            raise TypeError("contact page materials must be a complete pair mapping")
        bindings = tuple(self.bindings)
        if not all(type(b) in _CONTACT_BINDING_TYPES for b in bindings):
            raise ValueError("contact page requires supported tree/static bindings")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self, "materials_by_pair", MappingProxyType(dict(self.materials_by_pair))
        )
        object.__setattr__(self, "candidate_capacity", _capacity(self.candidate_capacity))


@dataclass(frozen=True, eq=False)
class ResidentContactBundleResult:
    """Complete per-page current-pose candidates, not narrow contact or CCD."""

    owner: object
    registration: object
    topology_generation: int
    state_epoch: int
    world_keys: tuple
    page_results: tuple
    candidates: tuple
    candidate_capacity: int

    def __post_init__(self):
        for name in ("world_keys", "page_results", "candidates"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


class _ContactCollisionPage(ResidentCollisionPlan):
    """Validate with the original collision code, defer all device resources."""

    def __init__(self, batch, spec, caps, tree_dynamic_pairs):
        self._contact_bindings = MappingProxyType({b.key: b for b in spec.bindings})
        self._materials = spec.materials_by_pair
        self._site_config = caps
        self._deferred_device = None
        super().__init__(
            batch,
            spec.model,
            spec.bindings,
            candidate_capacity=spec.candidate_capacity,
            tree_dynamic_pairs=tree_dynamic_pairs,
        )

    def _prepare_topology(self, slots):
        super()._prepare_topology(slots)
        _ResidentContactMechanics._prepare_contact_registration(self)

    def _prepare_device(self, view, slots, paths):
        # No scoped device view is retained, and no TYX kernel factory is called.
        self._deferred_device = (slots.copy(), dict(paths))

    def _allocate_device(self, view):
        if self._deferred_device is not None:
            slots, paths = self._deferred_device
            ResidentCollisionPlan._prepare_device(self, view, slots, paths)
            self._deferred_device = None


class ResidentContactBundle(_ResidentContactMechanics):
    """At most eight complete-world pages, one solve per resident tree storage."""

    def __init__(
        self,
        batch,
        pages,
        *,
        site_capacity=128,
        sampling="discrete_entry",
        tree_dynamic_pairs=False,
    ):
        from .tyx_world_batch import TyxWorldBatch

        if not isinstance(batch, TyxWorldBatch):
            raise TypeError("contact bundle requires a TyxWorldBatch owner")
        if sampling != "discrete_entry":
            raise ValueError("resident contact supports only discrete_entry sampling, not CCD")
        if type(tree_dynamic_pairs) is not bool:
            raise TypeError("tree_dynamic_pairs must be a bool")
        # Bound the iterable before constructing/allocating any page.
        specs = []
        for spec in pages:
            if len(specs) == MAX_CONTACT_PAGES:
                raise ValueError("resident contact bundle supports at most eight pages")
            if not isinstance(spec, ResidentContactPageSpec):
                raise TypeError("expected ResidentContactPageSpec entries")
            specs.append(spec)
        self._page_specs = tuple(specs)
        worlds, keys, page_worlds = set(), set(), []
        for spec in specs:
            _capacity(spec.candidate_capacity)
            current_keys = {s.key for s in spec.model.shapes}
            current_worlds = {key.world for key in current_keys}
            if keys & current_keys or worlds & current_worlds:
                raise ValueError("a contact-bound world must belong wholly to exactly one page")
            keys.update(current_keys)
            worlds.update(current_worlds)
            page_worlds.append(tuple(sorted(current_worlds)))
        if isinstance(site_capacity, Mapping):
            if (
                any(
                    isinstance(w, (bool, np.bool_)) or not isinstance(w, Integral)
                    for w in site_capacity
                )
                or set(site_capacity) != worlds
            ):
                raise ValueError("site capacities must cover exactly the contact-bound worlds")
            caps = {w: _site_limit(site_capacity[w]) for w in worlds}
        else:
            caps = dict.fromkeys(worlds, _site_limit(site_capacity))

        self._batch = batch
        self._closed = False
        self._registration = object()
        self._tree_dynamic_pairs = tree_dynamic_pairs
        self._pages = []
        self._page_worlds = tuple(page_worlds)
        self._candidate_capacity = sum(spec.candidate_capacity for spec in specs)
        self._last = None
        self._queries = self._failed_queries = 0
        self._initialize_contact_mechanics()
        with batch._lock:
            batch._check_open()
            try:
                # All topology, owner, shape, binding and material checks precede
                # ANY device allocation, including those in later pages.
                for spec, bound_worlds in zip(specs, page_worlds, strict=True):
                    page = _ContactCollisionPage(
                        batch, spec, {w: caps[w] for w in bound_worlds}, tree_dynamic_pairs
                    )
                    self._pages.append(page)
                self._pages = tuple(self._pages)
                self._contact_worlds = tuple(sorted(worlds))
                self._site_caps = MappingProxyType(caps)
                self._contact_bindings = MappingProxyType(
                    {
                        key: binding
                        for page in self._pages
                        for key, binding in page._contact_bindings.items()
                    }
                )
                self._contact_shapes = MappingProxyType(
                    {
                        w: shapes
                        for page in self._pages
                        for w, shapes in page._contact_shapes.items()
                    }
                )
                self._world_materials = {
                    w: material
                    for page in self._pages
                    for w, material in page._world_materials.items()
                }
                with batch._borrow_collision_state() as view:
                    self._owner = view.owner
                    self._topology_generation = view.topology_generation
                    for page in self._pages:
                        page._allocate_device(view)
                    if self._contact_worlds:
                        self._prepare_contact_device()
                batch._contact_bundles.add(self)
            except BaseException:
                self.close()
                raise

    @property
    def page_specs(self):
        return self._page_specs

    @property
    def last_result(self):
        with self._batch._lock:
            return self._last

    @property
    def last_query(self):
        return self.last_result

    @property
    def last_page_results(self):
        with self._batch._lock:
            return () if self._last is None else self._last.page_results

    def _check_open(self):
        if self._closed:
            raise RuntimeError("resident contact bundle is closed")
        self._batch._check_open()
        for page in self._pages:
            page._check_open()

    def _stage_query(self):
        self._check_open()
        publications = []
        for page in self._pages:
            try:
                result, pending = page._stage_query()
            except Exception:
                page._failed_queries += 1
                self._failed_queries += 1
                raise
            publications.append((page, result, pending))
        with self._batch._borrow_collision_state(
            owner=self._owner, topology_generation=self._topology_generation
        ) as view:
            results = tuple(item[1] for item in publications)
            if any(r.state_epoch != view.state_epoch for r in results):
                raise ValueError("contact pages do not share one accepted entry epoch")
            result = ResidentContactBundleResult(
                view.owner,
                self._registration,
                view.topology_generation,
                view.state_epoch,
                self._contact_worlds,
                results,
                tuple(pair for r in results for pair in r.candidates),
                self._candidate_capacity,
            )
        return result, tuple(publications)

    def require_current(self, result):
        with self._batch._lock:
            self._check_open()
            if not isinstance(result, ResidentContactBundleResult):
                raise TypeError("expected a ResidentContactBundleResult")
            if result.registration is not self._registration or result.owner is not self._owner:
                raise ValueError("contact result belongs to a different bundle owner/registration")
            if (
                result.topology_generation != self._topology_generation
                or result.world_keys != self._contact_worlds
                or len(result.page_results) != len(self._pages)
                or result.candidate_capacity != self._candidate_capacity
                or result.candidates
                != tuple(
                    pair for page_result in result.page_results for pair in page_result.candidates
                )
            ):
                raise ValueError("contact bundle result has different topology/pages")
            for page, page_result in zip(self._pages, result.page_results, strict=True):
                page.require_current(page_result)
                if page_result.state_epoch != result.state_epoch:
                    raise ValueError("contact pages have different entry epochs")
            with self._batch._borrow_collision_state(
                owner=self._owner, topology_generation=self._topology_generation
            ) as view:
                if result.state_epoch != view.state_epoch:
                    raise ValueError("stale resident contact bundle state epoch")
            return result

    def _publish_query(self, result, pending):
        # All objects and receipts already exist; no query, device IO or validation
        # is permitted after the batch's single bank publication.
        for page, page_result, layouts in pending:
            page._trees.layouts = layouts
            page._last = page_result
            page._queries += 1
        self._last = result
        self._queries += 1

    def query(self):
        with self._batch._lock:
            result, pending = self._stage_query()
            self._publish_query(result, pending)
            return result

    def stats(self):
        with self._batch._lock:
            pages = tuple(page.stats() for page in self._pages)
            result = dict(
                mode=self._batch._mode,
                closed=self._closed,
                page_count=len(self._pages),
                page_worlds=self._page_worlds,
                world_count=len(self._contact_worlds),
                candidate_capacity=self._candidate_capacity,
                page_candidate_capacities=tuple(s.candidate_capacity for s in self._page_specs),
                queries=self._queries,
                failed_queries=self._failed_queries,
                device_owned_bytes=sum(p["device_owned_bytes"] for p in pages),
                device_allocations=sum(p["device_allocations"] for p in pages),
                pages=pages,
            )
            return self._contact_stats(result)

    def close(self):
        with self._batch._lock:
            if self._closed:
                return
            self._closed = True
            try:
                with ExitStack() as closing:
                    for page in self._pages:
                        closing.callback(page.close)
                    closing.callback(self._close_contact_mechanics)
            finally:
                self._batch._contact_bundles.discard(self)

    def __enter__(self):
        with self._batch._lock:
            self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
