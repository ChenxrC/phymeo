"""World-complete contact pages sharing one resident mechanics/publication step.

Pages bound broad-phase storage; they never partition a world's forces or step
physics independently. All registrations are validated before device allocation.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionModel
from ._device_convex_query import ConvexQueryPlan
from ._resident_affine_binding import ResidentAffineTreeBinding
from ._resident_collision_plan import ResidentCollisionPlan, ResidentStaticBinding, _capacity
from ._resident_contact_plan import _ResidentContactMechanics, _site_limit
from ._resident_device_patch_bridge import DevicePatchConsumption, patches_from_device
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


@dataclass(frozen=True, eq=False)
class ResidentDevicePatchBundleConsumption:
    """Owned entry geometry from every page, published with one physical step.

    Page records retain their original independent registrations and complete
    raw-to-effective conversion evidence. No combined query identity is invented.
    """

    owner: object
    registration: object
    topology_generation: int
    state_epoch: int
    active: int
    page_consumptions: tuple
    page_queries: tuple = field(init=False)
    samples: Mapping = field(init=False)

    def __post_init__(self):
        consumptions = tuple(self.page_consumptions)
        queries, samples = [], {}
        for consumption in consumptions:
            if consumption is None:
                queries.append(None)
                continue
            if not isinstance(consumption, DevicePatchConsumption):
                raise TypeError("bundle requires qualified per-page device patch consumption")
            query = consumption.query
            if (
                query.owner is not self.owner
                or query.topology_generation != self.topology_generation
                or query.state_epoch != self.state_epoch
                or query.active != self.active
            ):
                raise ValueError("device contact pages differ from the shared entry state")
            if set(samples).intersection(consumption.samples):
                raise ValueError("device contact page samples overlap")
            samples.update(consumption.samples)
            queries.append(query)
        object.__setattr__(self, "page_consumptions", consumptions)
        object.__setattr__(self, "page_queries", tuple(queries))
        object.__setattr__(self, "samples", MappingProxyType(samples))


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
        narrow_phase="host_patch",
        device_patch_site_capacity=None,
    ):
        from .tyx_world_batch import TyxWorldBatch

        if not isinstance(batch, TyxWorldBatch):
            raise TypeError("contact bundle requires a TyxWorldBatch owner")
        if narrow_phase not in ("host_patch", "device_patch"):
            raise ValueError("narrow_phase must be host_patch or device_patch")
        if narrow_phase == "host_patch" and device_patch_site_capacity is not None:
            raise ValueError("HOST contact pages do not take a device patch capacity")
        patch_capacity = (
            4096 if device_patch_site_capacity is None else _site_limit(device_patch_site_capacity)
        )
        if patch_capacity < 1:
            raise ValueError("device patch capacity must be an integer from 1 to 4096")
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
        self._narrow_phase = narrow_phase
        self._device_patch_capacity = patch_capacity
        self._device_patch_queries = (None,) * len(specs)
        self._last_device_patch = None
        self._last_device_queries = (None,) * len(specs)
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
                # All pages exist before dependent queries bind their native
                # source/layout registrations. Each page owns its own query.
                if self._narrow_phase == "device_patch":
                    self._device_patch_queries = tuple(
                        ConvexQueryPlan(batch, page, site_capacity=patch_capacity)
                        if spec.model.shapes else None
                        for page, spec in zip(self._pages, self._page_specs, strict=True)
                    )
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

    @property
    def last_device_patch(self):
        """Last committed entry aggregate; never an endpoint-current query."""
        with self._batch._lock:
            return self._last_device_patch

    @property
    def last_device_queries(self):
        """Page-aligned committed queries; empty and HOST pages have None."""
        with self._batch._lock:
            return self._last_device_queries

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

    def _stage_contact_geometry(self, attempt, entry):
        if self._narrow_phase == "host_patch":
            return None
        if len(attempt["pending"]) != len(self._pages):
            raise ValueError("device contact pages differ from the staged broad phase")
        consumptions = []
        for page, query, worlds, broad, pending in zip(
            self._pages, self._device_patch_queries, self._page_worlds,
            attempt["result"].page_results, attempt["pending"], strict=True,
        ):
            pending_page, pending_broad, layouts = pending
            if pending_page is not page or pending_broad is not broad:
                raise ValueError("staged page owner or result identity differs")
            if query is None:
                if worlds or broad.shape_keys or broad.candidates:
                    raise ValueError("nonempty device contact page has no query")
                consumptions.append(None)
                continue
            result, retained_broad, retained_layouts = query._stage_query(
                (broad, layouts, attempt["src"])
            )
            if retained_broad is not broad or retained_layouts is not layouts:
                raise RuntimeError("device page did not retain its staged broad phase")
            query.require_current(result)
            if (
                result.state_epoch != attempt["epoch"]
                or result.active != attempt["src"]
                or tuple(row.pair for row in result.rows) != tuple(sorted(broad.candidates))
            ):
                raise ValueError("device page differs from complete entry candidates")
            consumptions.append(
                patches_from_device(
                    result, {world: entry[world]["frames"] for world in worlds},
                    candidates=broad.candidates,
                )
            )
        return ResidentDevicePatchBundleConsumption(
            self._owner, self._registration, self._topology_generation,
            attempt["epoch"], attempt["src"], tuple(consumptions),
        )

    def _check_contact_geometry(self, geometry):
        if geometry is None:
            if self._narrow_phase == "device_patch" and self._contact_worlds:
                raise ValueError("nonempty device contact bundle lost its geometry")
            return
        if (
            self._narrow_phase != "device_patch"
            or not isinstance(geometry, ResidentDevicePatchBundleConsumption)
            or geometry.owner is not self._owner
            or geometry.registration is not self._registration
            or geometry.topology_generation != self._topology_generation
            or len(geometry.page_consumptions) != len(self._pages)
        ):
            raise ValueError("device patch aggregate belongs to different contact pages")
        for query, consumption in zip(
            self._device_patch_queries, geometry.page_consumptions, strict=True
        ):
            if query is None:
                if consumption is not None:
                    raise ValueError("empty page acquired device geometry")
            else:
                if consumption is None:
                    raise ValueError("nonempty page lost device geometry")
                query.require_current(consumption.query)

    def _publish_contact_geometry(self, geometry):
        # No conversion, validation, allocation or device operation after swap.
        if geometry is not None:
            for query, result in zip(
                self._device_patch_queries, geometry.page_queries, strict=True
            ):
                if query is not None:
                    query._publish_staged(result)
            self._last_device_patch = geometry
            self._last_device_queries = geometry.page_queries

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
            device_pages = tuple(
                None if query is None else query.stats for query in self._device_patch_queries
            )
            result.update(
                narrow_phase=self._narrow_phase,
                device_patch_site_capacity=(
                    self._device_patch_capacity if self._narrow_phase == "device_patch" else 0
                ),
                device_patch_pages=device_pages,
                device_patch_live_bytes=sum(p["live_bytes"] for p in device_pages if p is not None),
                device_patch_allocations=sum(p["allocations"] for p in device_pages if p is not None),
            )
            result["device_owned_bytes"] += result["device_patch_live_bytes"]
            result["device_allocations"] += result["device_patch_allocations"]
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
