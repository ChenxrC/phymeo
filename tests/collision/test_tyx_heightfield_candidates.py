"""Inspect the actual TYX terrain face selector and the exact roots consuming its output."""

import json
from fractions import Fraction

import numpy as np
import pytest
from test_heightfield_sphere import scene

import phymeo as ph


pytestmark = pytest.mark.tyx


def test_device_selects_terrain_faces_before_exact_ridge_roots(tyx_mode, tmp_path):
    grid = ph.Heightfield([[0,1,0]]*3, (1,1), origin=(-1,-1,0))
    start, end = [-0.75,0.125,1.25], [0.75,0.125,1.25]
    model, paths, _ = scene(grid,start,5/16,end=end)
    with ph.CollisionWorkspace(model,tyx_mode,cache_root=tmp_path/"cache") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        assert row.status == "hit" and row.sweep.first_t == (3/8, 3/8)
        assert row.heightfield.last_contact.interval == (5/8, 5/8)
        assert row.sweep.witness.source == "host-heightfield"
        stats = work.stats()
        assert stats["heightfield_completed_launches"] == 1
        assert stats["heightfield_cells_tested"] == 4 and stats["heightfield_faces_kept"] > 0
        if tyx_mode == "cuda":
            assert stats["heightfield_downloads"] == stats["heightfield_synchronizations"] == 1
        (tmp_path/"terrain-ridge.json").write_text(json.dumps({
            "backend": row.sweep.witness.source, "first": row.sweep.first_t,
            "last": row.heightfield.last_contact.interval, "stats": stats
        },allow_nan=False),encoding="utf-8")


@pytest.fixture(scope="module")
def terrain_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("terrain-device-kernels")


def exact(values):
    return np.array([Fraction(float(v)) for v in values], dtype=object)


def cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]], dtype=object)


def closed_segment_d2(p, a, b):
    direction = b-a
    t = min(1, max(0, ((p-a) @ direction)/(direction @ direction)))
    d = p-(a+t*direction)
    return d @ d


def segment_triangle_d2(start, end, tri):
    """Independent exact segment/triangle distance through convex feature minima."""
    direction = end-start
    a, b, c = tri
    n = cross(b-a, c-a)
    nn = n @ n
    edges = ((a,b), (b,c), (c,a))

    def on_face(p):
        return all(cross(y-x, p-x) @ n >= 0 for x,y in edges)

    h0, speed = (start-a) @ n, direction @ n
    if speed:
        t = -h0/speed
        if 0 <= t <= 1 and on_face(start+t*direction):
            return 0
    candidates = []
    for p in (start,end):
        h = (p-a) @ n
        projection = p-n*(h/nn)
        if on_face(projection):
            candidates.append(h*h/nn)
        candidates.extend(closed_segment_d2(p,x,y) for x,y in edges)
    dd = direction @ direction
    for x,y in edges:
        candidates.append(closed_segment_d2(x,start,end))
        edge = y-x
        ee, de = edge @ edge, direction @ edge
        det = dd*ee-de*de
        if det:
            diff = start-x
            du, eu = direction @ diff, edge @ diff
            t, u = (de*eu-ee*du)/det, (dd*eu-de*du)/det
            if 0 <= t <= 1 and 0 <= u <= 1:
                offset = diff+t*direction-u*edge
                candidates.append(offset @ offset)
    return min(candidates)


def actual_swept_faces(grid, start, end, radius):
    points = [exact(p) for p in grid.world_vertices()]
    first, last = exact(start), exact(end)
    rows, cols = len(grid.heights_yx), len(grid.heights_yx[0])
    actual = []
    for row in range(rows-1):
        for col in range(cols-1):
            sw = row*cols+col
            triangles = ((sw,sw+1,sw+cols+1), (sw,sw+cols+1,sw+cols))
            for index, ids in enumerate(triangles):
                if segment_triangle_d2(first,last,[points[i] for i in ids]) <= Fraction(radius)**2:
                    actual.append(2*(row*(cols-1)+col)+index)
    return actual


def paths_for(name):
    if name == "rectangular-rotated":
        rot = np.array([[0,0,1],[1,0,0],[0,1,0]])
        origin = np.array([2**24 + 3/8,-3,2])
        grid = ph.Heightfield([[0,0,1,0,0,0,0]]*4,(1,0.5),origin=origin,rotation=rot)
        return grid,origin+rot@[1.25,0.625,1.25],origin+rot@[2.75,0.625,1.25],5/16,0
    if name == "raised-miss":
        grid = ph.Heightfield([[0,1,0]]*3,(1,1),origin=(-1,-1,0))
        return grid,[-0.75,0.125,1.375],[0.75,0.125,1.375],5/16,0.25
    if name == "one-triangle":
        return ph.Heightfield([[0,0],[1,0]],(1,1)),[0.125,0.875,1.125],[0.125,0.875,0.8125],0.125,0
    if name == "empty-mask":
        return ph.Heightfield([[0,0,0,0,1]]*3,(1,1)),[0.125,0.5,0.75],[0.5,0.5,0.75],0.125,0
    x = 1.1875 if name == "edge-graze" else np.nextafter(1.1875,np.inf)
    return ph.Heightfield([[0,0],[0,0]],(1,1)),[x,1.25,-0.25],[x,1.25,0.5],5/16,0


@pytest.mark.parametrize("name",[
    "rectangular-rotated","raised-miss","one-triangle","empty-mask","edge-graze","edge-ulp-miss",
])
def test_masks_cover_every_true_face_and_drive_the_root_search(
    name,tyx_mode,tmp_path,terrain_cache,monkeypatch
):
    from phymeo.geometry._heightfield import _TerrainSphere

    grid, start, end, radius, margin = paths_for(name)
    model, motion, _ = scene(grid,start,radius,reverse=name=="rectangular-rotated",end=end)
    policy = ph.CollisionPolicy(margin=margin)
    with ph.CollisionWorkspace(model) as host:
        (host_row,) = ph.detect(model,motion,policy,host).pairs
    actual = actual_swept_faces(grid,start,end,radius)
    assert bool(actual) == (host_row.status == "hit")
    if name == "rectangular-rotated":
        assert host_row.sweep.first_t == (3/8,)*2 and host_row.heightfield.last_contact.interval == (5/8,)*2
    if name == "edge-graze":
        lo, hi = host_row.sweep.first_t
        assert Fraction(lo) <= Fraction(1,3) <= Fraction(hi)
    with ph.CollisionWorkspace(model,tyx_mode,cache_root=terrain_cache) as work:
        captured, consumed = [], []
        selector = work._engine.heightfield_face_candidates
        original_bounds = _TerrainSphere.event_bounds

        def select(*args):
            result = selector(*args)
            captured.append(result)
            return result

        def bounds(query, selected):
            consumed.append(selected)
            return original_bounds(query,selected)

        def reject_host_screen(*args):
            raise AssertionError("The device path invoked the HOST swept face screening")

        monkeypatch.setattr(work._engine,"heightfield_face_candidates",select)
        monkeypatch.setattr(_TerrainSphere,"event_bounds",bounds)
        monkeypatch.setattr(_TerrainSphere,"host_swept_faces",reject_host_screen)
        result = ph.detect(model,motion,policy,work)
        (row,) = result.pairs
        assert row == host_row and result.required_candidates == 1
        assert len(captured) == 1 and captured == consumed
        (ids,) = captured
        assert len(set(ids)) == len(ids) and set(actual) <= set(ids)
        if name == "one-triangle":
            assert actual == [1] and ids == (1,)
        if name == "empty-mask":
            assert ids == () and row.witness.gap_m > 0
        if name == "rectangular-rotated":
            assert len(ids) < 2*(len(grid.heights_yx)-1)*(len(grid.heights_yx[0])-1)
        stats = work.stats()
        assert stats["heightfield_cells_tested"] == (len(grid.heights_yx)-1)*(len(grid.heights_yx[0])-1)
        assert stats["heightfield_faces_kept"] == len(ids)
        assert stats["heightfield_completed_launches"] == stats["heightfield_launches"] == 1
        if tyx_mode == "cuda":
            assert stats["heightfield_downloads"] == stats["heightfield_synchronizations"] == 1
            assert stats["heightfield_downloaded_bytes"] > 0
        record = {"case":name,"mode":tyx_mode,"face_ids":ids,"true_face_ids":actual,
                  "entry":row.sweep.first_t,"last":row.heightfield.last_contact.interval if actual else None,
                  "endpoint_gap_m":row.witness.gap_m,"stats":stats}
        (tmp_path/"terrain-candidates.json").write_text(json.dumps(record,allow_nan=False),encoding="utf-8")


@pytest.mark.parametrize("fault,match", [("capacity","capacity exceeded"),("stale","incomplete"),
                                         ("mask","invalid heightfield candidate mask"),("launch","injected"),
                                         ("missing","invalid heightfield candidate selection")])
def test_device_fault_preserves_publication_and_never_falls_back(fault,match,tyx_mode,terrain_cache,monkeypatch):
    from phymeo.backends import _tyx_heightfield_kernels as kernels

    grid,start,end,radius,margin = paths_for("one-triangle")
    model,motion,_ = scene(grid,start,radius,end=end)
    policy = ph.CollisionPolicy(margin=margin)
    with ph.CollisionWorkspace(model,tyx_mode,cache_root=terrain_cache) as work:
        good = ph.detect(model,motion,policy,work)
        assert good.pairs[0].status == "hit" and work.stats()["heightfield_completed_launches"] == 1
        driver = work._engine._heightfield_candidates
        if fault == "capacity":
            monkeypatch.setattr(driver,"capacity",0)
        elif fault == "missing":
            monkeypatch.setattr(work._engine,"heightfield_face_candidates",lambda *args: None)
        elif fault == "launch":
            def bad_launch(*args,**kwargs):
                raise RuntimeError("injected terrain kernel failure")
            monkeypatch.setattr(kernels,"heightfield_masks",bad_launch)
        else:
            read = driver._read
            def damaged():
                values = read().copy()
                if fault == "stale":
                    values[0,0] -= 1
                else:
                    values[0,1] = 4
                return values
            monkeypatch.setattr(driver,"_read",damaged)
        with pytest.raises(RuntimeError,match=match):
            ph.detect(model,motion,policy,work)
        assert work.last_result is good and work.stats()["heightfield_completed_launches"] == 1
        with pytest.raises(RuntimeError,match="closed"):
            ph.detect(model,motion,policy,work)
