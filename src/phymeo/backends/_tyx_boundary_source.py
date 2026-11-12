"""Device-only accepted FK gather, finite AABBs and COL2 pair-local packing.

This module adds no closest-point mathematics. The original 18-candidate
triangle distance kernel remains the only narrow-phase producer.
"""
import tyx as tx

from ._tyx_resident_revolute_bounds import _lower, _upper


@tx.kernel
def gather_boundary_points(
    count:tx.i32, selected_layout:tx.i32, body_extra:tx.i32,
    shape_meta:tx.ndarray(tx.i32,2), local:tx.ndarray(tx.f32,2),
    local_bounds:tx.ndarray(tx.f32,2), tree_meta:tx.ndarray(tx.i32,2),
    fk:tx.ndarray(tx.f32,3), ticks:tx.ndarray(tx.i32,1),
    poses:tx.ndarray(tx.f32,2), stamps:tx.ndarray(tx.i32,2),
    points:tx.ndarray(tx.f32,2), point_bounds:tx.ndarray(tx.f32,2),
):
    for shape in tx.range(count):
        if shape_meta[shape,0]==selected_layout:
            row=shape_meta[shape,1]
            group=shape_meta[shape,2]
            slot=shape_meta[shape,3]
            begin=shape_meta[shape,4]
            end=shape_meta[shape,5]
            valid=1
            stamps[shape,1]=ticks[slot]
            stamps[shape,2]=slot
            stamps[shape,3]=row
            for a in tx.static_range(12):
                pose=0.0
                if a==3 or a==7 or a==11:
                    pose=1.0
                if row>=0:
                    if group>=0 and group<tree_meta[row,1]+body_extra and slot==tree_meta[row,0]:
                        pose=fk[row,group,a]
                    else:
                        valid=0
                poses[shape,a]=pose
                if not tx.isfinite(pose):
                    valid=0
            for vertex in range(begin,end):
                for a in tx.static_range(3):
                    center=local[vertex,a]
                    low=local_bounds[vertex,a]
                    high=local_bounds[vertex,3+a]
                    if row>=0:
                        center=poses[shape,a]
                        low=poses[shape,a]
                        high=poses[shape,a]
                        for b in tx.static_range(3):
                            coefficient=poses[shape,3+3*a+b]
                            center=tx.add_rn(center,tx.mul_rn(coefficient,local[vertex,b]))
                            left=local_bounds[vertex,b]
                            right=local_bounds[vertex,3+b]
                            if coefficient<0.0:
                                left=local_bounds[vertex,3+b]
                                right=local_bounds[vertex,b]
                            low=_lower(tx.add_rn(low,_lower(tx.mul_rn(coefficient,left))))
                            high=_upper(tx.add_rn(high,_upper(tx.mul_rn(coefficient,right))))
                    points[vertex,a]=center
                    point_bounds[vertex,a]=low
                    point_bounds[vertex,3+a]=high
                    if not tx.isfinite(center) or not tx.isfinite(low) or not tx.isfinite(high):
                        valid=0
                    if low>center or high<center:
                        valid=0
            stamps[shape,0]=valid


@tx.kernel
def boundary_leaf_bounds(count:tx.i32,faces:tx.ndarray(tx.i32,2),
    bounds:tx.ndarray(tx.f32,2),leaves:tx.ndarray(tx.f32,2)):
    for face in tx.range(count):
        for a in tx.static_range(3):
            low=bounds[faces[face,0],a]
            high=bounds[faces[face,0],3+a]
            for corner in tx.static_range(1,3):
                low=tx.min(low,bounds[faces[face,corner],a])
                high=tx.max(high,bounds[faces[face,corner],3+a])
            leaves[face,a]=low
            leaves[face,3+a]=high


@tx.kernel
def boundary_round_bounds(count:tx.i32,margin_upper:tx.f32,
    rounds:tx.ndarray(tx.i32,1),shape_meta:tx.ndarray(tx.i32,2),
    radii:tx.ndarray(tx.f32,2),bounds:tx.ndarray(tx.f32,2),
    queries:tx.ndarray(tx.f32,2)):
    for index in tx.range(count):
        shape=rounds[index]
        begin=shape_meta[shape,4]
        end=shape_meta[shape,5]
        skin=_upper(tx.add_rn(radii[shape,1],margin_upper))
        for a in tx.static_range(3):
            low=bounds[begin,a]
            high=bounds[begin,3+a]
            for vertex in range(begin+1,end):
                low=tx.min(low,bounds[vertex,a])
                high=tx.max(high,bounds[vertex,3+a])
            queries[index,a]=_lower(tx.sub_rn(low,skin))
            queries[index,3+a]=_upper(tx.add_rn(high,skin))


@tx.kernel
def pack_boundary_pairs(
    count:tx.i32,generation:tx.i32,selected:tx.ndarray(tx.i32,2),
    faces:tx.ndarray(tx.i32,2),rounds:tx.ndarray(tx.i32,1),
    shape_meta:tx.ndarray(tx.i32,2),source_radii:tx.ndarray(tx.f32,2),
    points:tx.ndarray(tx.f32,2),geometry:tx.ndarray(tx.f32,3),
    radii:tx.ndarray(tx.f32,1),scales:tx.ndarray(tx.f32,1),
    owners:tx.ndarray(tx.i32,2),frames:tx.ndarray(tx.f32,2),
    status:tx.ndarray(tx.i32,2),
):
    for index in tx.range(count):
        segment=selected[index,0]
        facet=selected[index,1]
        shape=rounds[segment]
        first=shape_meta[shape,4]
        last=shape_meta[shape,5]-1
        good=1
        owners[index,0]=segment
        owners[index,1]=facet
        for a in tx.static_range(3):
            anchor=points[faces[facet,0],a]
            frames[index,a]=anchor
            for corner in tx.static_range(3):
                geometry[index,corner,a]=tx.sub_rn(points[faces[facet,corner],a],anchor)
            geometry[index,3,a]=tx.sub_rn(points[first,a],anchor)
            geometry[index,4,a]=tx.sub_rn(points[last,a],anchor)
        radius=source_radii[shape,0]
        radii[index]=radius
        length=radius
        for edge in tx.static_range(4):
            ia=edge
            ib=edge+1
            if edge==2: ib=0
            if edge==3: ib=4
            square=0.0
            for a in tx.static_range(3):
                delta=tx.sub_rn(geometry[index,ib,a],geometry[index,ia,a])
                square=tx.add_rn(square,tx.mul_rn(delta,delta))
            length=tx.max(length,tx.sqrt(square))
        if not tx.isfinite(length) or length<=0.0 or not tx.isfinite(radius) or radius<0.0:
            good=0
        for corner in tx.static_range(5):
            for a in tx.static_range(3):
                if not tx.isfinite(geometry[index,corner,a]):
                    good=0
        scales[index]=length
        status[index,0]=good
        status[index,1]=generation
