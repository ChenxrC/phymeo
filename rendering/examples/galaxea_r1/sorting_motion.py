"""Indexed conveyor feed, class-directed bin placement and FK contact evidence."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .sorting_layout import ITEM_KINDS, ITEM_NODES, LAYOUT
from .workshop_motion import ARM, FINGERS, OPEN, STOCK_BEVEL_M, _Arm, _calibrate, _ease


_STAGES = ("feed", "approach", "close", "lift", "transfer", "lower", "release", "withdraw")
_ENDS = (.20, .32, .40, .51, .67, .77, .86, 1.)


@dataclass(frozen=True)
class SortSample:
    base: object
    joints: dict
    payloads: tuple
    belt_travel_m: float
    phase: str
    active_index: int
    attached_index: int | None
    placed_count: int


def _allocate(index, kind, occupied, layout):
    bins = dict(zip(("block", "hex"), layout.bin_centers, strict=True))
    if kind not in bins:
        raise ValueError(f"station returned unknown sort class {kind!r}")
    slot = occupied[kind]
    if slot >= 2:
        raise ValueError(f"both slots in the {kind} bin are occupied")
    x,y = bins[kind]
    x += (-1. if slot == 0 else 1.) * layout.slot_offset_x
    occupied[kind] += 1
    return dict(index=index,node=ITEM_NODES[index],kind=kind,bin=kind,slot_index=slot,
                slot_xy=(x,y),center_m=(x,y,layout.support_z+layout.payload_size[2]/2))


def scripted_classifier(kinds=ITEM_KINDS):
    """Explicit CPU test double. The film driver supplies optical station reads."""
    labels = tuple(kinds)
    def read(sample, frame_index):
        return labels[sample.active_index]
    read.scripted_dry_run = True
    return read


def _schedule(frames, cycles):
    # Put each physical event on an exposure, even for a short CPU probe.
    # Exactly one frame is emitted at a shared cycle endpoint.
    if type(frames) is not int or frames-1 < cycles*len(_STAGES):
        raise ValueError(f"need at least {cycles*len(_STAGES)+1} exposures for all sorting events")
    cuts = []
    for i in range(cycles):
        start,end = round(i*(frames-1)/cycles),round((i+1)*(frames-1)/cycles)
        edges = [start]
        for j, fraction in enumerate(_ENDS):
            desired = start + round((end-start)*fraction)
            edges.append(min(end-(len(_ENDS)-j-1), max(edges[-1]+1, desired)))
        cuts.append(edges)
    return cuts


def _command(frame, edges, start_xy, place_xy, last_cycle, layout):
    stage = min(len(_STAGES)-1, int(np.searchsorted(edges[1:],frame,side="left")))
    start,end = edges[stage:stage+2]
    t = (frame-start)/(end-start)
    p = _ease(t,0.,1.)
    pick, place = np.asarray(layout.pick_xy), np.asarray(place_xy)
    inspection = pick + np.array([-.10,-.055])
    phase = _STAGES[stage]
    z = layout.support_z + layout.payload_size[2]/2
    # A full-width stop precedes descent; the next feed starts only after
    # release and withdrawal. The return flight occurs above the moving belt.
    if stage == 0:
        p = _ease(t,0.,.78)
        xy = (1.-p)*np.asarray(start_xy) + p*inspection
        raise_z, grip, feed = 1., 0., p
        if p == 1.:
            phase = "inspect"
    elif stage == 1:
        align = _ease(t,0.,.43)
        xy = (1.-align)*inspection + align*pick
        raise_z, grip, feed = 1.-_ease(t,.43,1.), 0., 1.
    elif stage == 2:
        xy, raise_z, grip, feed = pick, 0., p, 1.
    elif stage == 3:
        xy, raise_z, grip, feed = pick, p, 1., 1.
    elif stage == 4:
        xy, raise_z, grip, feed = (1.-p)*pick+p*place, 1., 1., 1.
    elif stage == 5:
        xy, raise_z, grip, feed = place, 1.-p, 1., 1.
    elif stage == 6:
        xy, raise_z, grip, feed = place, 0., 1.-p, 1.
    else:
        xy, raise_z, grip, feed = place, p, 0., 1.
        if last_cycle:
            # Leave the final bin visible; move sideways after first clearing it.
            xy = place + _ease(t,.55,1.)*np.array([-.055,.065])
    return np.r_[xy, z+raise_z*layout.lift_m], grip, feed, phase


def _objects(index, frame, edges, travel, measured, bins, layout):
    from phymeo_rendering.scene import Pose
    # A stopped item attaches at the fully closed sample and rests in its bin
    # at touchdown, while the jaw still surrounds it during the opening phase.
    attached = index if edges[3] <= frame < edges[6] else None
    placed_count = index + int(frame >= edges[7])
    height = layout.support_z + layout.payload_size[2]/2
    poses = []
    for i, slot in enumerate(bins):
        if i == attached:
            pose = measured
        elif i < index or (i == index and frame >= edges[6]):
            pose = slot
        else:
            y = layout.pick_xy[1] + (i+1)*layout.pitch - travel
            pose = Pose((layout.pick_xy[0],y,height))
        poses.append(pose)
    return tuple(poses), attached, placed_count


def build_sort_motion(model, frames, fps=30, kinds=ITEM_KINDS, *, read_station):
    """Return four persistent part poses per exposure, plus the sorting audit."""
    from _panda_scene import pose_matrix
    from phymeo_rendering.scene import Pose, compose_pose
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("sorting FPS must be finite and positive")
    if not callable(read_station):
        raise ValueError("an explicit optical station read or scripted test classifier is required")
    kinds = tuple(kinds)
    if len(kinds) != len(ITEM_NODES) or any(k not in ("block","hex") for k in kinds):
        raise ValueError("provide one registered geometry class per workpiece")
    layout = LAYOUT
    routes, occupied = [], Counter()
    cuts = _schedule(frames,len(kinds))
    bins = [None] * len(kinds)
    center_x, contacts = _calibrate(model,layout)
    local_contacts = []
    for row in contacts:
        joint = model.joints[row["joint"]]
        zero = pose_matrix(joint.origin)
        local = zero[:3,:3].T @ (np.asarray(row["contact_at_zero_tool_m"])-zero[:3,3])
        local_contacts.append((joint.child,local))
    base = Pose((0.,0.,model.floor_offset_m))
    # A stationary torso faces slightly toward the second bin to leave the
    # elbow clearance on its inward reach. The idle arm remains folded.
    posture = dict(torso_joint1=-.24,torso_joint2=.47,torso_joint3=.23,torso_joint4=-.25)
    for j,value in enumerate((-.98,1.82,-1.45,-.08,.46,-.10),start=1):
        posture[f"right_arm_joint{j}"] = value
    posture.update(right_gripper_finger_joint1=.012,right_gripper_finger_joint2=.012)
    arm = _Arm(model,posture,base,center_x)
    relative = Pose((center_x,0.,0.),(0.,-math.sqrt(.5),0.,math.sqrt(.5)))
    home_xy = (.39,.39)
    initial = _command(0,cuts[0],home_xy,layout.pick_xy,False,layout)[0]
    seeds = ((.24,2.3,-2.,.2,1.3,-.7), (1.,2.2,-1.6,0.,.75,0.),
             (.8,2.6,-1.85,1.,1.05,-.65), (1.45,2.4,-1.45,-1.,.9,1.))
    for seed in seeds:
        q, score = arm.solve(initial,np.asarray(seed))
        if score <= 1e-6:
            break
    else:
        raise ValueError(f"cannot reach the initial sorting hover, residual {score}")

    samples, commands, cycles = [], [], []
    worst_position = worst_angle = worst_contact = worst_handoff = worst_slip = 0.
    worst_support = 0.
    max_hex_x = 0.
    for i,edges in enumerate(cuts):
        route = None
        previous_xy = routes[i-1]["slot_xy"] if i else home_xy
        peak = 0.
        for index in range(edges[0]+int(i>0),edges[-1]+1):
            # The feed and inspection use no destination. Allocate only after
            # the reader sees this item's actual stopped, open-jaw exposure.
            destination = route["slot_xy"] if route else layout.pick_xy
            target,closure,feed,phase = _command(index,edges,previous_xy,destination,i==len(kinds)-1,layout)
            travel = float((i+feed)*layout.pitch)
            q,score = arm.solve(target,q)
            if score > 5e-6:
                raise ValueError(f"sorting IK failed at frame {index}, {phase}, item {i}, "
                                 f"target {target.tolist()}, residual {score:.6g}")
            joints = dict(posture)
            joints.update(zip(ARM,q.tolist()))
            for row in contacts:
                joints[row["joint"]] = OPEN+closure*(row["closed_command_m"]-OPEN)
            nodes = {n.node_id:n.world_pose for n in model.pose_nodes(joints,base)}
            grasp = compose_pose(nodes["left_gripper_link"],relative)
            grasp_matrix = pose_matrix(grasp)
            error = float(np.linalg.norm(grasp_matrix[:3,3]-target))
            angle = math.acos(np.clip((np.trace(grasp_matrix[:3,:3])-1.)/2.,-1.,1.))
            worst_position, worst_angle = max(worst_position,error), max(worst_angle,angle)
            payloads, attached, placed = _objects(i,index,edges,travel,grasp,bins,layout)
            origin = np.asarray(payloads[i].position_m)
            above = float(origin[2] - layout.payload_size[2]/2 - layout.support_z)
            peak, worst_support = max(peak,above), max(worst_support,-above)
            if index in (edges[3],edges[6]):
                worst_handoff = max(worst_handoff,error)
            if attached is not None or index == edges[6]:
                obj = pose_matrix(payloads[i])
                for row,(child,local) in zip(contacts, local_contacts):
                    finger = pose_matrix(nodes[child])
                    point = obj[:3,:3].T @ (finger[:3,:3]@local+finger[:3,3]-obj[:3,3])
                    bevel = max(0.,point[2]-(layout.payload_size[2]/2-STOCK_BEVEL_M))
                    gap = abs(row["sign"]*point[1]-(layout.payload_size[1]/2-bevel))
                    worst_contact = max(worst_contact,float(gap))
                    if kinds[i] == "hex":
                        max_hex_x = max(max_hex_x,abs(float(point[0])))
            if samples:
                # Audit unpicked parts (including the next active one) against
                # the same belt displacement, independently of the IK targets.
                delta = travel-samples[-1].belt_travel_m
                for j in range(i+1 if index >= edges[3] else i,len(kinds)):
                    shift = np.asarray(payloads[j].position_m)-samples[-1].payloads[j].position_m
                    worst_slip = max(worst_slip,float(np.linalg.norm(shift+[0.,delta,0.])))
            samples.append(SortSample(base,joints,payloads,travel,phase,i,attached,placed))
            commands.append(q.copy())
            if index == edges[1]:
                observed = read_station(samples[-1],index)
                route = _allocate(i,observed,occupied,layout)
                route.update(read_frame=index,geometry_kind=kinds[i],classification_matches=(observed == kinds[i]))
                routes.append(route)
                bins[i] = Pose(route["center_m"])
        proof = dict(feed=(edges[0]+edges[1]+1)//2, stopped=edges[1], pick=edges[3], lift=edges[4],
                     transfer=(edges[4]+edges[5]+1)//2, place=edges[6], release=edges[7], withdraw=edges[8])
        cycles.append(dict(**route,proof_frames=proof, peak_lift_m=peak,
                           stopped_belt_m=samples[edges[1]].belt_travel_m))
    if (worst_position > .0001 or worst_angle > .001 or worst_contact > .0001 or
            worst_handoff > .0001 or worst_slip > 1e-8 or max_hex_x > .006201 or worst_support > .0001):
        raise ValueError("registered sorting FK, handoff, support, belt, or jaw contacts exceed tolerance")
    commands = np.asarray(commands)
    evidence = dict(method="kinematic pose from registered URDF FK and CAD jaw support; class-directed free-slot assignment",
        frame_count=frames, fps=fps, routes=routes, cycles=cycles, finger_contacts=contacts,
        classifier="scripted dry run" if getattr(read_station,"scripted_dry_run",False) else "station callback",
        classification_matches=all(r["classification_matches"] for r in routes),
        final_slots={kind:[r["node"] for r in routes if r["bin"] == kind] for kind in ("block","hex")},
        final_item_poses=[dict(node=node,center_m=pose.position_m,orientation_xyzw=pose.orientation_xyzw)
                          for node,pose in zip(ITEM_NODES,samples[-1].payloads)],
        total_belt_travel_m=samples[-1].belt_travel_m, placed_count=samples[-1].placed_count,
        max_conveyor_slip_m=worst_slip, max_handoff_position_error_m=worst_handoff,
        max_contact_gap_m=worst_contact, max_hex_contact_abs_x_m=max_hex_x,
        max_support_undershoot_m=worst_support, max_fk_position_error_m=worst_position,
        max_fk_orientation_error_rad=worst_angle, max_arm_step_rad=float(np.abs(np.diff(commands,axis=0)).max()),
        joint_extrema_rad={key:[float(commands[:,j].min()),float(commands[:,j].max())] for j,key in enumerate(ARM)})
    return samples,evidence
