"""World-space layout of the indexed conveyor and its two receiving bins (metres)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SortingLayout:
    pick_xy: tuple = (.46, .47)
    support_z: float = .88
    payload_size: tuple = (.078, .056, .068)
    lift_m: float = .145
    pitch: float = .32
    bin_centers: tuple = ((.48, .237), (.48, .036))
    slot_offset_x: float = .045


LAYOUT = SortingLayout()
ITEM_KINDS = ("block", "hex", "hex", "block")
ITEM_NODES = tuple(f"sort-item-{i}" for i in range(1, 5))
