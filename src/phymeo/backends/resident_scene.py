"""Public static schemas for resident multi-participant scenes.

ROB1a exposes usable registration/preflight objects only. Scene stepping and
device allocation are intentionally absent until the shared transaction exists.
Importing this module does not create a runtime or select a compute backend.
"""

from ._resident_participant_layout import (
    MAX_PARTICIPANTS_PER_WORLD,
    ResidentParticipant,
    ResidentParticipantBinding,
    ResidentParticipantGeometry,
    ResidentParticipantKey,
    ResidentParticipantLayout,
    ResidentParticipantLayoutEntry,
    ResidentParticipantWorldLayout,
)

__all__ = [
    "MAX_PARTICIPANTS_PER_WORLD",
    "ResidentParticipant",
    "ResidentParticipantBinding",
    "ResidentParticipantGeometry",
    "ResidentParticipantKey",
    "ResidentParticipantLayout",
    "ResidentParticipantLayoutEntry",
    "ResidentParticipantWorldLayout",
]
