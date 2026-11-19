"""One bounded artifact for a manipulation environment and its update-boundary learner."""

import hashlib
import os
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile

from .manipulation_checkpoint import MAX_BYTES as MAX_ENV
from .manipulation_checkpoint import ManipulationEnvCheckpoint
from .mujoco_tyx_checkpoint import MAX_TRAINING, _decode, _encode, _json_bytes, _json_read, _read

_MAGIC = b"PHYMEOMANIPPPO1\n"
_HEADER = struct.Struct("<QQ32s")


def write_checkpoint(path, environment, training):
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    physical = environment.to_bytes()
    learner = _json_bytes(_encode(training))
    if len(learner) > MAX_TRAINING:
        raise ValueError("manipulation PPO checkpoint exceeds training byte limit")
    digest = hashlib.sha256(physical)
    digest.update(learner)
    stream = NamedTemporaryFile(dir=target.parent, prefix=".manip-ppo-", delete=False)
    staging = Path(stream.name)
    try:
        with stream:
            stream.write(_MAGIC)
            stream.write(_HEADER.pack(len(physical), len(learner), digest.digest()))
            stream.write(physical)
            stream.write(learner)
            stream.flush()
            os.fsync(stream.fileno())
        # A completed file becomes visible in one operation. Concurrent saves cannot overwrite it.
        os.link(staging, target)
    finally:
        staging.unlink(missing_ok=True)


def read_checkpoint(path):
    header_size = len(_MAGIC) + _HEADER.size
    raw = _read(Path(path), header_size + MAX_ENV + MAX_TRAINING)
    if not raw.startswith(_MAGIC) or len(raw) < header_size:
        raise ValueError("manipulation PPO checkpoint header")
    physical_size, training_size, digest = _HEADER.unpack_from(raw, len(_MAGIC))
    payload = memoryview(raw)[header_size:]
    if (
        not 0 < physical_size <= MAX_ENV
        or not 0 < training_size <= MAX_TRAINING
        or len(payload) != physical_size + training_size
    ):
        raise ValueError("manipulation PPO checkpoint lengths")
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("manipulation PPO checkpoint checksum")
    environment = ManipulationEnvCheckpoint.from_bytes(payload[:physical_size])
    training = _decode(_json_read(bytes(payload[physical_size:])), [0, 0])
    return environment, training
