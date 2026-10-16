"""Binary framing between game_camera.py (writer) and spell_engine.CameraWorker (reader).

Each frame is: magic b"GFR1", big-endian uint32 metadata length, uint32 payload
length, UTF-8 JSON metadata, then raw payload bytes (the JPEG preview). This
avoids base64 (+33% size) and parsing a large JSON string every frame.
Standard library only, so neither process pulls in the other's native libraries.
"""

from __future__ import annotations

import json
import struct

MAGIC = b"GFR1"
_HEADER = struct.Struct(">4sII")
MAX_META_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


def write_frame(stream, meta, payload=b""):
    """Write one frame to a binary stream and flush it."""
    encoded = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    stream.write(_HEADER.pack(MAGIC, len(encoded), len(payload)) + encoded + payload)
    stream.flush()


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("Camera stream ended mid-frame.")
    return data


def read_frame(stream):
    """Return (meta, payload), or None at a clean end of stream."""
    header = stream.read(_HEADER.size)
    if not header:
        return None
    if len(header) != _HEADER.size:
        raise EOFError("Camera stream ended mid-frame.")
    magic, meta_size, payload_size = _HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError("Camera stream is out of sync (unexpected output on stdout).")
    if meta_size > MAX_META_BYTES or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError("Camera frame is larger than the protocol allows.")
    meta = json.loads(_read_exact(stream, meta_size).decode("utf-8"))
    return meta, _read_exact(stream, payload_size)
