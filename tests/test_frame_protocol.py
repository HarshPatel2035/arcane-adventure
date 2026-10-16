import io

import pytest

from frame_protocol import MAGIC, read_frame, write_frame


def test_round_trip_multiple_frames_then_clean_eof():
    stream = io.BytesIO()
    jpeg = bytes(range(256)) * 40
    write_frame(stream, {"generation": 3, "gesture_id": None}, jpeg)
    write_frame(stream, {"error": "camera unplugged"})
    stream.seek(0)
    assert read_frame(stream) == ({"generation": 3, "gesture_id": None}, jpeg)
    assert read_frame(stream) == ({"error": "camera unplugged"}, b"")
    assert read_frame(stream) is None


def test_stray_stdout_text_is_reported_not_misparsed():
    stream = io.BytesIO(b"INFO: Created delegate\n" + MAGIC)
    with pytest.raises(ValueError, match="out of sync"):
        read_frame(stream)


def test_truncated_frame_raises():
    stream = io.BytesIO()
    write_frame(stream, {"generation": 1}, b"x" * 100)
    stream = io.BytesIO(stream.getvalue()[:-10])
    with pytest.raises(EOFError):
        read_frame(stream)
