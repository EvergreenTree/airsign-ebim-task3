import importlib.util
import socket
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "udp_over_tcp.py"
SPEC = importlib.util.spec_from_file_location("udp_over_tcp", SCRIPT)
assert SPEC and SPEC.loader
udp_over_tcp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(udp_over_tcp)


def test_endpoint_parser() -> None:
    assert udp_over_tcp.parse_endpoint("127.0.0.1:47998") == ("127.0.0.1", 47998)
    with pytest.raises(Exception):
        udp_over_tcp.parse_endpoint("127.0.0.1")
    with pytest.raises(Exception):
        udp_over_tcp.parse_endpoint("127.0.0.1:70000")


def test_tcp_framing_preserves_datagram_boundaries() -> None:
    left, right = socket.socketpair()
    try:
        udp_over_tcp.send_frame(left, b"first")
        udp_over_tcp.send_frame(left, b"second-packet")
        assert udp_over_tcp.receive_frame(right) == b"first"
        assert udp_over_tcp.receive_frame(right) == b"second-packet"
    finally:
        left.close()
        right.close()
