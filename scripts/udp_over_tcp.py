#!/usr/bin/env python3
"""Preserve UDP datagrams across a loopback-only SSH TCP forward.

The local side listens for the Isaac WebRTC client's UDP packets. The remote
side sends them to Isaac's loopback UDP stream port and returns replies. Both
TCP listeners stay on loopback; SSH remains the only network exposure.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time


HEADER = struct.Struct("!I")
MAX_DATAGRAM = 65535


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    try:
        parsed_port = int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("endpoint port must be an integer") from exc
    if not 1 <= parsed_port <= 65535:
        raise argparse.ArgumentTypeError("endpoint port must be in 1..65535")
    return host, parsed_port


def receive_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("TCP framing connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(sock: socket.socket) -> bytes:
    (length,) = HEADER.unpack(receive_exact(sock, HEADER.size))
    if length > MAX_DATAGRAM:
        raise ValueError(f"invalid UDP datagram length: {length}")
    return receive_exact(sock, length)


def send_frame(sock: socket.socket, payload: bytes) -> None:
    if len(payload) > MAX_DATAGRAM:
        raise ValueError(f"UDP datagram is too large: {len(payload)}")
    sock.sendall(HEADER.pack(len(payload)) + payload)


def pump_udp_to_tcp(udp_sock: socket.socket, tcp_sock: socket.socket) -> None:
    while True:
        payload = udp_sock.recv(MAX_DATAGRAM)
        send_frame(tcp_sock, payload)


def serve_remote_connection(
    tcp_sock: socket.socket,
    udp_target: tuple[str, int],
) -> None:
    with tcp_sock, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
        udp_sock.connect(udp_target)
        threading.Thread(
            target=pump_udp_to_tcp,
            args=(udp_sock, tcp_sock),
            daemon=True,
        ).start()
        while True:
            udp_sock.send(receive_frame(tcp_sock))


def run_remote(tcp_listen: tuple[str, int], udp_target: tuple[str, int]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(tcp_listen)
        listener.listen(1)
        while True:
            connection, _ = listener.accept()
            try:
                serve_remote_connection(connection, udp_target)
            except (ConnectionError, OSError, ValueError):
                connection.close()


def _connect_with_retry(target: tuple[str, int]) -> socket.socket:
    while True:
        try:
            connected = socket.create_connection(target, timeout=5.0)
            connected.settimeout(None)
            return connected
        except OSError:
            time.sleep(0.5)


def run_local(udp_listen: tuple[str, int], tcp_target: tuple[str, int]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
        udp_sock.bind(udp_listen)
        client: tuple[str, int] | None = None
        tcp_sock: socket.socket | None = None
        tcp_lock = threading.Lock()

        def return_packets(active_tcp: socket.socket) -> None:
            nonlocal tcp_sock
            try:
                while True:
                    payload = receive_frame(active_tcp)
                    if client is not None:
                        udp_sock.sendto(payload, client)
            except (ConnectionError, OSError, ValueError):
                with tcp_lock:
                    if tcp_sock is active_tcp:
                        tcp_sock = None
                active_tcp.close()

        while True:
            payload, client = udp_sock.recvfrom(MAX_DATAGRAM)
            with tcp_lock:
                if tcp_sock is None:
                    tcp_sock = _connect_with_retry(tcp_target)
                    threading.Thread(
                        target=return_packets,
                        args=(tcp_sock,),
                        daemon=True,
                    ).start()
                active_tcp = tcp_sock
            try:
                send_frame(active_tcp, payload)
            except (ConnectionError, OSError):
                with tcp_lock:
                    if tcp_sock is active_tcp:
                        tcp_sock = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    local = subparsers.add_parser("local")
    local.add_argument(
        "--udp-listen", type=parse_endpoint, default=("127.0.0.1", 47998)
    )
    local.add_argument(
        "--tcp-target", type=parse_endpoint, default=("127.0.0.1", 47999)
    )
    remote = subparsers.add_parser("remote")
    remote.add_argument(
        "--tcp-listen", type=parse_endpoint, default=("127.0.0.1", 47999)
    )
    remote.add_argument(
        "--udp-target", type=parse_endpoint, default=("127.0.0.1", 47998)
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "local":
        run_local(args.udp_listen, args.tcp_target)
    else:
        run_remote(args.tcp_listen, args.udp_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
