#!/usr/bin/env python3
"""Simple UDP listener to inspect MORAI packets."""

import socket
import sys


def test_udp_receiver(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(5.0)  # seconds

    try:
        sock.bind(("0.0.0.0", port))
        print(f"[OK] UDP bind success: 0.0.0.0:{port}")
        print("[INFO] Waiting for packets (timeout 5 s per receive)...")

        packet_count = 0
        while packet_count < 10:
            try:
                data, addr = sock.recvfrom(4096)
                packet_count += 1
                print(f"\n[Packet #{packet_count}] from {addr[0]}:{addr[1]}")
                print(f"  size: {len(data)} bytes")
                print(f"  head: {data[:30]}")
            except socket.timeout:
                print(f"\n[WARN] Timeout – no data received on port {port}")
                print("Check MORAI is running, UDP enabled, Destination IP/Port, firewall.")
                break
    except OSError as exc:
        print(f"[ERROR] Bind failed: {exc}")
        print(f"Maybe port {port} is in use; try:")
        print(f"  sudo netstat -tulpn | grep {port}")
    finally:
        sock.close()
        print("\n[INFO] UDP receiver stopped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/test_udp_receiver.py <port>")
        print("  e.g. python3 tools/test_udp_receiver.py 15002")
        sys.exit(1)

    test_udp_receiver(int(sys.argv[1]))
