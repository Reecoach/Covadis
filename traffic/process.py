#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCAP -> 5-tuple flows -> JSON dict (+ protocol stats)

Kept architecture and existing functionality.
Enhancements:
- Per-packet features:
    - timestamp
    - ip_len
    - direction (5-tuple direction)
    - pair_direction (alias of direction)
    - ip_direction (IP-pair first-seen based)
- Prints:
    - protocol share among flows and packets
    - pair_direction packet share + byte share
    - ip_direction packet share + byte share

Notes:
- Flow building: IP (v4/v6) + TCP/UDP only.
- Protocol summary: best-effort, safe (no TLS parsing / no cryptography dependency).
"""

import os
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

from scapy.all import (
    PcapReader,
    IP,
    IPv6,
    TCP,
    UDP,
    DNS,
    Raw,
)

# ============================================================
# Protocol summary (best-effort, safe)
# ============================================================

def pct(count: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * count / total


def counter_with_percentage(counter: Counter, total: int) -> Dict[str, Dict[str, float]]:
    """Convert Counter -> {key: {"count": x, "percentage": y}}"""
    return {
        k: {"count": int(v), "percentage": round(pct(int(v), int(total)), 3)}
        for k, v in counter.items()
    }


def summarize_protocols_from_pcaps(pcap_files: List[str]) -> Dict:
    """
    Summarize observable protocol information from PCAP files.

    Scope:
      - IP only (IPv4 / IPv6)
      - Transport: TCP / UDP
      - Application (observable / heuristic):
          * TCP-HTTP   (cleartext only)
          * TCP-TLS    (port-based, no TLS parsing)
          * TCP-DNS
          * UDP-DNS
          * UDP-QUIC   (UDP + known ports)
          * TCP-UNKNOWN
          * UDP-UNKNOWN
    """
    stats = {
        "total_packets": 0,
        "ip": Counter(),
        "transport": Counter(),
        "application": Counter(),
    }

    QUIC_PORTS = {443, 784, 785, 788, 1700}

    for pcap in pcap_files:
        if not os.path.isfile(pcap):
            continue

        with PcapReader(pcap) as reader:
            for pkt in reader:
                stats["total_packets"] += 1

                # L3
                if pkt.haslayer(IP):
                    stats["ip"]["IPv4"] += 1
                elif pkt.haslayer(IPv6):
                    stats["ip"]["IPv6"] += 1
                else:
                    continue  # non-IP ignored

                # L4 + best-effort L7 label
                if pkt.haslayer(TCP):
                    stats["transport"]["TCP"] += 1

                    if pkt.haslayer(DNS):
                        stats["application"]["TCP-DNS"] += 1
                    elif pkt[TCP].sport == 443 or pkt[TCP].dport == 443:
                        stats["application"]["TCP-TLS"] += 1
                    elif pkt.haslayer(Raw):
                        payload = bytes(pkt[Raw].load)
                        if (
                            payload.startswith(b"GET ")
                            or payload.startswith(b"POST ")
                            or payload.startswith(b"PUT ")
                            or payload.startswith(b"HEAD ")
                            or payload.startswith(b"HTTP/")
                        ):
                            stats["application"]["TCP-HTTP"] += 1
                        else:
                            stats["application"]["TCP-UNKNOWN"] += 1
                    else:
                        stats["application"]["TCP-UNKNOWN"] += 1

                elif pkt.haslayer(UDP):
                    stats["transport"]["UDP"] += 1
                    udp = pkt[UDP]

                    if pkt.haslayer(DNS):
                        stats["application"]["UDP-DNS"] += 1
                    elif udp.sport in QUIC_PORTS or udp.dport in QUIC_PORTS:
                        stats["application"]["UDP-QUIC"] += 1
                    else:
                        stats["application"]["UDP-UNKNOWN"] += 1

    total_packets = stats["total_packets"]
    return {
        "total_packets": int(total_packets),
        "ip": counter_with_percentage(stats["ip"], total_packets),
        "transport": counter_with_percentage(stats["transport"], total_packets),
        "application": counter_with_percentage(stats["application"], total_packets),
    }


def print_protocol_summary(summary: Dict) -> None:
    print("\n========== Protocol Summary ==========")
    print(f"Total packets: {summary['total_packets']}")

    for level in ["ip", "transport", "application"]:
        print(f"\n--- {level.upper()} ---")
        for k, v in summary[level].items():
            print(f"{k:>12}: {v['count']:>10}  ({v['percentage']:6.2f}%)")


# ============================================================
# 1) Collect pcaps
# ============================================================

def collect_pcap_files(directories: List[str]) -> List[str]:
    pcap_files: List[str] = []
    for d in directories:
        if not d:
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".pcap", ".pcapng")):
                    pcap_files.append(os.path.join(root, f))
    return sorted(pcap_files)


# ============================================================
# 2) Packet parsing utilities
# ============================================================

def get_l3_tuple(pkt) -> Optional[Tuple[str, str, int]]:
    """Return (src_ip, dst_ip, ip_version) if IP/IPv6 exists, else None."""
    if pkt.haslayer(IP):
        ip = pkt[IP]
        return ip.src, ip.dst, 4
    if pkt.haslayer(IPv6):
        ip6 = pkt[IPv6]
        return ip6.src, ip6.dst, 6
    return None


def get_transport_tuple(pkt) -> Optional[Tuple[int, int, str]]:
    """Return (sport, dport, proto_str) if TCP/UDP exists, else None."""
    if pkt.haslayer(TCP):
        t = pkt[TCP]
        return int(t.sport), int(t.dport), "TCP"
    if pkt.haslayer(UDP):
        u = pkt[UDP]
        return int(u.sport), int(u.dport), "UDP"
    return None


def get_ip_len(pkt, ip_version: int) -> Optional[int]:
    """
    L3 length:
      IPv4: ip.len (total length including IPv4 header)
      IPv6: payload length + 40 (IPv6 header is 40 bytes)
    """
    try:
        if ip_version == 4:
            return int(pkt[IP].len)
        if ip_version == 6:
            return int(pkt[IPv6].plen) + 40
    except Exception:
        return None
    return None


def canonical_flow_key_from_first_packet(
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    proto: str,
) -> Tuple[str, str, int, int, str]:
    """Define canonical 5-tuple key by the first observed direction in that flow."""
    return (src_ip, dst_ip, sport, dport, proto)


def key_to_str(key: Tuple[str, str, int, int, str]) -> str:
    """JSON-friendly stable flow key string."""
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}|{key[4]}"


# ============================================================
# Direction helpers
# ============================================================

def ip_pair_key(src_ip: str, dst_ip: str) -> Tuple[str, str]:
    """Unordered IP pair key."""
    return tuple(sorted((src_ip, dst_ip)))


def get_ip_direction(
    src_ip: str,
    dst_ip: str,
    ip_pair_ref: Dict[Tuple[str, str], Tuple[str, str]],
) -> int:
    """
    IP-pair-based direction (ports ignored), defined by first-seen packet orientation
    per unordered IP pair.

    - First packet between (ipa, ipb) defines +1 orientation (ref_src -> ref_dst)
    - Reverse direction is -1
    """
    key = ip_pair_key(src_ip, dst_ip)
    if key not in ip_pair_ref:
        ip_pair_ref[key] = (src_ip, dst_ip)
        return 1

    ref_src, ref_dst = ip_pair_ref[key]
    return 1 if (src_ip == ref_src and dst_ip == ref_dst) else -1


# ============================================================
# 3) Parse a single pcap into flows
# ============================================================

def parse_pcap_to_flows(
    pcap_path: str,
    *,
    include_ipv6: bool = True,
) -> Tuple[Dict[str, List[dict]], dict]:
    """
    Returns:
      flows: dict[str_key] -> list[packet_feature_dict]
      stats: dict with counters

    Each packet feature dict contains:
      - timestamp
      - ip_len
      - direction       (original 5-tuple direction)
      - pair_direction  (alias of direction)
      - ip_direction    (IP-pair direction, ports ignored)
    """
    flows: Dict[str, List[dict]] = defaultdict(list)

    # Map directional 5-tuples to canonical flow key string.
    dir2canon: Dict[Tuple[str, str, int, int, str], str] = {}

    # Track first-seen orientation per unordered IP pair
    ip_pair_ref: Dict[Tuple[str, str], Tuple[str, str]] = {}

    stats = {
        "pcap": pcap_path,
        "packets_total": 0,
        "packets_ip": 0,
        "packets_ip_skipped_non_tcpudp": 0,
        "packets_ipv6_skipped": 0,
        "packets_l3len_missing": 0,
        "flows_new": 0,
    }

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            stats["packets_total"] += 1

            l3 = get_l3_tuple(pkt)
            if l3 is None:
                continue

            src_ip, dst_ip, ip_ver = l3
            if ip_ver == 6 and not include_ipv6:
                stats["packets_ipv6_skipped"] += 1
                continue

            stats["packets_ip"] += 1

            l4 = get_transport_tuple(pkt)
            if l4 is None:
                stats["packets_ip_skipped_non_tcpudp"] += 1
                continue

            sport, dport, proto = l4

            ip_len = get_ip_len(pkt, ip_ver)
            if ip_len is None:
                stats["packets_l3len_missing"] += 1
                continue

            t = float(pkt.time)

            dir_key = (src_ip, dst_ip, sport, dport, proto)
            rev_key = (dst_ip, src_ip, dport, sport, proto)

            # 5-tuple flow direction (pair_direction)
            if dir_key in dir2canon:
                canon_str = dir2canon[dir_key]
                pair_direction = 1
            elif rev_key in dir2canon:
                canon_str = dir2canon[rev_key]
                pair_direction = -1
            else:
                canon = canonical_flow_key_from_first_packet(src_ip, dst_ip, sport, dport, proto)
                canon_str = key_to_str(canon)

                dir2canon[dir_key] = canon_str
                dir2canon[rev_key] = canon_str
                stats["flows_new"] += 1
                pair_direction = 1

            # IP-pair direction (ports ignored)
            ip_direction = get_ip_direction(src_ip, dst_ip, ip_pair_ref)

            # Keep "direction" for backward compatibility, and add explicit names
            flows[canon_str].append(
                {
                    "timestamp": t,
                    "ip_len": int(ip_len),
                    "direction": int(pair_direction),
                    "pair_direction": int(pair_direction),
                    "ip_direction": int(ip_direction),
                }
            )

    return flows, stats


# ============================================================
# 4) Process multiple pcaps and export JSON
# ============================================================

def process_pcaps(
    directories: List[str],
    output_json: str,
    *,
    include_ipv6: bool = True,
) -> None:
    pcap_files = collect_pcap_files(directories)
    if not pcap_files:
        print("No pcap/pcapng files found.")
        return

    all_flows: Dict[str, List[dict]] = defaultdict(list)

    # Global stats
    total_stats = Counter()
    proto_flows = Counter()
    proto_packets = Counter()

    # Direction stats (packet-level)
    pair_direction_packets = Counter()
    ip_direction_packets = Counter()

    # Direction stats (byte-level, sum of ip_len)
    pair_direction_bytes = Counter()
    ip_direction_bytes = Counter()

    for i, pcap_path in enumerate(pcap_files, 1):
        print(f"[{i}/{len(pcap_files)}] Parsing: {pcap_path}")

        flows, stats = parse_pcap_to_flows(pcap_path, include_ipv6=include_ipv6)

        # Merge flows + direction stats
        for _, pkt_list in flows.items():
            for p in pkt_list:
                pd = int(p.get("pair_direction", p.get("direction", 0)))
                idd = int(p.get("ip_direction", 0))
                bl = int(p.get("ip_len", 0))

                pair_direction_packets[pd] += 1
                ip_direction_packets[idd] += 1

                pair_direction_bytes[pd] += bl
                ip_direction_bytes[idd] += bl

        for k, v in flows.items():
            all_flows[k].extend(v)

        # Accumulate counters
        total_stats.update({k: v for k, v in stats.items() if isinstance(v, int)})

        # Protocol share among flows (5-tuples) and packets
        for flow_key_str, pkt_list in flows.items():
            proto = flow_key_str.rsplit("|", 1)[-1]
            proto_flows[proto] += 1
            proto_packets[proto] += len(pkt_list)

        print(
            f"  flows_new={stats['flows_new']}, "
            f"packets_total={stats['packets_total']}, "
            f"packets_ip={stats['packets_ip']}, "
            f"non_tcpudp={stats['packets_ip_skipped_non_tcpudp']}"
        )

    # Dump JSON
    out_obj = dict(all_flows)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # Final prints
    print("\n========== DONE ==========")
    print(f"PCAP files: {len(pcap_files)}")
    print(f"Total packets: {total_stats['packets_total']}")
    print(f"Total IP/IPv6 packets: {total_stats['packets_ip']}")
    print(f"Total skipped non-TCP/UDP (within IP): {total_stats['packets_ip_skipped_non_tcpudp']}")
    print(f"Total flows (unique canonical 5-tuples): {len(all_flows)}")
    print(f"Output JSON: {output_json}")

    # Protocol share among flows
    print("\n--- Protocol share among flows (5-tuples) ---")
    total_flow_count = sum(proto_flows.values()) or 1
    for proto, cnt in proto_flows.most_common():
        print(f"{proto:>4}: {cnt:>10} flows  ({pct(cnt, total_flow_count):6.2f}%)")

    # Protocol share among packets (extra)
    print("\n--- Protocol share among packets (extra) ---")
    total_pkt_count = sum(proto_packets.values()) or 1
    for proto, cnt in proto_packets.most_common():
        print(f"{proto:>4}: {cnt:>10} packets ({pct(cnt, total_pkt_count):6.2f}%)")

    # Pair-direction share among packets
    print("\n--- Pair-direction share among packets (5-tuple based) ---")
    total_pair_dir_pkts = sum(pair_direction_packets.values()) or 1
    pos_cnt = pair_direction_packets.get(1, 0)
    neg_cnt = pair_direction_packets.get(-1, 0)
    other_cnt = total_pair_dir_pkts - pos_cnt - neg_cnt
    print(f"+1: {pos_cnt:>10} packets ({pct(pos_cnt, total_pair_dir_pkts):6.2f}%)")
    print(f"-1: {neg_cnt:>10} packets ({pct(neg_cnt, total_pair_dir_pkts):6.2f}%)")
    if other_cnt > 0:
        print(f" 0/other: {other_cnt:>10} packets ({pct(other_cnt, total_pair_dir_pkts):6.2f}%)")

    # Pair-direction share among bytes (sum ip_len)
    print("\n--- Pair-direction share among bytes (sum ip_len, 5-tuple based) ---")
    total_pair_dir_bytes = sum(pair_direction_bytes.values()) or 1
    pos_b = pair_direction_bytes.get(1, 0)
    neg_b = pair_direction_bytes.get(-1, 0)
    other_b = total_pair_dir_bytes - pos_b - neg_b
    print(f"+1: {pos_b:>10} bytes  ({pct(pos_b, total_pair_dir_bytes):6.2f}%)")
    print(f"-1: {neg_b:>10} bytes  ({pct(neg_b, total_pair_dir_bytes):6.2f}%)")
    if other_b > 0:
        print(f" 0/other: {other_b:>10} bytes  ({pct(other_b, total_pair_dir_bytes):6.2f}%)")

    # IP-direction share among packets
    print("\n--- IP-direction share among packets (IP-pair first-seen based) ---")
    total_ip_dir_pkts = sum(ip_direction_packets.values()) or 1
    pos_cnt = ip_direction_packets.get(1, 0)
    neg_cnt = ip_direction_packets.get(-1, 0)
    other_cnt = total_ip_dir_pkts - pos_cnt - neg_cnt
    print(f"+1: {pos_cnt:>10} packets ({pct(pos_cnt, total_ip_dir_pkts):6.2f}%)")
    print(f"-1: {neg_cnt:>10} packets ({pct(neg_cnt, total_ip_dir_pkts):6.2f}%)")
    if other_cnt > 0:
        print(f" 0/other: {other_cnt:>10} packets ({pct(other_cnt, total_ip_dir_pkts):6.2f}%)")

    # IP-direction share among bytes (sum ip_len)
    print("\n--- IP-direction share among bytes (sum ip_len, IP-pair first-seen based) ---")
    total_ip_dir_bytes = sum(ip_direction_bytes.values()) or 1
    pos_b = ip_direction_bytes.get(1, 0)
    neg_b = ip_direction_bytes.get(-1, 0)
    other_b = total_ip_dir_bytes - pos_b - neg_b
    print(f"+1: {pos_b:>10} bytes  ({pct(pos_b, total_ip_dir_bytes):6.2f}%)")
    print(f"-1: {neg_b:>10} bytes  ({pct(neg_b, total_ip_dir_bytes):6.2f}%)")
    if other_b > 0:
        print(f" 0/other: {other_b:>10} bytes  ({pct(other_b, total_ip_dir_bytes):6.2f}%)")


# ============================================================
# Script entry
# ============================================================

if __name__ == "__main__":
    DATASET_NAME = "/home/lihaozhi/dataset/NonTor"
    APP_TYPES = ["video_streaming", "audio_streaming"]
    PARSED_OUTPUT_DIR = "parsed"

    for app_type in APP_TYPES:
        dataset_dir = f"{DATASET_NAME}/{app_type}"
        print(f"\nProcessing {dataset_dir}")

        pcap_files_in_dir = collect_pcap_files([dataset_dir])

        stats = summarize_protocols_from_pcaps(pcap_files_in_dir)
        print_protocol_summary(stats)

        process_pcaps(
            directories=[dataset_dir],
            output_json=f"{PARSED_OUTPUT_DIR}/{DATASET_NAME}-{app_type}-parsed.json",
            include_ipv6=True,
        )
