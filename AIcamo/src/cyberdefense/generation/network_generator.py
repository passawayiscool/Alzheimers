"""Network traffic generation and PCAP export"""

import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

try:
    from scapy.all import (
        IP,
        TCP,
        UDP,
        DNS,
        DNSQR,
        Ether,
        wrpcap,
        Raw,
    )
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available, network generation will be limited")

logger = logging.getLogger(__name__)


class NetworkTrafficGenerator:
    """
    Generates realistic network traffic based on host events.

    Creates PCAP files for testing NIDS systems.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize network traffic generator.

        Args:
            config: Configuration dictionary
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for network traffic generation")

        self.config = config
        self.network_config = config.get("network", {})

        # Common IP ranges for simulation
        self.internal_ips = self._generate_internal_ips()
        self.external_ips = self._generate_external_ips()

    def _generate_internal_ips(self, count: int = 50) -> List[str]:
        """Generate internal IP addresses (private ranges)"""
        ips = []
        for i in range(count):
            ips.append(f"192.168.1.{random.randint(10, 254)}")
        return ips

    def _generate_external_ips(self, count: int = 100) -> List[str]:
        """Generate external IP addresses (public ranges)"""
        ips = []
        for i in range(count):
            ips.append(f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}")
        return ips

    def generate_from_events(
        self, events: List[Dict[str, Any]], output_path: str
    ) -> int:
        """
        Generate network traffic from host events.

        Args:
            events: List of host events
            output_path: Path to save PCAP file

        Returns:
            Number of packets generated
        """
        logger.info(f"Generating network traffic from {len(events)} events...")

        packets = []

        for event in events:
            event_type = event.get("event_type", "")

            # Generate network packets for network events
            if event_type == "network_connect":
                pkts = self._generate_connection_packets(event)
                packets.extend(pkts)

            elif event_type == "network_dns":
                pkts = self._generate_dns_packets(event)
                packets.extend(pkts)

            # Generate background traffic for other events
            elif random.random() < 0.1:  # 10% chance of background traffic
                pkts = self._generate_background_traffic()
                packets.extend(pkts)

        # Save to PCAP
        if packets:
            wrpcap(output_path, packets)
            logger.info(f"Saved {len(packets)} packets to {output_path}")

        return len(packets)

    def generate_profile_traffic(
        self, profile_name: str, duration: int, output_path: str
    ) -> int:
        """
        Generate traffic for a specific profile.

        Args:
            profile_name: Name of traffic profile
            duration: Duration in seconds
            output_path: Path to save PCAP file

        Returns:
            Number of packets generated
        """
        profiles = self.network_config.get("profiles", {})
        profile = profiles.get(profile_name)

        if not profile:
            logger.error(f"Profile '{profile_name}' not found")
            return 0

        logger.info(f"Generating '{profile_name}' traffic for {duration} seconds...")

        packets = []
        start_time = time.time()
        packet_time = start_time

        while packet_time - start_time < duration:
            # Generate packet based on profile
            if profile["protocol"] == "https":
                pkt = self._generate_https_packet(profile)
            elif profile["protocol"] == "smb":
                pkt = self._generate_smb_packet(profile)
            elif profile["protocol"] == "dns":
                pkt = self._generate_dns_query()
            else:
                pkt = self._generate_tcp_packet(profile)

            packets.append(pkt)

            # Random inter-packet delay
            packet_time += random.uniform(0.001, 0.1)

        # Save to PCAP
        if packets:
            wrpcap(output_path, packets)
            logger.info(f"Saved {len(packets)} packets to {output_path}")

        return len(packets)

    def _generate_connection_packets(self, event: Dict[str, Any]) -> List:
        """Generate TCP connection packets"""
        packets = []

        src_ip = event.get("local_address", random.choice(self.internal_ips))
        dst_ip = event.get("remote_address", random.choice(self.external_ips))
        src_port = event.get("local_port", random.randint(49152, 65535))
        dst_port = event.get("remote_port", 443)

        # TCP handshake (SYN, SYN-ACK, ACK)
        packets.append(
            Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=src_port, dport=dst_port, flags="S")
        )
        packets.append(
            Ether() / IP(src=dst_ip, dst=src_ip) / TCP(sport=dst_port, dport=src_port, flags="SA")
        )
        packets.append(
            Ether() / IP(src=src_ip, dst=dst_ip) / TCP(sport=src_port, dport=dst_port, flags="A")
        )

        # Data transfer (1-5 packets)
        for _ in range(random.randint(1, 5)):
            data = b"X" * random.randint(64, 1460)
            packets.append(
                Ether()
                / IP(src=src_ip, dst=dst_ip)
                / TCP(sport=src_port, dport=dst_port, flags="PA")
                / Raw(load=data)
            )

        return packets

    def _generate_dns_packets(self, event: Dict[str, Any]) -> List:
        """Generate DNS query packets"""
        packets = []

        src_ip = event.get("local_address", random.choice(self.internal_ips))
        dns_server = "8.8.8.8"
        query = event.get("dns_query", "example.com")

        # DNS query
        packets.append(
            Ether()
            / IP(src=src_ip, dst=dns_server)
            / UDP(sport=random.randint(49152, 65535), dport=53)
            / DNS(rd=1, qd=DNSQR(qname=query))
        )

        return packets

    def _generate_https_packet(self, profile: Dict[str, Any]) -> Any:
        """Generate HTTPS packet"""
        src_ip = random.choice(self.internal_ips)
        dst_ip = random.choice(self.external_ips)
        src_port = random.randint(49152, 65535)
        dst_port = 443

        size_range = profile.get("packet_size_range", [64, 1460])
        data_size = random.randint(*size_range)

        return (
            Ether()
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=src_port, dport=dst_port, flags="PA")
            / Raw(load=b"X" * data_size)
        )

    def _generate_smb_packet(self, profile: Dict[str, Any]) -> Any:
        """Generate SMB packet"""
        src_ip = random.choice(self.internal_ips)
        dst_ip = random.choice(self.internal_ips)  # SMB typically internal
        src_port = random.randint(49152, 65535)
        dst_port = 445

        size_range = profile.get("packet_size_range", [64, 4096])
        data_size = random.randint(*size_range)

        return (
            Ether()
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=src_port, dport=dst_port, flags="PA")
            / Raw(load=b"X" * data_size)
        )

    def _generate_tcp_packet(self, profile: Dict[str, Any]) -> Any:
        """Generate generic TCP packet"""
        src_ip = random.choice(self.internal_ips)
        dst_ip = random.choice(self.external_ips)
        src_port = random.randint(49152, 65535)
        dst_port = random.choice(profile.get("ports", [80, 443]))

        size_range = profile.get("packet_size_range", [64, 1460])
        data_size = random.randint(*size_range)

        return (
            Ether()
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=src_port, dport=dst_port, flags="PA")
            / Raw(load=b"X" * data_size)
        )

    def _generate_dns_query(self) -> Any:
        """Generate random DNS query"""
        domains = [
            "google.com",
            "microsoft.com",
            "github.com",
            "stackoverflow.com",
            "reddit.com",
        ]

        src_ip = random.choice(self.internal_ips)
        dns_server = "8.8.8.8"
        query = random.choice(domains)

        return (
            Ether()
            / IP(src=src_ip, dst=dns_server)
            / UDP(sport=random.randint(49152, 65535), dport=53)
            / DNS(rd=1, qd=DNSQR(qname=query))
        )

    def _generate_background_traffic(self) -> List:
        """Generate background network traffic"""
        packets = []

        # Random traffic type
        traffic_type = random.choice(["https", "dns", "http"])

        if traffic_type == "https":
            packets.append(self._generate_https_packet({"ports": [443], "packet_size_range": [64, 1460]}))
        elif traffic_type == "dns":
            packets.append(self._generate_dns_query())
        else:
            packets.append(self._generate_tcp_packet({"ports": [80], "packet_size_range": [64, 1460]}))

        return packets
