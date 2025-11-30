"""Correlates host events with network traffic for realistic simulation"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class HostNetworkCorrelator:
    """
    Correlates host-based events with network traffic.

    Ensures that network traffic is generated in sync with
    corresponding host events for realistic simulation.
    """

    def __init__(self):
        """Initialize correlator"""
        self.correlation_rules = self._build_correlation_rules()

    def _build_correlation_rules(self) -> Dict[str, List[str]]:
        """
        Build rules for correlating host events with network traffic.

        Returns:
            Dictionary mapping host event patterns to network events
        """
        return {
            "browser": {
                "processes": ["chrome.exe", "firefox.exe", "msedge.exe"],
                "network_types": ["https", "dns", "http"],
                "frequency": "high",
            },
            "cloud_sync": {
                "processes": ["onedrive.exe", "dropbox.exe", "googledrivesync.exe"],
                "network_types": ["https"],
                "frequency": "medium",
            },
            "system_update": {
                "processes": ["wuauclt.exe", "update.exe", "windowsupdate.exe"],
                "network_types": ["https"],
                "frequency": "low",
            },
            "remote_desktop": {
                "processes": ["mstsc.exe", "rdpclip.exe"],
                "network_types": ["rdp"],
                "frequency": "low",
            },
            "file_sharing": {
                "processes": ["explorer.exe"],
                "file_paths": ["\\\\"],  # UNC paths
                "network_types": ["smb"],
                "frequency": "medium",
            },
        }

    def correlate(
        self, host_events: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Correlate host events and generate corresponding network events.

        Args:
            host_events: List of host-based events

        Returns:
            Tuple of (correlated_host_events, network_events)
        """
        logger.info(f"Correlating {len(host_events)} host events with network traffic...")

        correlated_host = []
        network_events = []

        for event in host_events:
            correlated_host.append(event)

            # Check if event should generate network traffic
            net_event = self._should_generate_network(event)

            if net_event:
                network_events.append(net_event)

        logger.info(f"Generated {len(network_events)} correlated network events")

        return correlated_host, network_events

    def _should_generate_network(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if host event should generate network traffic.

        Args:
            event: Host event

        Returns:
            Network event dictionary or None
        """
        event_type = event.get("event_type", "")
        process_name = event.get("process_name", "").lower()

        # Already a network event
        if event_type in ["network_connect", "network_dns"]:
            return event

        # Check correlation rules
        for category, rules in self.correlation_rules.items():
            # Check if process matches
            processes = rules.get("processes", [])
            if any(proc.lower() in process_name for proc in processes):
                frequency = rules.get("frequency", "medium")

                # Probabilistic generation based on frequency
                prob = {"high": 0.8, "medium": 0.4, "low": 0.1}.get(frequency, 0.2)

                if random.random() < prob:
                    # Generate network event
                    network_types = rules.get("network_types", ["https"])
                    net_type = random.choice(network_types)

                    return self._create_network_event(event, net_type)

        return None

    def _create_network_event(
        self, host_event: Dict[str, Any], network_type: str
    ) -> Dict[str, Any]:
        """
        Create network event from host event.

        Args:
            host_event: Source host event
            network_type: Type of network traffic

        Returns:
            Network event dictionary
        """
        timestamp = host_event.get("timestamp", datetime.now().isoformat())

        if network_type == "https":
            return {
                "event_type": "network_connect",
                "timestamp": timestamp,
                "pid": host_event.get("pid", 0),
                "process_name": host_event.get("process_name", "unknown"),
                "protocol": "TCP",
                "local_address": "192.168.1.100",
                "local_port": random.randint(49152, 65535),
                "remote_address": f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                "remote_port": 443,
            }

        elif network_type == "dns":
            domains = [
                "google.com",
                "microsoft.com",
                "github.com",
                "stackoverflow.com",
            ]
            return {
                "event_type": "network_dns",
                "timestamp": timestamp,
                "pid": host_event.get("pid", 0),
                "process_name": host_event.get("process_name", "unknown"),
                "dns_query": random.choice(domains),
                "local_address": "192.168.1.100",
            }

        elif network_type == "smb":
            return {
                "event_type": "network_connect",
                "timestamp": timestamp,
                "pid": host_event.get("pid", 0),
                "process_name": host_event.get("process_name", "unknown"),
                "protocol": "TCP",
                "local_address": "192.168.1.100",
                "local_port": random.randint(49152, 65535),
                "remote_address": f"192.168.1.{random.randint(10, 254)}",
                "remote_port": 445,
            }

        else:
            # Default HTTP
            return {
                "event_type": "network_connect",
                "timestamp": timestamp,
                "pid": host_event.get("pid", 0),
                "process_name": host_event.get("process_name", "unknown"),
                "protocol": "TCP",
                "local_address": "192.168.1.100",
                "local_port": random.randint(49152, 65535),
                "remote_address": f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                "remote_port": 80,
            }

    def add_timing_jitter(
        self, events: List[Dict[str, Any]], max_jitter_ms: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Add realistic timing jitter to events.

        Args:
            events: List of events
            max_jitter_ms: Maximum jitter in milliseconds

        Returns:
            Events with adjusted timestamps
        """
        jittered = []

        for event in events:
            timestamp_str = event.get("timestamp", "")

            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    jitter = random.randint(-max_jitter_ms, max_jitter_ms)
                    new_timestamp = timestamp + timedelta(milliseconds=jitter)
                    event["timestamp"] = new_timestamp.isoformat()
                except:
                    pass

            jittered.append(event)

        return jittered
