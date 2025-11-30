import pandas as pd
import random
import time
from pathlib import Path
import logging
from typing import List, Dict, Optional
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTS ---
USERS = ["{username}"]
HOSTS = ["{hostname}"]
DOMAINS = [
    "google.com", "github.com", "microsoft.com", "api.internal", 
    "stackoverflow.com", "jira.corp.local", "update.windows.com",
    "drive.google.com", "slack.com", "discord.com"
]

class NetworkChain:
    def __init__(self, name: str, process: str, domain: str, port: int, weight: int = 1):
        self.name = name
        self.process = process
        self.domain = domain
        self.port = port
        self.weight = weight

# Define Realistic Network Chains
import ipaddress

# Define Realistic Network Chains
# Based on actual observation of the host
CHAINS = [
    NetworkChain("Chrome Browsing", "chrome.exe", "google.com", 443, weight=10),
    NetworkChain("Chrome MTalk", "chrome.exe", "mtalk.google.com", 5228, weight=5), # Observed
    NetworkChain("Edge Browsing", "msedge.exe", "bing.com", 443, weight=8),
    NetworkChain("GitHub Push", "git.exe", "github.com", 443, weight=5),
    NetworkChain("Windows Update", "svchost.exe", "update.windows.com", 443, weight=5),
    NetworkChain("Slack API", "slack.exe", "slack.com", 443, weight=3),
    NetworkChain("Discord Voice", "Discord.exe", "discord.com", 50000, weight=2),
    NetworkChain("DNS Lookup", "svchost.exe", "8.8.8.8", 53, weight=4),
    NetworkChain("Search App", "SearchApp.exe", "traffic.bing.com", 443, weight=6), # Observed
    NetworkChain("Copilot", "M365Copilot.exe", "copilot.microsoft.com", 443, weight=4), # Observed
    NetworkChain("Telemetry", "svchost.exe", "settings-win.data.microsoft.com", 443, weight=7),
    NetworkChain("OneDrive Sync", "OneDrive.exe", "skyapi.onedrive.live.com", 443, weight=6),
    NetworkChain("Outlook Sync", "olk.exe", "outlook.office365.com", 443, weight=5),
    NetworkChain("SmartScreen", "smartscreen.exe", "smartscreen.microsoft.com", 443, weight=3),
    NetworkChain("Antigravity Check", "chrome.exe", "antigravity-unleash.goog", 443, weight=2),
]

# CIDR Blocks for Dynamic IP Generation
DOMAIN_IP_RANGES = {
    "google.com": ["142.250.0.0/15", "172.217.0.0/16"],
    "mtalk.google.com": ["142.250.0.0/15", "172.217.0.0/16"],
    "bing.com": ["204.79.197.0/24", "13.107.21.0/24"],
    "github.com": ["140.82.112.0/20"],
    "update.windows.com": ["20.0.0.0/11", "51.0.0.0/10"], # Azure ranges
    "slack.com": ["3.0.0.0/9", "54.0.0.0/8"], # AWS ranges
    "discord.com": ["162.158.0.0/15", "35.200.0.0/13"], # Cloudflare/GCP
    "8.8.8.8": ["8.8.8.8/32"],
    "traffic.bing.com": ["204.79.197.0/24"],
    "copilot.microsoft.com": ["20.0.0.0/10"],
    "settings-win.data.microsoft.com": ["51.10.0.0/14"],
    "skyapi.onedrive.live.com": ["13.107.0.0/16"],
    "outlook.office365.com": ["52.96.0.0/14", "40.96.0.0/12"],
    "smartscreen.microsoft.com": ["23.192.0.0/11"], # Akamai/Azure
    "antigravity-unleash.goog": ["142.250.0.0/15"],
    "api.internal": ["10.0.0.0/24"],
    "stackoverflow.com": ["151.101.0.0/16"], # Fastly
    "jira.corp.local": ["192.168.10.0/24"],
    "drive.google.com": ["142.250.0.0/15"]
}

def get_random_ip(domain):
    """Generates a random IP from the domain's CIDR blocks."""
    ranges = DOMAIN_IP_RANGES.get(domain, ["1.1.1.1/32"])
    cidr_str = random.choice(ranges)
    try:
        network = ipaddress.ip_network(cidr_str)
        # Pick a random integer within the network range
        # network.num_addresses - 1 to avoid broadcast if applicable, though for /32 it's 1
        num_addrs = network.num_addresses
        if num_addrs <= 1:
            return str(network.network_address)
        
        random_int = int(network.network_address) + random.randint(0, num_addrs - 1)
        return str(ipaddress.IPv4Address(random_int))
    except ValueError:
        return "1.1.1.1"

# Artifact Paths
ARTIFACT_PATHS = {
    "SECURITY_LOG": r"C:\Windows\System32\winevt\Logs\Security.evtx",
    "DNS_LOG": r"C:\Windows\System32\winevt\Logs\Microsoft-Windows-DNS-Client%4Operational.evtx",
    "CHROME_HISTORY": r"C:\Users\{user}\AppData\Local\Google\Chrome\User Data\Default\History",
    "EDGE_HISTORY": r"C:\Users\{user}\AppData\Local\Microsoft\Edge\User Data\Default\History"
}

def generate_host_network_data(output_path, num_events=100000):
    logger.info(f"Generating {num_events} HOST-BASED NETWORK ARTIFACT events...")
    
    data = []
    current_time = time.time()
    
    i = 0
    while i < num_events:
        # Pick a chain
        chain = random.choices(CHAINS, weights=[c.weight for c in CHAINS], k=1)[0]
        user = random.choice(USERS)
        host = random.choice(HOSTS)
        
        # Resolve IP (dynamic)
        dest_ip = get_random_ip(chain.domain)

        # 1. PROCESS START (The trigger)
        proc_event = {
            "timestamp": current_time,
            "event_type": "PROCESS_START",
            "user": user,
            "host": host,
            "process_name": chain.process,
            "parent_process": "explorer.exe",
            "command_line": f'"{chain.process}" --url={chain.domain}',
            "artifact_path": ARTIFACT_PATHS["SECURITY_LOG"]
        }
        data.append(proc_event)
        i += 1
        current_time += 0.1
        
        # 2. DNS CACHE ARTIFACT (Resolve-DnsName)
        dns_event = {
            "timestamp": current_time,
            "event_type": "DNS_QUERY",
            "user": user,
            "host": host,
            "process_name": chain.process,
            "query_name": chain.domain,
            "query_type": "A",
            "query_status": "0", # Success
            "artifact_path": ARTIFACT_PATHS["DNS_LOG"]
        }
        data.append(dns_event)
        i += 1
        current_time += 0.05
        
        # 4. BROWSER HISTORY (Interleaved)
        # Randomly decide to log history BEFORE or AFTER network to break the strict sequence
        log_history_now = False
        if chain.process in ["chrome.exe", "msedge.exe"]:
            log_history_now = True
            
        if log_history_now and random.random() < 0.5:
             # Log History BEFORE Network (Navigation Start)
            hist_path = ARTIFACT_PATHS["CHROME_HISTORY"] if chain.process == "chrome.exe" else ARTIFACT_PATHS["EDGE_HISTORY"]
            hist_event = {
                "timestamp": current_time,
                "event_type": "BROWSER_HISTORY",
                "user": user,
                "host": host,
                "process_name": chain.process,
                "url": f"https://{chain.domain}/",
                "title": f"{chain.domain} - Visited",
                "artifact_path": hist_path.format(user=user)
            }
            data.append(hist_event)
            i += 1
            current_time += 0.05

        # 3. NETWORK CONNECTION LOG (Event ID 5156)
        net_event = {
            "timestamp": current_time,
            "event_type": "NETWORK_CONNECTION", # Maps to Event ID 5156
            "user": user,
            "host": host,
            "process_name": chain.process,
            "dest_address": dest_ip, # Use IP now!
            "dest_port": str(chain.port),
            "protocol": "6" if chain.port != 53 else "17", # TCP vs UDP
            "direction": "Outbound",
            "event_id": "5156", # Explicitly include the ID we want to fake
            "artifact_path": ARTIFACT_PATHS["SECURITY_LOG"]
        }
        data.append(net_event)
        i += 1
        current_time += 0.05
        
        # Log History AFTER Network (if not done before)
        if log_history_now and random.random() >= 0.5:
            hist_path = ARTIFACT_PATHS["CHROME_HISTORY"] if chain.process == "chrome.exe" else ARTIFACT_PATHS["EDGE_HISTORY"]
            hist_event = {
                "timestamp": current_time,
                "event_type": "BROWSER_HISTORY",
                "user": user,
                "host": host,
                "process_name": chain.process,
                "url": f"https://{chain.domain}/",
                "title": f"{chain.domain} - Visited",
                "artifact_path": hist_path.format(user=user)
            }
            data.append(hist_event)
            i += 1
            current_time += 0.1
            
        # 5. BACKGROUND NOISE (Random Keep-Alives & History)
        if random.random() < 0.3:
            # Network Noise
            noise_chain = random.choice([c for c in CHAINS if c.process == "svchost.exe"])
            noise_ip = get_random_ip(noise_chain.domain)
            noise_event = {
                "timestamp": current_time + random.uniform(0.1, 0.5),
                "event_type": "NETWORK_CONNECTION",
                "user": "SYSTEM",
                "host": host,
                "process_name": noise_chain.process,
                "dest_address": noise_ip,
                "dest_port": str(noise_chain.port),
                "protocol": "6",
                "direction": "Outbound",
                "event_id": "5156",
                "artifact_path": ARTIFACT_PATHS["SECURITY_LOG"]
            }
            data.append(noise_event)
            i += 1
            
        # History Noise (User restoring session / typing) - DRASTICALLY INCREASED FOR TRAINING
        if random.random() < 0.8: # Was 0.2
            noise_user = random.choice(USERS)
            noise_domain = random.choice(DOMAINS)
            hist_event = {
                "timestamp": current_time + random.uniform(0.1, 0.5),
                "event_type": "BROWSER_HISTORY",
                "user": noise_user,
                "host": host,
                "process_name": "chrome.exe",
                "url": f"https://{noise_domain}/search?q=test",
                "title": f"{noise_domain} - Search",
                "artifact_path": ARTIFACT_PATHS["CHROME_HISTORY"].format(user=noise_user)
            }
            data.append(hist_event)
            i += 1

        # Random time gap
        current_time += random.uniform(1.0, 5.0)
        
        if i % 10000 == 0:
            logger.info(f"Generated {i} events...")

    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    generate_host_network_data("data/processed/host_network_train.parquet", num_events=50000)
