import time
import random
import logging
import torch
from pathlib import Path

try:
    from scapy.all import IP, TCP, UDP, Raw, Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from cyberdefense.models.network_gan import NetworkGAN

logger = logging.getLogger(__name__)

class TrafficCamouflage:
    """
    Engine for generating camouflaged network traffic.
    Uses NetworkGAN to determine packet timing/sizes, and embeds C2 payloads
    into benign protocol templates.
    """
    def __init__(self, model=None, model_path=None):
        self.model = model
        
        # Legacy support for path loading if needed, but we prefer passing the object
        if not self.model and model_path and Path(model_path).exists():
            try:
                # Placeholder for self-loading if implemented later
                pass
            except Exception as e:
                logger.warning(f"Failed to load NetworkGAN: {e}")
        
        self.templates = {
            "HTTPS_GOOGLE_DRIVE": {
                "dst_port": 443,
                "domain": "drive.google.com",
                "avg_size": 800,
                "burst_len": 10
            },
            "HTTP_WINDOWS_UPDATE": {
                "dst_port": 80,
                "domain": "download.windowsupdate.com",
                "avg_size": 1200,
                "burst_len": 5
            },
            "DNS_QUERY": {
                "dst_port": 53,
                "domain": "8.8.8.8",
                "avg_size": 100,
                "burst_len": 1
            },
            "QUIC_BROWSING": {
                "dst_port": 443,
                "domain": "google.com",
                "avg_size": 1200,
                "burst_len": 8,
                "protocol": "UDP" # Explicitly use UDP for QUIC
            }
        }

    def generate_session(self, payload: str, template_name="HTTPS_GOOGLE_DRIVE", dst_ip="8.8.8.8", condition=None):
        """
        Generate a sequence of Scapy packets representing a session.
        Uses NetworkGAN to generate realistic timing and sizing.
        condition: Host latent vector (torch.Tensor) to condition generation.
        """
        if not SCAPY_AVAILABLE:
            logger.warning("Scapy not available. Cannot generate packets.")
            return []

        template = self.templates.get(template_name, self.templates["HTTPS_GOOGLE_DRIVE"])
        packets = []
        
        # Determine Protocol
        proto_layer = TCP
        if template.get("protocol") == "UDP":
            proto_layer = UDP
            
        # 1. Handshake (Only for TCP)
        if proto_layer == TCP:
            syn = IP(dst=dst_ip)/TCP(dport=template["dst_port"], flags="S")
            packets.append(syn)
        
        # 2. Generate Non-Functional Features using GAN
        burst_len = template["burst_len"]
        
        # If model is loaded, use it. Else fallback to heuristics (but better ones)
        if self.model:
            try:
                # Generate features: [TimeDelta, Size, Flags]
                # We feed a random noise vector z
                with torch.no_grad():
                    # Pass condition if available
                    features = self.model.generate(batch_size=1, condition=condition) # [1, seq_len, 4]
                    features = features.squeeze(0).numpy()
            except:
                features = None
        else:
            features = None

        # Split payload
        payload_bytes = payload.encode('utf-8')
        chunk_size = len(payload_bytes) // burst_len + 1
        
        for i in range(burst_len):
            # Functional Component: The Payload
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(payload_bytes))
            chunk = payload_bytes[start:end]
            
            # Non-Functional Component: Size & Timing
            if features is not None and i < len(features):
                # Use GAN output (normalized)
                # feat[0] = TimeDelta, feat[1] = Size
                gan_size = int(abs(features[i][1]) * 1500) # Scale to MTU
                target_size = max(len(chunk) + 20, gan_size)
            else:
                # Fallback Heuristic
                target_size = int(random.gauss(template["avg_size"], 100))
            
            # Construct Packet
            padding_len = max(0, target_size - len(chunk))
            padding = b'\x00' * padding_len
            data = chunk + padding
            
            if proto_layer == TCP:
                pkt = IP(dst=dst_ip)/TCP(dport=template["dst_port"], flags="PA")/Raw(load=data)
            else:
                pkt = IP(dst=dst_ip)/UDP(dport=template["dst_port"])/Raw(load=data)
                
            packets.append(pkt)
            
        # 3. Teardown (Only for TCP)
        if proto_layer == TCP:
            fin = IP(dst=dst_ip)/TCP(dport=template["dst_port"], flags="FA")
            packets.append(fin)
        
        return packets

    def _embed_payload(self, packet, payload):
        """
        Advanced steganography logic would go here.
        For now, we just append it as Raw data.
        """
        pass
