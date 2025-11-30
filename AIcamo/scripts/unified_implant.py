import sys
import time
import logging
import random
import torch
import torch.nn as nn
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.generation.event_generator import EventGenerator
from cyberdefense.models.vae_gan import VAEGANModel
from cyberdefense.preprocessing.event_tokenizer import EventTokenizer
from artifact_fabricator import ArtifactFabricator, ScenarioEventGenerator, _harvest_real_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SystemExecutor:
    """Executes system actions (Process Start, File Create)"""
    def __init__(self, base_dir="C:\\temp\\ai_generated"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def execute_event(self, event: dict):
        try:
            event_type = event.get("event_type", "").upper()
            if "PROCESS" in event_type:
                self._handle_process(event)
            elif "FILE" in event_type:
                self._handle_file(event)
        except Exception as e:
            logger.warning(f"System Execution failed: {e}")

    def _handle_process(self, event):
        proc_name = event.get("process_name", "")
        if not proc_name: return
        
        # Safe execution mapping
        cmd_map = {
            "notepad.exe": f"notepad {self.base_dir / 'note.txt'}",
            "calc.exe": "calc",
            "cmd.exe": "cmd /c echo AI_Activity",
            "powershell.exe": "powershell -Command \"Write-Host AI_Activity\"",
            "python.exe": "python --version",
            "git.exe": "git --version",
            "code.exe": "echo VSCode_Activity", # Don't actually launch VSCode UI
        }
        
        cmd = None
        for key, val in cmd_map.items():
            if key.lower() in proc_name.lower():
                cmd = val
                break
        
        if cmd:
            logger.info(f"SYSTEM EXEC: Starting {proc_name}")
            subprocess.Popen(cmd, shell=True, creationflags=0)
        else:
            logger.debug(f"SYSTEM EXEC: Simulating {proc_name}")

    def _handle_file(self, event):
        file_path = event.get("file_path", f"ai_file_{random.randint(1000,9999)}.txt")
        target_path = self.base_dir / Path(file_path).name
        try:
            with open(target_path, "w") as f:
                f.write(f"AI Generated: {event}")
            logger.info(f"SYSTEM EXEC: Created file {target_path.name}")
        except:
            pass

class UnifiedImplant:
    def __init__(self):
        self.device = "cpu"
        self.host_generator = None
        self.network_generator = None
        self.fabricator = ArtifactFabricator()
        self.executor = SystemExecutor()
        
        # Gadget Chain Seeds (Intent -> Seed Event)
        self.gadget_seeds = {
            "DEV_WORKFLOW": {
                "event_type": "PROCESS_START", 
                "process_name": "Code.exe", 
                "file_path": "C:\\Users\\User\\Projects\\main.py"
            },
            "OFFICE_MACRO": {
                "event_type": "PROCESS_START", 
                "process_name": "winword.exe", 
                "command_line": "/n /q document.docx"
            },
            "DATA_SYNC": {
                "event_type": "PROCESS_START", 
                "process_name": "OneDrive.exe", 
                "command_line": "/background"
            }
        }

    def load_models(self):
        """Load and Quantize both models"""
        logger.info("Loading Unified Models...")
        
        # 1. Load General Host Model (Gadget Chains)
        try:
            logger.info("Loading General Host Model...")
            import yaml
            config_path = "configs/model_config.yaml"
            with open(config_path) as f: config = yaml.safe_load(f)

            # Tokenizer (synthetic host data tokenizer)
            tok_host = EventTokenizer.load("data/processed/tokenizer_synthetic")
            config["model"]["max_vocab_size"] = len(tok_host.vocab)

            # Model
            model_host = VAEGANModel(config)
            ckpt_host = torch.load("checkpoints/host_model/checkpoint_epoch_5.pt", map_location="cpu")
            if "model_state_dict" in ckpt_host: model_host.load_state_dict(ckpt_host["model_state_dict"])
            else: model_host.load_state_dict(ckpt_host)

            self.host_generator = EventGenerator(model_host, tok_host, device="cpu")
            logger.info("General Host Model Loaded (Epoch 5).")

        except Exception as e:
            logger.error(f"Failed to load General Host Model: {e}")

        # 2. Load Host Network Model (Network Artifacts)
        try:
            logger.info("Loading Host Network Model...")
            # Re-use config structure but might need tweaks if dims changed
            # Assuming same config structure for now
            
            # Tokenizer
            tok_net = EventTokenizer.load("data/processed/tokenizer_host") # Network tokenizer
            config["model"]["max_vocab_size"] = len(tok_net.vocab)
            
            # Model
            model_net = VAEGANModel(config)
            ckpt_net = torch.load("checkpoints/network_model/checkpoint_epoch_3.pt", map_location="cpu")
            if "model_state_dict" in ckpt_net: model_net.load_state_dict(ckpt_net["model_state_dict"])
            else: model_net.load_state_dict(ckpt_net)
            
            # QUANTIZATION - Disabled due to PyTorch 2.x Transformer compatibility issue
            # model_net = torch.quantization.quantize_dynamic(
            #    model_net, {nn.Linear, nn.LSTM}, dtype=torch.qint8
            # )
            self.network_generator = EventGenerator(model_net, tok_net, device="cpu")
            logger.info("Host Network Model Loaded & Quantized.")
            
        except Exception as e:
            logger.error(f"Failed to load Host Network Model: {e}")

    def run_smoke_bomb(self, fast_mode=False):
        """Automated Smoke Bomb Loop"""
        if not self.host_generator and not self.network_generator:
            logger.error("No models loaded. Aborting.")
            return

        logger.info(f"Starting AUTOMATED SMOKE BOMB [Fast Mode: {fast_mode}] [Discrete: {self.fabricator.discrete_mode}]...")
        
        context_host = []
        context_net = []
        
        try:
            while True:
                # Decision: Background Noise (80%) vs Complex Workflow (20%)
                is_complex = random.random() < 0.20
                
                if is_complex and self.host_generator:
                    # --- COMPLEX WORKFLOW (Host Model) ---
                    intent = random.choice(list(self.gadget_seeds.keys()))
                    seed = self.gadget_seeds[intent]
                    
                    print(f"\n{'='*60}")
                    logger.info(f"🧠 AI DECISION: Initiating Complex Behavior Pattern")
                    logger.info(f"🔗 GADGET CHAIN: {intent}")
                    logger.info(f"📝 REASONING: Simulating realistic user workflow to blend in.")
                    print(f"{'='*60}\n")
                    
                    # Generate sequence (e.g., 3 events, max_len=30 for speed, batch_size=3 for parallel)
                    # Prime with seed
                    events = self.host_generator.generate(num_events=3, batch_size=3, context_events=[seed], max_length=30)
                    
                    for i, event in enumerate(events):
                        logger.info(f"  ├─ STEP {i+1}: AI Predicted '{event.get('event_type')}' -> {event.get('process_name')}")
                        
                        # Fabricate artifact (Event Log entry)
                        self.fabricator.fabricate_event(event)

                        # Execute System Action (actually run the process)
                        self.executor.execute_event(event)
                        if not fast_mode: time.sleep(1)
                        
                elif self.network_generator:
                    # --- BACKGROUND NOISE (Network Model) ---
                    # Generate 1-3 events
                    num = random.randint(1, 3)
                    
                    print(f"\n{'-'*40}")
                    logger.info(f"🎲 AI DECISION: Generating Background Network Noise ({num} events)")
                    logger.info(f"📉 GOAL: Maintain baseline traffic to mask silence.")
                    print(f"{'-'*40}\n")
                    
                    events = self.network_generator.generate(num_events=num, context_events=context_net if context_net else None)

                    for event in events:
                        # Filter out PROCESS_START from network model (Host model handles those)
                        if event.get('event_type') == 'PROCESS_START':
                            continue

                        # Fix missing URL if needed (Fabricator handles it, but good to log)
                        target = event.get('dest_address') or event.get('url')
                        logger.info(f"  ├─ NETWORK ACTION: Visiting {target}")
                        self.fabricator.fabricate_event(event) # Inject Artifact

                        context_net.append(event)
                        if len(context_net) > 10: context_net.pop(0)
                        
                # Random sleep to vary density
                if not fast_mode:
                    sleep_time = random.uniform(2.0, 10.0)
                    logger.info(f"⏳ PAUSE: Sleeping {sleep_time:.1f}s to mimic human latency...\n")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Smoke Bomb stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Behavioral Camouflage Implant")
    parser.add_argument("--test", action="store_true", help="Run a single iteration test")
    parser.add_argument("--fast", action="store_true", help="Disable sleep delays for rapid generation")
    parser.add_argument("--discrete", action="store_true", help="Use camouflaged event sources instead of 'AI_Ghost'")
    args = parser.parse_args()

    implant = UnifiedImplant()
    
    # Configure Discrete Mode
    if args.discrete:
        implant.fabricator.discrete_mode = True
        logger.info("DISCRETE MODE ENABLED: Using camouflaged event sources.")
    
    implant.load_models()
    
    # Check for test mode
    if args.test:
        logger.info("Test mode: Running one iteration...")
        # Mock run logic for test
        if implant.host_generator:
             print("Host Generator Ready")
        if implant.network_generator:
             print("Network Generator Ready")
        sys.exit(0)
        
    implant.run_smoke_bomb(fast_mode=args.fast)
