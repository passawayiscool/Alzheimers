import sys
import time
import logging
import subprocess
import random
import sqlite3
import datetime
import os
import ctypes
from pathlib import Path
import yaml
import torch
import json
import base64
try:
    from scapy.all import IP, TCP, UDP, DNS, DNSQR, send
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.getLogger(__name__).warning("Scapy not available, network generation will be limited")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdefense.generation.event_generator import EventGenerator
from cyberdefense.models.vae_gan import VAEGANModel
from cyberdefense.preprocessing.event_tokenizer import EventTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ArtifactFabricator:
    """
    Takes events predicted by the AI model and FABRICATES them on the system
    by injecting directly into forensic stores (Event Logs, SQLite, MFT).
    Does NOT run the actual applications.
    """
    def __init__(self, base_dir="C:\\temp\\ai_artifacts", discrete_mode=False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.chrome_history_path = self._find_chrome_history()
        self.edge_history_path = self._find_edge_history()
        self.discrete_mode = discrete_mode
        self.real_sources = ["VSS", "ESENT", "Software Protection Platform Service", "SecurityCenter"]

    def _get_source(self):
        if self.discrete_mode:
            return random.choice(self.real_sources)
        return "AI_Ghost"

    def _find_chrome_history(self):
        """Locate Chrome History DB"""
        # Try standard location
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            path = Path(local_app_data) / "Google" / "Chrome" / "User Data" / "Default" / "History"
            if path.exists():
                return path
        
        # Try project specific profile
        project_profile = Path("chrome_profile") / "Default" / "History"
        if project_profile.exists():
            return project_profile.absolute()
            
        return None

    def _find_edge_history(self):
        """Locate Edge History DB"""
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            path = Path(local_app_data) / "Microsoft" / "Edge" / "User Data" / "Default" / "History"
            if path.exists():
                return path
        return None

    def fabricate_event(self, event: dict):
        """Map abstract event to concrete forensic artifact"""
        try:
            event_type = event.get("event_type", "").upper()
            
            if "PROCESS" in event_type:
                self._fabricate_process_artifact(event)
            elif "FILE" in event_type:
                self._fabricate_file_artifact(event)
            elif "NETWORK" in event_type:
                self._fabricate_network_artifact(event)
            elif "BROWSER" in event_type:
                self._fabricate_browser_history_artifact(event)
            else:
                logger.debug(f"Skipping unknown event type: {event_type}")
                
        except Exception as e:
            logger.warning(f"Failed to fabricate artifact for {event}: {e}")

    def _fabricate_process_artifact(self, event):
        """Write to Windows Event Log (Application)"""
        proc_name = event.get("process_name", "")
        if not proc_name:
            if event.get("file_path"):
                proc_name = Path(event["file_path"]).name
            elif event.get("command_line"):
                # Try to extract from command line (e.g. "C:\Path\to\exe" args)
                import shlex
                try:
                    # Simple heuristic: first token ending in .exe
                    parts = shlex.split(event["command_line"])
                    for part in parts:
                        if part.lower().endswith(".exe"):
                            proc_name = Path(part).name
                            break
                    if not proc_name and parts:
                         proc_name = Path(parts[0]).name # Fallback to first token
                except:
                    pass
        
        if not proc_name:
            proc_name = "unknown.exe"
        pid = event.get("pid", random.randint(1000, 9999))
        
        # Event ID 4688 is "A new process has been created" (Security Log)
        # Writing to Security Log is hard/restricted.
        # We will write to Application Log as a "Trace" or "Audit" event from our source.
        # Or if we have Admin, we can try to write to a custom log that looks like Sysmon.
        
        # For this demo, we use the Application Log with a source that mimics a system component
        # or we just use our "AI_Ghost" source we created.
        
        # Format message to look like real Security Event 4688
        real_cmd = event.get("command_line", f"{proc_name}")
        parent_pid = event.get("ppid", random.randint(1000, 9999))
        
        msg = f"""A new process has been created.

Creator Subject:
    Security ID:        S-1-5-21-{random.randint(1000000000, 9999999999)}-{random.randint(1000000000, 9999999999)}-{random.randint(1000000000, 9999999999)}-1001
    Account Name:       {os.getlogin()}
    Account Domain:     {os.environ.get('USERDOMAIN', 'WORKGROUP')}
    Logon ID:           0x{random.randint(100000, 999999):x}

Target Process:
    Process ID:         0x{pid:x}
    Process Name:       {event.get('file_path', proc_name)}

Process Information:
    New Process ID:     0x{pid:x}
    New Process Name:   {event.get('file_path', proc_name)}
    Token Elevation Type: %%1937
    Mandatory Label:    S-1-16-12288
    Creator Process ID: 0x{parent_pid:x}
    Process Command Line: {real_cmd}"""
        
        # Select Source based on mode
        source = self._get_source()
        
        if is_admin():
            # Ensure Source exists (Try-Catch in PowerShell)
            # Only try to create if it's our custom one. System ones should exist.
            if source == "AI_Ghost":
                create_source_cmd = f'try {{ if (![System.Diagnostics.EventLog]::SourceExists("{source}")) {{ New-EventLog -LogName Application -Source "{source}" -ErrorAction SilentlyContinue }} }} catch {{}}'
                subprocess.run(["powershell", "-Command", create_source_cmd], creationflags=0, check=False)
            
            # Write Event using Base64 to avoid quoting issues
            msg_b64 = base64.b64encode(msg.encode('utf-8')).decode('ascii')
            
            # PowerShell command to decode and write
            cmd = f'$msg = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String("{msg_b64}")); Write-EventLog -LogName Application -Source "{source}" -EventID 4688 -EntryType Information -Category 0 -Message $msg'
            
            subprocess.run(["powershell", "-Command", cmd], creationflags=0, check=False)
            logger.info(f"FABRICATED: Process Event Log for {proc_name} [Source: {source}]")
        else:
            logger.info(f"FABRICATED: Process Event Log for {proc_name} (Skipped - Non-Admin)")
            # print(f"DEBUG: Skipped Event Log for {proc_name} (Not Admin)", flush=True)

    def _fabricate_file_artifact(self, event):
        """Create a dummy file and timestomp it"""
        file_path = event.get("file_path", "")
        if not file_path: 
            file_path = f"doc_{random.randint(1000,9999)}.docx"
            
        target_path = self.base_dir / Path(file_path).name
        
        # 1. Create File (MFT Entry)
        with open(target_path, "w") as f:
            f.write(f"Generated content for {file_path}")
            
        # 2. Timestomp (Set creation time to match event timestamp if provided, else now)
        # For now, just leave as is (current time is fine for "live" simulation)
        logger.info(f"FABRICATED: File Artifact {target_path}")
        
        # 3. Create LNK in Recent (Simulate User Access)
        self._fabricate_lnk_artifact(target_path)

    def _fabricate_lnk_artifact(self, target_path: Path):
        """Create a LNK shortcut in Recent Items to simulate file access"""
        try:
            recent_dir = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Recent"
            lnk_name = f"{target_path.name}.lnk"
            lnk_path = recent_dir / lnk_name
            
            # Use PowerShell WScript.Shell to create LNK
            # $WshShell = New-Object -comObject WScript.Shell
            # $Shortcut = $WshShell.CreateShortcut("C:\...\Recent\file.lnk")
            # $Shortcut.TargetPath = "C:\...\file.txt"
            # $Shortcut.Save()
            
            ps_script = f'''
            $WshShell = New-Object -comObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut("{lnk_path}")
            $Shortcut.TargetPath = "{target_path}"
            $Shortcut.Save()
            '''
            
            subprocess.run(["powershell", "-Command", ps_script], check=False, creationflags=0)
            logger.info(f"FABRICATED: LNK Artifact {lnk_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create LNK: {e}")

    def _fabricate_network_artifact(self, event):
        """Inject URL into Chrome History"""
        if not self.chrome_history_path:
            logger.warning("Chrome History DB not found. Skipping network artifact.")
            return

        dest = event.get("destination", "google.com")
        url = f"https://{dest}/"
        title = f"{dest} - Visited by AI"
        
        # Chrome time: microseconds since 1601-01-01
        # Current time
        delta = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc)
        chrome_time = int(delta.total_seconds() * 1000000)
        
        try:
            conn = sqlite3.connect(str(self.chrome_history_path), timeout=10) # Increased timeout
            cursor = conn.cursor()
            
            # Check if URL exists
            cursor.execute("SELECT id, visit_count FROM urls WHERE url = ?", (url,))
            row = cursor.fetchone()
            
            if row:
                # Update
                new_count = row[1] + 1
                cursor.execute("UPDATE urls SET visit_count = ?, last_visit_time = ? WHERE id = ?", (new_count, chrome_time, row[0]))
            else:
                # Insert
                cursor.execute(
                    "INSERT INTO urls (url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?)",
                    (url, title, 1, chrome_time)
                )
                
            conn.commit()
            conn.close()
            logger.info(f"FABRICATED: Chrome History Entry for {dest}")
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                logger.warning(f"Chrome is running (DB Locked). Skipped history injection for {dest}")
            else:
                logger.error(f"Chrome Injection Failed: {e}")
        except Exception as e:
            logger.error(f"Chrome Injection Failed: {e}")

        # HOST-BASED NETWORK ARTIFACTS (Anti-Forensics)
        # Goal: Plant evidence of network activity on the HOST without sending packets.
        
        # 1. Populate DNS Cache (Real Artifact)
        # Running a resolution leaves a trace in 'ipconfig /displaydns'
        try:
            # Use PowerShell to resolve (stealthy, looks like system activity)
            cmd = f"Resolve-DnsName -Name {dest} -Type A -ErrorAction SilentlyContinue"
            subprocess.run(["powershell", "-Command", cmd], creationflags=0, check=False)
            logger.info(f"FABRICATED: DNS Cache Entry for {dest}")
        except Exception as e:
            logger.warning(f"DNS Cache population failed: {e}")

        # 2. Fabricate Network Connection Log (Event ID 5156 - WFP Connection)
        # This simulates the Windows Filtering Platform logging a connection
        if is_admin():
            try:
                # Use predicted values if available, else random
                src_port = random.randint(49152, 65535)
                dst_port = event.get("dest_port", 443 if "https" in url else 80)
                dst_ip = event.get("dest_address", "8.8.8.8")
                target_event_id = event.get("event_id", "5156")
                
                # Event ID 5156: "The Windows Filtering Platform has permitted a connection."
                # We write this to the Security Log (if possible) or Application Log as camouflage
                
                msg = f"""The Windows Filtering Platform has permitted a connection.

Application Information:
    Process ID:     {event.get('pid', random.randint(1000,9999))}
    Application Name: \\device\\harddiskvolume1\\windows\\system32\\{event.get('process_name', 'chrome.exe')}

Network Information:
    Direction:      {event.get('direction', 'Outbound')}
    Source Address: 192.168.1.{random.randint(2,254)}
    Source Port:    {src_port}
    Destination Address: {dst_ip}
    Destination Port:    {dst_port}
    Protocol:       {event.get('protocol', '6')}"""
                
                # Select Source based on mode
                source = self._get_source()

                # Ensure Source exists (Try-Catch in PowerShell)
                if source == "AI_Ghost":
                    create_source_cmd = f'try {{ if (![System.Diagnostics.EventLog]::SourceExists("{source}")) {{ New-EventLog -LogName Application -Source "{source}" -ErrorAction SilentlyContinue }} }} catch {{}}'
                    subprocess.run(["powershell", "-Command", create_source_cmd], creationflags=0, check=False)

                # Base64 encode for PowerShell injection
                msg_b64 = base64.b64encode(msg.encode('utf-8')).decode('ascii')
                cmd = f'$msg = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String("{msg_b64}")); Write-EventLog -LogName Application -Source "{source}" -EventID 5156 -EntryType Information -Category 0 -Message $msg'

                subprocess.run(["powershell", "-Command", cmd], creationflags=0, check=False)
                logger.info(f"FABRICATED: WFP Connection Log (Event 5156) for {dest} [Source: {source}]")
                
            except Exception as e:
                logger.warning(f"Event Log injection failed: {e}")
        else:
            logger.info("Skipping Event Log injection (Not Admin)")

    def _fabricate_browser_history_artifact(self, event):
        """Inject Browser History based on event"""
        # Select Target Browser (Randomly if both exist, else whatever is available)
        targets = []
        if self.chrome_history_path: targets.append(("Chrome", self.chrome_history_path))
        if self.edge_history_path: targets.append(("Edge", self.edge_history_path))
        
        if not targets:
            logger.warning("No Browser History DBs found (Chrome/Edge). Skipping artifact.")
            return

        browser_name, db_path = random.choice(targets)

        # 1. Get URL (Try 'url' field, then construct from 'dest_address', then fallback)
        url = event.get("url")
        title = event.get("title")
        
        if not url:
            # Fallback: Construct from dest_address (IP or Domain)
            dest = event.get("dest_address") or event.get("destination")
            if dest:
                url = f"https://{dest}/"
                if not title:
                    title = f"{dest} - Visited"
            else:
                # Last resort
                url = "https://google.com/"
                title = "Google"
                
        if not title:
            title = "Visited Page"

        # Chrome/Edge time: microseconds since 1601-01-01
        delta = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc)
        chrome_time = int(delta.total_seconds() * 1000000)
        
        try:
            conn = sqlite3.connect(str(db_path), timeout=10)
            cursor = conn.cursor()
            
            # Check if URL exists
            cursor.execute("SELECT id, visit_count FROM urls WHERE url = ?", (url,))
            row = cursor.fetchone()
            
            url_id = 0
            if row:
                # Update
                url_id = row[0]
                new_count = row[1] + 1
                cursor.execute("UPDATE urls SET visit_count = ?, last_visit_time = ? WHERE id = ?", (new_count, chrome_time, url_id))
            else:
                # Insert
                cursor.execute(
                    "INSERT INTO urls (url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?)",
                    (url, title, 1, chrome_time)
                )
                url_id = cursor.lastrowid
                
            # --- CRITICAL FIX: Insert into 'visits' table as well ---
            # Without this, the history entry is incomplete and may corrupt the profile or not show up.
            # transition = 805306368 (CHAIN_END | LINK) or 0 (TYPED)
            # We'll use LINK (0) + CHAIN_START (0x10000000) -> 268435456? No, let's just use 0 (LINK) or 1 (TYPED)
            # Standard transition for clicking a link is 0 (LINK).
            # But 'visits' table 'transition' column is masked.
            # Let's use a standard value observed in real DBs: 805306368 (Link) or 805306369 (Typed)
            
            cursor.execute(
                "INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration) VALUES (?, ?, ?, ?, ?, ?)",
                (url_id, chrome_time, 0, 805306368, 0, 0)
            )
            
            conn.commit()
            conn.close()
            logger.info(f"FABRICATED: {browser_name} History Entry for {url} (ID: {url_id})")
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                logger.warning(f"{browser_name} is running (DB Locked). Skipped history injection for {url}")
            else:
                logger.error(f"{browser_name} Injection Failed: {e}")
        except Exception as e:
            logger.error(f"{browser_name} Injection Failed: {e}")

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

class ScenarioEventGenerator:
    """
    Generates realistic, state-aware event sequences based on user personas.
    Uses REAL system sources harvested from the machine for camouflage.
    """
    def __init__(self, real_sources=None):
        self.real_sources = real_sources if real_sources else ["Application", "System", "Security-Auditing"]
        # Filter out our own fake sources if they got harvested
        self.real_sources = [s for s in self.real_sources if "AI_Ghost" not in s]
        
        self.scenarios = [
            self._scenario_developer_work,
            self._scenario_web_browsing,
            self._scenario_system_maintenance
        ]
        self.current_scenario = None
        self.scenario_step = 0
        
    def _get_camouflaged_source(self):
        """Pick a real system source to blend in"""
        if self.real_sources:
            return random.choice(self.real_sources)
        return "Microsoft-Windows-Security-Auditing" # Fallback

    def generate(self, num_events=1, context_events=None):
        events = []
        for _ in range(num_events):
            if not self.current_scenario or self.scenario_step >= len(self.current_scenario):
                # Pick a new scenario
                scenario_func = random.choice(self.scenarios)
                self.current_scenario = scenario_func()
                self.scenario_step = 0
                
            # Get next event from current scenario
            event = self.current_scenario[self.scenario_step]
            self.scenario_step += 1
            events.append(event)
            
        return events

    def _scenario_developer_work(self):
        """Sequence: Open VSCode -> Edit File -> Run Python -> Check Network"""
        pid_code = random.randint(1000, 9999)
        pid_py = random.randint(1000, 9999)
        src = self._get_camouflaged_source()
        
        return [
            {"event_type": "PROCESS_START", "process_name": "Code.exe", "pid": pid_code, "source": src},
            {"event_type": "FILE_CREATE", "file_path": f"C:\\Users\\User\\Projects\\src\\main_{random.randint(1,99)}.py"},
            {"event_type": "PROCESS_START", "process_name": "python.exe", "pid": pid_py, "source": src},
            {"event_type": "NETWORK_CONNECT", "destination": "pypi.org"},
            {"event_type": "FILE_CREATE", "file_path": f"C:\\Users\\User\\Projects\\src\\__pycache__\\main.cpython-39.pyc"}
        ]

    def _scenario_web_browsing(self):
        """Sequence: Chrome -> DNS -> HTTPS -> Download"""
        pid_chrome = random.randint(1000, 9999)
        domain = random.choice(["github.com", "stackoverflow.com", "docs.python.org", "medium.com"])
        src = self._get_camouflaged_source()
        
        return [
            {"event_type": "PROCESS_START", "process_name": "chrome.exe", "pid": pid_chrome, "source": src},
            {"event_type": "NETWORK_CONNECT", "destination": domain},
            {"event_type": "FILE_CREATE", "file_path": f"C:\\Users\\User\\Downloads\\{domain.split('.')[0]}_page.html"},
            {"event_type": "NETWORK_CONNECT", "destination": "fonts.googleapis.com"}
        ]

    def _scenario_system_maintenance(self):
        """Sequence: PowerShell -> System Info -> Log Check"""
        pid_ps = random.randint(1000, 9999)
        src = self._get_camouflaged_source()
        
        return [
            {"event_type": "PROCESS_START", "process_name": "powershell.exe", "pid": pid_ps, "source": src},
            {"event_type": "FILE_CREATE", "file_path": f"C:\\Users\\User\\AppData\\Local\\Temp\\ps_transcript_{random.randint(1000,9999)}.txt"},
            {"event_type": "PROCESS_START", "process_name": "conhost.exe", "pid": pid_ps + 1, "source": src},
            {"event_type": "NETWORK_CONNECT", "destination": "update.microsoft.com"}
        ]

def _harvest_real_sources():
    """Harvest actual Event Sources from the local machine for camouflage"""
    try:
        cmd = "Get-EventLog -LogName Application -Newest 1000 | Select-Object -ExpandProperty Source -Unique"
        result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, creationflags=0)
        if result.returncode == 0:
            sources = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            logger.info(f"Harvested {len(sources)} real event sources for camouflage.")
            return sources
    except Exception as e:
        logger.warning(f"Failed to harvest sources: {e}")
    return None

def _harvest_local_context():
    """Harvest recent local events to prime the model"""
    context_events = []
    try:
        # PRIORITY 1: Look for specific "Interesting" processes regardless of start time
        # This fixes the issue where long-running browsers (Chrome/Edge) are missed
        interesting_names = [
            "chrome", "msedge", "firefox", "brave", # Browsers
            "code", "devenv", "idea", "pycharm",    # Dev
            "winword", "excel", "powerpnt", "outlook", # Office
            "powershell", "cmd", "wt"               # Terminals
        ]
        
        # Construct a filter string: "ProcessName -eq 'chrome' -or ProcessName -eq 'msedge' ..."
        filter_str = " -or ".join([f"($_.ProcessName -eq '{name}')" for name in interesting_names])
        
        # Command 1: Get Priority Processes
        cmd_priority = f"Get-Process -ErrorAction SilentlyContinue | Where-Object {{ {filter_str} }} | Select-Object ProcessName, StartTime, Path | ConvertTo-Json"
        
        # Command 2: Get Recent Processes (Fallback)
        cmd_recent = "Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.StartTime } | Sort-Object StartTime -Descending | Select-Object -First 15 | Select-Object ProcessName, StartTime, Path | ConvertTo-Json"
        
        procs = []
        
        # Run Priority
        res_p = subprocess.run(["powershell", "-Command", cmd_priority], capture_output=True, text=True, creationflags=0)
        if res_p.stdout.strip():
            try:
                p_data = json.loads(res_p.stdout)
                if not isinstance(p_data, list): p_data = [p_data]
                procs.extend(p_data)
            except: pass
            
        # Run Recent
        res_r = subprocess.run(["powershell", "-Command", cmd_recent], capture_output=True, text=True, creationflags=0)
        if res_r.stdout.strip():
            try:
                r_data = json.loads(res_r.stdout)
                if not isinstance(r_data, list): r_data = [r_data]
                procs.extend(r_data)
            except: pass
            
        # Deduplicate by Path (or Name if Path missing)
        seen = set()
        unique_procs = []
        for p in procs:
            key = p.get("Path") or p.get("ProcessName")
            if key and key not in seen:
                seen.add(key)
                unique_procs.append(p)
        
        # Common noisy Windows services to ignore
        NOISY_PROCS = [
            "svchost", "conhost", "RuntimeBroker", "SearchUI", "explorer",
            "MsMpEng", "Memory Compression", "Registry", "System", "Idle",
            "taskhostw", "dllhost", "sihost", "smss", "csrss", "wininit", "services", "lsass",
            "Antigravity", "language_server" # Ignore self
        ]
        
        for proc in unique_procs:
            name = proc.get("ProcessName", "Unknown")
            path = proc.get("Path", "")
            
            # FILTER: Ignore system background processes
            if any(noisy.lower() in name.lower() for noisy in NOISY_PROCS):
                continue
                
            # Heuristic mapping
            evt = {
                "event_type": "PROCESS_START", 
                "user": os.getlogin(),
                "host": os.environ.get("COMPUTERNAME", "Unknown"),
                "process_name": f"{name}.exe",
                "file_path": path if path else f"C:\\Program Files\\{name}\\{name}.exe",
                "command_line": f'"{path}"' if path else f"{name}.exe",
                "timestamp": proc.get("StartTime", "")
            }
            context_events.append(evt)
            
            if len(context_events) >= 15: # Increased context size
                break
            
        logger.info(f"Harvested {len(context_events)} running processes for context priming.")
        return context_events
        
    except Exception as e:
        logger.warning(f"Failed to harvest local context: {e}")
        
    return []

def main():
    if not is_admin():
        logger.warning("WARNING: Not running as Admin. Event Log injection may fail.")
        
    logger.info("Initializing Artifact Fabricator (Generative Anti-Forensics)...")
    
    generator = None
    
    # 1. Load Model (Try Real, Fallback to Mock)
    real_sources = _harvest_real_sources()
    
    try:
        config_path = "configs/model_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        tokenizer = EventTokenizer.load("data/processed/tokenizer_lanl")
        config["model"]["max_vocab_size"] = len(tokenizer.vocab)
        
        model = VAEGANModel(config)
        # Use dummy model for verification
        checkpoint_path = Path("checkpoints/dummy_model.pt")
             
        if checkpoint_path.exists() and False: # FORCE REAL MODEL LOADING
            # OPTIMIZATION: If using dummy model, use Mock Generator to save CPU
            logger.info("Dummy model detected. Using Scenario Generator for efficiency.")
            generator = ScenarioEventGenerator(real_sources=real_sources)
        else:
            # Try to load real checkpoint
            real_checkpoint = Path("checkpoints/checkpoint_epoch_1.pt") # LANL Model
            if real_checkpoint.exists():
                checkpoint = torch.load(real_checkpoint, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                
                generator = EventGenerator(model=model, tokenizer=tokenizer, device="cpu")
                logger.info("Loaded VAE-GAN Model (LANL Trained).")
            else:
                logger.warning("No valid checkpoint found. Using Scenario Generator.")
                generator = ScenarioEventGenerator(real_sources=real_sources)
            
    except Exception as e:
        logger.error(f"Model load failed ({e}). Using Scenario Generator.")
        generator = ScenarioEventGenerator(real_sources=real_sources)
        
    fabricator = ArtifactFabricator()
    logger.info("Starting Ghost Writer Loop...")
    
    # 2. Loop
    # Initialize context with REAL local events
    context = _harvest_local_context()
    if not context:
        context = []
        
    logger.info("Generating exactly 20 AI events...")
    
    for i in range(20):
        # Predict next events
        try:
            # Generate 1 event
            events = generator.generate(num_events=1, context_events=context if context else None)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            events = []

        if events:
            event = events[0]
            
            # Skip invalid events
            if "event_type" not in event:
                continue
            
            # --- CONTEXT AWARE SUBSTITUTION ---
            # The model predicts generic behavior, we map it to the local user context
            current_user = os.getlogin()
            import platform
            current_host = platform.node()
            
            if "user" in event:
                event["user"] = current_user
            if "host" in event:
                event["host"] = current_host
                
            # Deep Substitution: Fix paths like C:\Users\jdoe\... -> C:\Users\Xander\...
            import re
            user_path_pattern = re.compile(r"(C:\\Users\\[^\\]+)", re.IGNORECASE)
            # Escape backslashes for re.sub (replace \ with \\)
            real_user_path = f"C:\\Users\\{current_user}".replace("\\", "\\\\")
            
            for field in ["file_path", "command_line"]:
                if event.get(field):
                    # Replace C:\Users\ANYONE with C:\Users\CURRENT_USER
                    event[field] = user_path_pattern.sub(real_user_path, event[field])
                    
                    # Also replace hostnames if present in UNC paths
                    if "\\\\" in event[field]:
                         # Simple heuristic: replace any word after \\ with current host
                         event[field] = re.sub(r"(\\\\)[^\\]+(\\)", f"\\\\{current_host}\\\\", event[field])
            # ----------------------------------
            
            logger.info(f"AI PREDICTION [{i+1}/20]: {event}")
            print(f"AI PREDICTION: {event}", flush=True)
            
            # FABRICATE IT
            fabricator.fabricate_event(event)
            
            # Update context
            context.append(event)
            if len(context) > 10: context.pop(0)
            
        # No sleep, no timeout, just pure speed
            


if __name__ == "__main__":
    main()
