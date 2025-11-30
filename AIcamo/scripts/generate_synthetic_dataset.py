import pandas as pd
import random
import time
from pathlib import Path
import logging
from typing import List, Dict, Optional
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTS & CATALOGS ---

USERS = ["Xander", "SYSTEM", "NETWORK SERVICE", "LOCAL SERVICE", "Admin", "Guest", "jdoe", "dev_user"]
HOSTS = ["DESKTOP-XANDER", "WORKSTATION-01", "SERVER-AD", "LAPTOP-DEV", "BUILD-SERVER-03"]
DRIVES = ["C:", "D:", "E:"]
DOMAINS = ["google.com", "github.com", "microsoft.com", "api.internal", "cdn.content.net", "stackoverflow.com", "jira.corp.local"]

# Common flags for variation
ELECTRON_FLAGS = [
    "--no-sandbox", "--disable-gpu", "--disable-software-rasterizer", "--disable-dev-shm-usage",
    "--enable-logging", "--v=1", "--force-device-scale-factor=1", "--disable-features=OutOfBlinkCors",
    "--enable-features=NetworkService,NetworkServiceInProcess", "--type=renderer", "--type=gpu-process",
    "--type=utility", "--type=crashpad-handler"
]

def gen_ver():
    return f"{random.randint(1, 100)}.{random.randint(0, 9)}.{random.randint(100, 9999)}.{random.randint(0, 99)}"

class AppDef:
    def __init__(self, name: str, vendor: str, product: str, exe: str, category: str, 
                 install_scope: str = "system", 
                 versions: Optional[List[str]] = None,
                 cli_template: Optional[List[str]] = None,
                 arg_style: str = "standard",
                 weight: int = 1): # standard, electron, java, gnu
        self.name = name
        self.vendor = vendor
        self.product = product
        self.exe = exe
        self.category = category
        self.install_scope = install_scope
        self.versions = versions or [gen_ver() for _ in range(3)]
        self.cli_template = cli_template or []
        self.arg_style = arg_style
        self.weight = weight

APP_CATALOG = [
    # BROWSERS (High Volume)
    AppDef("Google Chrome", "Google", "Chrome", "chrome.exe", "browser", "system", arg_style="electron",
           cli_template=["{electron_flags} {url}", "{electron_flags} --user-data-dir={temp}"], weight=15),
    AppDef("Microsoft Edge", "Microsoft", "Edge", "msedge.exe", "browser", "system", arg_style="electron",
           cli_template=["{electron_flags} {url}"], weight=15),
    
    # ELECTRON APPS (Medium Volume)
    AppDef("VS Code", "Microsoft", "VS Code", "Code.exe", "dev", "user", arg_style="electron",
           cli_template=["{electron_flags} {file}", "{electron_flags} .", "{electron_flags} --status"], weight=5),
    AppDef("Slack", "Slack", "Slack", "slack.exe", "comms", "user", arg_style="electron",
           cli_template=["{electron_flags}"], weight=5),
    AppDef("Spotify", "Spotify", "Spotify", "Spotify.exe", "media", "user", arg_style="electron",
           cli_template=["{electron_flags} --prefetch={random_int}"], weight=3),
    AppDef("Discord", "Discord", "Discord", "Discord.exe", "comms", "user", arg_style="electron",
           cli_template=["{electron_flags} --url={url}"], weight=3),
           
    # CLOUD / SYNC
    AppDef("OneDrive", "Microsoft", "OneDrive", "FileCoAuth.exe", "utils", "user",
           versions=["25.209.1026.0002", "25.210.1103.0001"],
           cli_template=["-Embedding", "-Embedding {random_guid}"], weight=8),
           
    # DEV TOOLS (Low Volume)
    AppDef("Python", "Python", "Python39", "python.exe", "dev", "system",
           cli_template=[
               "{script}", 
               "{script} --verbose",
               "{script} --config config.json",
               "-m pip install {package}", 
               "-m pip install --upgrade {package}",
               "-c 'print(1)'", 
               "-c 'import sys; print(sys.version)'",
               "--version",
               "-m http.server {random_int}"
           ], weight=2),
    AppDef("Node.js", "NodeJS", "Node", "node.exe", "dev", "system",
           cli_template=["{script}", "server.js", "--version", "-e 'console.log(1)'"], weight=2),
    AppDef("Git", "Git", "Git", "git.exe", "dev", "system",
           cli_template=["commit -m '{message}'", "clone {url}", "status", "push origin main", "checkout -b feature/{random_word}"], weight=2),
    
    # SYSTEM (Very High Volume - Background Noise)
    AppDef("Svchost", "Microsoft", "Windows", "svchost.exe", "system", "system",
           cli_template=["-k {service_group} -s {service_name}", "-k {service_group}"], weight=20),
    AppDef("Conhost", "Microsoft", "Windows", "conhost.exe", "system", "system",
           cli_template=["0x4", "{random_hex}"], weight=15),
    AppDef("RuntimeBroker", "Microsoft", "Windows", "RuntimeBroker.exe", "system", "system",
           cli_template=["-Embedding"], weight=10),
]

# --- GENERATORS ---

def generate_file_path(app: AppDef, user: str) -> str:
    """Generates a realistic file path based on app definition and variation rules."""
    drive = "C:"
    
    if app.install_scope == "system":
        if app.category == "system":
            base = r"Windows\System32"
            return rf"{drive}\{base}\{app.exe}"
        else:
            root = "Program Files"
            return rf"{drive}\{root}\{app.vendor}\{app.product}\Application\{app.exe}"
                
    elif app.install_scope == "user":
        # AppData
        if app.vendor == "Microsoft" and app.product == "OneDrive":
             return rf"{drive}\Users\{user}\AppData\Local\Microsoft\OneDrive\{random.choice(app.versions)}\{app.exe}"
        return rf"{drive}\Users\{user}\AppData\Local\Programs\{app.product}\{app.exe}"
    
    return rf"{drive}\Tools\{app.product}\{app.exe}"

def generate_command_line(app: AppDef, file_path: str, user: str) -> str:
    """Generates a context-aware command line string with high variation."""
    cmd = f'"{file_path}"'
    
    if not app.cli_template:
        return cmd
        
    # Pick a random template
    template = random.choice(app.cli_template)
    args = template
    
    # ELECTRON FLAGS MIXER
    if "{electron_flags}" in args:
        # Pick 2-5 random flags
        num_flags = random.randint(2, 5)
        selected_flags = random.sample(ELECTRON_FLAGS, num_flags)
        
        # Add some dynamic flags
        if random.random() < 0.5:
            selected_flags.append(f"--renderer-client-id={random.randint(1, 100)}")
        if random.random() < 0.5:
            selected_flags.append(f"--field-trial-handle={random.randint(1000, 9999)}")
            
        args = args.replace("{electron_flags}", " ".join(selected_flags))

    # GENERIC PLACEHOLDERS
    if "{random_int}" in args:
        args = args.replace("{random_int}", str(random.randint(1, 9999)))
    if "{random_hex}" in args:
        args = args.replace("{random_hex}", hex(random.randint(0, 65535)))
    if "{random_guid}" in args:
        args = args.replace("{random_guid}", str(uuid.uuid4()))
    if "{random_word}" in args:
        args = args.replace("{random_word}", random.choice(["login", "auth", "ui", "backend", "api"]))

    # URL
    if "{url}" in args:
        domain = random.choice(DOMAINS)
        path = f"page_{random.randint(1,100)}"
        args = args.replace("{url}", f"https://{domain}/{path}")
        
    # FILE / DOC
    if "{file}" in args or "{doc}" in args or "{script}" in args:
        ext_map = {"office": ".docx", "dev": ".py", "utils": ".txt", "system": ".ps1"}
        ext = ext_map.get(app.category, ".dat")
        
        # Diverse Vocabulary
        vocab = {
            "office": ["invoice", "budget", "Q3_report", "meeting_notes", "proposal", "resume", "schedule", "client_list"],
            "dev": ["main", "utils", "config", "test", "api", "models", "views", "setup", "deploy"],
            "system": ["backup", "install", "update", "check_disk", "monitor", "cleanup"],
            "utils": ["log", "notes", "todo", "temp", "data"]
        }
        
        category = app.category if app.category in vocab else "utils"
        base_name = random.choice(vocab[category])
        
        # Add some randomness (versioning, dates)
        if random.random() < 0.3:
            base_name += f"_{random.randint(1, 99)}"
        elif random.random() < 0.3:
            base_name += "_v2"
        elif random.random() < 0.3:
            base_name += "_final"
            
        filename = f"{base_name}{ext}"
        
        path_type = random.choice(["abs", "rel", "unc"])
        
        if path_type == "abs":
            fpath = rf"C:\Users\{user}\Documents\{filename}"
        elif path_type == "unc":
            fpath = rf"\\{random.choice(HOSTS)}\Share\{filename}"
        else:
            fpath = rf".\{filename}"
            
        args = args.replace("{file}", fpath).replace("{doc}", fpath).replace("{script}", fpath)
        
    # TEMP
    if "{temp}" in args:
        args = args.replace("{temp}", rf"C:\Users\{user}\AppData\Local\Temp\{uuid.uuid4().hex[:8]}")
        
    # COMMAND
    if "{command}" in args:
        cmds = ["whoami", "ipconfig", "dir", "net user", "echo hello"]
        args = args.replace("{command}", random.choice(cmds))
        
    # SERVICE GROUP
    if "{service_group}" in args:
        groups = ["netsvcs", "LocalService", "NetworkService", "DcomLaunch"]
        args = args.replace("{service_group}", random.choice(groups))
        
    # SERVICE NAME
    if "{service_name}" in args:
        args = args.replace("{service_name}", random.choice(["PrintWorkflowUserSvc", "WlanSvc", "AudioSrv", "TermService"]))
        
    # MESSAGE
    if "{message}" in args:
        msgs = ["fix bug", "update readme", "initial commit", "wip"]
        args = args.replace("{message}", random.choice(msgs))
        
    # PACKAGE
    if "{package}" in args:
        pkgs = ["requests", "pandas", "numpy", "flask", "django"]
        args = args.replace("{package}", random.choice(pkgs))

    cmd += f" {args}"
        
    return cmd

def generate_synthetic_data(output_path, num_events=500000):
    logger.info(f"Generating {num_events} synthetic events with ENHANCED REALISM & WEIGHTS...")
    
    data = []
    start_time = time.time()
    current_time = start_time
    
class GadgetChain:
    def __init__(self, name: str, steps: List[AppDef], description: str, weight: int = 1):
        self.name = name
        self.steps = steps
        self.description = description
        self.weight = weight

# Define Chains
# 1. Office Macro Style: Word -> CMD -> PowerShell
CHAIN_OFFICE_MACRO = GadgetChain(
    "Office Macro",
    [
        AppDef("Word", "Microsoft", "Office", "winword.exe", "office", "system", cli_template=["/n /q {file}"]),
        AppDef("CMD", "Microsoft", "Windows", "cmd.exe", "system", "system", cli_template=["/c {script}"]),
        AppDef("PowerShell", "Microsoft", "Windows", "powershell.exe", "system", "system", cli_template=["-nop -w hidden -c {command}"])
    ],
    "Simulates a macro spawning a shell",
    weight=5
)

# 2. Dev Chain: VSCode -> Git -> Python
CHAIN_DEV_WORKFLOW = GadgetChain(
    "Dev Workflow",
    [
        AppDef("VS Code", "Microsoft", "VS Code", "Code.exe", "dev", "user", cli_template=["{file}"]),
        AppDef("Git", "Git", "Git", "git.exe", "dev", "system", cli_template=["status"]),
        AppDef("Python", "Python", "Python39", "python.exe", "dev", "system", cli_template=["{script}"])
    ],
    "Standard developer cycle",
    weight=5
)

GADGET_CHAINS = [CHAIN_OFFICE_MACRO, CHAIN_DEV_WORKFLOW]

def generate_synthetic_data(output_path, num_events=500000):
    logger.info(f"Generating {num_events} synthetic events with GADGET CHAINS...")
    
    data = []
    start_time = time.time()
    current_time = start_time
    
    # Pre-calculate weights
    apps = APP_CATALOG
    app_weights = [app.weight for app in apps]
    
    chain_probability = 0.20 # 20% chance to start a chain (Increased for stronger learning)
    
    i = 0
    while i < num_events:
        # Decide: Single Event or Chain?
        if random.random() < chain_probability:
            # Generate Chain
            chain = random.choice(GADGET_CHAINS)
            
            # Common context for the chain
            user = random.choice(USERS)
            host = random.choice(HOSTS)
            
            # Link PIDs
            parent_pid = random.randint(1000, 9999)
            parent_name = "explorer.exe"
            
            for step_app in chain.steps:
                pid = random.randint(1000, 9999)
                file_path = generate_file_path(step_app, user)
                cmd_line = generate_command_line(step_app, file_path, user)
                
                event = {
                    "timestamp": current_time,
                    "event_type": "PROCESS_START",
                    "user": user,
                    "host": host,
                    "process_name": step_app.exe,
                    "parent_process": parent_name,
                    "file_path": file_path,
                    "command_line": cmd_line,
                    # "pid": pid, # Model doesn't learn PID integers well, but order matters
                    # "ppid": parent_pid
                }
                data.append(event)
                
                # Update for next step in chain
                parent_name = step_app.exe
                parent_pid = pid
                current_time += random.uniform(0.1, 1.0) # Fast sequence
                i += 1
                if i >= num_events: break
            
            continue # Skip single event logic
                
        else:
            # Single Event
            app = random.choices(apps, weights=app_weights, k=1)[0]
        
        # User correlation: Enforce strict user/system separation
        if app.install_scope == "user" or app.install_scope == "custom":
            # User-specific apps must run as a human user
            user = random.choice(["Xander", "dev_user", "jdoe"])
        elif app.category == "system":
            # System apps usually run as SYSTEM or Service accounts
            user = random.choice(["SYSTEM", "NETWORK SERVICE", "LOCAL SERVICE"])
        else:
            # Global apps (Program Files) can be run by anyone
            user = random.choice(USERS)
            
        host = random.choice(HOSTS)
        
        # 2. Generate Path
        file_path = generate_file_path(app, user)
        
        # 3. Generate Command Line
        cmd_line = generate_command_line(app, file_path, user)
        
        # 4. Parent Process Logic
        if app.category == "system":
            parent = "services.exe" if app.exe == "svchost.exe" else "explorer.exe"
        else:
            parent = "explorer.exe"
            
        event = {
            "timestamp": current_time,
            "event_type": "PROCESS_START",
            "user": user,
            "host": host,
            "process_name": app.exe,
            "parent_process": parent,
            "file_path": file_path,
            "command_line": cmd_line
        }
        
        data.append(event)
        
        # Time drift
        current_time += random.expovariate(1.0 / 2.0) # Avg 2 seconds between events
        
        if (i + 1) % 50000 == 0:
            logger.info(f"Generated {i + 1} events...")

    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    generate_synthetic_data("data/processed/local_train.parquet", num_events=200000)
