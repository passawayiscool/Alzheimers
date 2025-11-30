"""Persona simulator for automated data collection in VMs"""

import time
import random
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PersonaType(Enum):
    """Types of user personas to simulate"""

    DEVELOPER = "developer"
    OFFICE_WORKER = "office_worker"
    CASUAL_USER = "casual_user"
    STUDENT = "student"
    PRODUCT_MANAGER = "product_manager"


@dataclass
class Activity:
    """Represents a single user activity"""

    name: str
    commands: List[str]
    duration_range: tuple[int, int]  # min, max seconds
    weight: float = 1.0  # Probability weight


class PersonaSimulator:
    """
    Simulates realistic user behavior for data collection.

    This class orchestrates various activities typical of different user personas,
    ensuring diverse and realistic training data for the ML model.
    """

    def __init__(self, persona_type: PersonaType, base_dir: Path):
        """
        Initialize persona simulator.

        Args:
            persona_type: Type of user persona to simulate
            base_dir: Base directory for file operations
        """
        self.persona_type = persona_type
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.activities = self._load_activities()

    def _load_activities(self) -> List[Activity]:
        """Load activities for the persona type"""
        if self.persona_type == PersonaType.DEVELOPER:
            return self._get_developer_activities()
        elif self.persona_type == PersonaType.OFFICE_WORKER:
            return self._get_office_worker_activities()
        elif self.persona_type == PersonaType.CASUAL_USER:
            return self._get_casual_user_activities()
        elif self.persona_type == PersonaType.STUDENT:
            return self._get_student_activities()
        elif self.persona_type == PersonaType.PRODUCT_MANAGER:
            return self._get_product_manager_activities()
        else:
            raise ValueError(f"Unknown persona type: {self.persona_type}")

    def _generate_realistic_url(self, base_domain: str, context: str = "general") -> str:
        """Generate a realistic URL with paths, queries, and fragments"""
        import uuid
        import urllib.parse
        
        paths = {
            "github": ["issues", "pulls", "actions", "wiki", "blob/main/src"],
            "stackoverflow": ["questions", "tags", "users"],
            "google": ["search", "maps", "news"],
            "amazon": ["dp", "gp/product", "s"],
            "general": ["article", "blog", "page", "category"]
        }
        
        # Select path
        key = next((k for k in paths if k in base_domain), "general")
        path = random.choice(paths[key])
        
        # Add random ID or slug
        if random.random() > 0.3:
            path += f"/{random.randint(1000, 99999)}"
        
        # Add query parameters
        params = {}
        if "google" in base_domain:
            queries = [
                "python error", "how to fix bug", "best ide 2025", "weather today",
                "news", "chatgpt vs claude", "stackoverflow", "github copilot"
            ]
            params["q"] = random.choice(queries)
            if random.random() > 0.5:
                params["tbs"] = "qdr:y" # Past year
        else:
            if random.random() > 0.5:
                params["ref"] = "sim_user"
            if random.random() > 0.7:
                params["session_id"] = str(uuid.uuid4())[:8]
                
        # Construct URL
        url = f"{base_domain}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
            
        return url

    def _browse_with_playwright(self, query: str, domain_hint: str = "google.com"):
        """
        Use Playwright to perform a realistic Google Search and Click.
        This generates 'real' traffic and avoids 404s by clicking actual results.
        """
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                # Launch headless chrome
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()
                
                # 1. Go to Google
                page.goto("https://www.google.com")
                time.sleep(random.uniform(1, 2))
                
                # 2. Type Query
                page.fill("textarea[name='q']", query) # Google changed input to textarea recently
                time.sleep(random.uniform(0.5, 1.5))
                page.press("textarea[name='q']", "Enter")
                
                # 3. Wait for results
                page.wait_for_selector("h3", timeout=5000)
                time.sleep(random.uniform(1, 3))
                
                # 4. Click a result (preferably matching domain_hint)
                links = page.query_selector_all("h3")
                target_link = None
                
                if links:
                    # Try to find a link that matches our hint, otherwise pick random top 3
                    for link in links[:5]:
                        # Get parent anchor
                        parent = link.xpath("..")[0] 
                        href = parent.get_attribute("href")
                        if href and domain_hint in href:
                            target_link = link
                            break
                    
                    if not target_link and links:
                        target_link = random.choice(links[:3])
                        
                    if target_link:
                        target_link.click()
                        time.sleep(random.uniform(5, 10)) # Stay on page
                        
                browser.close()
                
        except Exception as e:
            # Fallback if playwright fails or not installed
            print(f"Playwright failed: {e}. Falling back to start chrome.")
            import subprocess
            subprocess.run(f'start chrome "https://google.com/search?q={query}"', shell=True)

    def _get_developer_activities(self) -> List[Activity]:
        """Activities for developer persona"""
        import uuid
        return [
            Activity(
                name="code_editing",
                commands=[
                    f"notepad {self.base_dir / f'test_{str(uuid.uuid4())[:8]}.py'}",
                    f"echo print('Hello World') > {self.base_dir / f'script_{str(uuid.uuid4())[:8]}.py'}",
                    f"code {self.base_dir / 'src'} || notepad {self.base_dir / 'README.md'}",
                ],
                duration_range=(30, 300),
                weight=3.0,
            ),
            Activity(
                name="git_operations",
                commands=[
                    "git status",
                    "git log --oneline -10",
                    f"cd {self.base_dir} && git init",
                    "git diff",
                    "git branch -a"
                ],
                duration_range=(10, 60),
                weight=2.0,
            ),
            Activity(
                name="web_research",
                # We use a special prefix "PLAYWRIGHT:" to tell the runner to use the python method
                # But wait, the runner executes commands via subprocess.
                # We need to wrap this logic.
                # Actually, we can just execute a python one-liner that calls the method?
                # No, that's too complex.
                # BETTER: The 'commands' list usually expects shell commands.
                # But we can make the runner handle a special callable?
                # OR: We just run a python script that does the playwright stuff.
                # EASIEST: We update 'run_simulation_and_log.py' to handle "python:..." commands?
                # OR: We just put the python code in a string and run `python -c ...`?
                # That's messy with imports.
                
                # Let's stick to the existing architecture: The Simulator returns a list of commands.
                # I will create a helper script `scripts/browse.py` that takes a query and uses Playwright.
                commands=[
                    f'python scripts/browse.py "python list index out of range" "stackoverflow.com"',
                    f'python scripts/browse.py "how to use git" "atlassian.com"',
                    f'python scripts/browse.py "python 3.12 features" "python.org"',
                    f'python scripts/browse.py "best vscode extensions" "marketplace.visualstudio.com"',
                ],
                duration_range=(60, 300),
                weight=3.0,
            ),
            Activity(
                name="debugging",
                commands=[
                    f"python {self.base_dir / 'script.py'}",
                    "tasklist | findstr python",
                    "where python",
                    "python --version"
                ],
                duration_range=(30, 180),
                weight=1.5,
            ),
            Activity(
                name="powershell_ops",
                commands=[
                    "powershell -Command \"Get-Process | Sort-Object CPU -Descending | Select-Object -First 5\"",
                    "powershell -Command \"Get-Service | Where-Object {$_.Status -eq 'Running'}\"",
                    "powershell -Command \"Get-ChildItem -Path C:\\Windows\\System32 -Filter *.dll | Measure-Object\"",
                    "powershell -Command \"$PSVersionTable\"",
                    "powershell -Command \"Get-EventLog -LogName Application -Newest 10\"",
                    f"powershell -Command \"Get-Content {self.base_dir / 'README.md'} | Select-Object -First 5\"",
                    "powershell -Command \"Test-NetConnection -ComputerName google.com -Port 443\"",
                    "powershell -Command \"Get-Date -Format 'yyyy-MM-dd HH:mm:ss'\"",
                ],
                duration_range=(30, 120),
                weight=2.5,
            ),
            Activity(
                name="network_diagnostics",
                commands=[
                    "powershell -Command \"Resolve-DnsName google.com\"",
                    "powershell -Command \"Get-NetAdapter\"",
                    "powershell -Command \"Get-NetIPAddress | Format-Table\"",
                    "powershell -Command \"Invoke-WebRequest -Uri https://www.google.com -UseBasicParsing | Select-Object StatusCode, StatusDescription\"",
                ],
                duration_range=(30, 90),
                weight=2.0,
            ),
        ]

    def _get_office_worker_activities(self) -> List[Activity]:
        """Activities for office worker persona"""
        return [
            Activity(
                name="productivity_apps",
                commands=[
                    'start outlook || start chrome "https://outlook.office.com"',
                    'start onenote || notepad "Notes.txt"',
                    'start excel || start chrome "https://docs.google.com/spreadsheets"',
                    'start winword || write',
                ],
                duration_range=(60, 600),
                weight=3.0,
            ),
            Activity(
                name="file_management",
                commands=[
                    f"explorer {self.base_dir}",
                    f"dir {self.base_dir}",
                    f"mkdir {self.base_dir / 'Quarterly_Reports'}",
                    f"copy NUL {self.base_dir / 'Meeting_Notes.docx'}",
                ],
                duration_range=(30, 120),
                weight=2.5,
            ),
            Activity(
                name="web_browsing",
                commands=[
                    f'python scripts/browse.py "linkedin login" "linkedin.com"',
                    f'python scripts/browse.py "salesforce crm" "salesforce.com"',
                    f'python scripts/browse.py "google news" "news.google.com"',
                    f'python scripts/browse.py "office 365 login" "office.com"',
                ],
                duration_range=(60, 300),
                weight=2.0,
            ),
        ]

    def _get_casual_user_activities(self) -> List[Activity]:
        """Activities for casual user persona"""
        return [
            Activity(
                name="web_surfing",
                commands=[
                    f'python scripts/browse.py "funny cat videos" "youtube.com"',
                    f'python scripts/browse.py "reddit front page" "reddit.com"',
                    f'python scripts/browse.py "amazon best sellers" "amazon.com"',
                    f'python scripts/browse.py "netflix movies" "netflix.com"',
                    f'python scripts/browse.py "twitch top streamers" "twitch.tv"',
                ],
                duration_range=(120, 600),
                weight=4.0,
            ),
            Activity(
                name="social_media",
                commands=[
                    f'python scripts/browse.py "twitter login" "twitter.com"',
                    f'python scripts/browse.py "facebook login" "facebook.com"',
                    f'python scripts/browse.py "instagram login" "instagram.com"',
                ],
                duration_range=(60, 300),
                weight=3.0,
            ),
            Activity(
                name="file_browsing",
                commands=[
                    f"explorer {self.base_dir}",
                    "explorer C:\\Users",
                ],
                duration_range=(30, 180),
                weight=2.0,
            ),
        ]

    def _get_student_activities(self) -> List[Activity]:
        """Activities for student persona"""
        return [
            Activity(
                name="research",
                commands=[
                    f'python scripts/browse.py "google scholar" "scholar.google.com"',
                    f'python scripts/browse.py "wikipedia history of computing" "wikipedia.org"',
                    f'python scripts/browse.py "chegg homework help" "chegg.com"',
                    f'python scripts/browse.py "stackoverflow python help" "stackoverflow.com"',
                    f'python scripts/browse.py "canvas lms login" "instructure.com"',
                ],
                duration_range=(120, 600),
                weight=3.0,
            ),
            Activity(
                name="writing",
                commands=[
                    "write", # WordPad
                    f"notepad {self.base_dir / 'Essay_Draft_v1.txt'}",
                    f"echo Bibliography > {self.base_dir / 'Sources.txt'}",
                ],
                duration_range=(300, 900),
                weight=3.0,
            ),
        ]

    def _get_product_manager_activities(self) -> List[Activity]:
        """Activities for product manager persona"""
        return [
            Activity(
                name="planning",
                commands=[
                    f'python scripts/browse.py "jira dashboard" "atlassian.com"',
                    f'python scripts/browse.py "trello boards" "trello.com"',
                    f'python scripts/browse.py "confluence documentation" "atlassian.com"',
                ],
                duration_range=(120, 600),
                weight=3.0,
            ),
            Activity(
                name="communication",
                commands=[
                    f'python scripts/browse.py "slack login" "slack.com"',
                    f'python scripts/browse.py "microsoft teams" "teams.microsoft.com"',
                    f'python scripts/browse.py "outlook web" "outlook.office.com"',
                ],
                duration_range=(60, 300),
                weight=3.0,
            ),
        ]

    def run_simulation(self, duration_seconds: int):
        """
        Run persona simulation for specified duration.

        Args:
            duration_seconds: How long to run simulation
        """
        logger.info(
            f"Starting {self.persona_type.value} persona simulation for {duration_seconds} seconds"
        )

        start_time = time.time()
        activities_performed = 0

        try:
            while time.time() - start_time < duration_seconds:
                # Select activity based on weights
                activity = random.choices(
                    self.activities, weights=[a.weight for a in self.activities], k=1
                )[0]

                logger.info(f"Performing activity: {activity.name}")

                # Execute activity commands
                for cmd in activity.commands:
                    try:
                        # Execute command but don't wait for completion (non-blocking)
                        subprocess.Popen(
                            cmd,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            creationflags=0,
                        )
                        time.sleep(random.uniform(1, 5))  # Random delay between commands

                    except Exception as e:
                        logger.warning(f"Failed to execute command '{cmd}': {e}")

                # Wait for activity duration
                duration = random.randint(*activity.duration_range)
                logger.debug(f"Activity '{activity.name}' will run for {duration} seconds")
                time.sleep(min(duration, duration_seconds - (time.time() - start_time)))

                activities_performed += 1

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")

        elapsed = time.time() - start_time
        logger.info(
            f"Simulation complete. Performed {activities_performed} activities in {elapsed:.1f} seconds"
        )


class VMAutomationController:
    """
    Controller for managing VM-based data collection.

    This would integrate with virtualization platforms (VirtualBox, VMware, Hyper-V)
    to automate VM lifecycle and run persona simulations inside VMs.
    """

    def __init__(self, vm_platform: str = "virtualbox"):
        """
        Initialize VM controller.

        Args:
            vm_platform: Virtualization platform (virtualbox, vmware, hyper-v)
        """
        self.vm_platform = vm_platform
        logger.info(f"Initialized VM controller for platform: {vm_platform}")

    def start_vm(self, vm_name: str):
        """Start a VM"""
        logger.info(f"Starting VM: {vm_name}")

        if self.vm_platform == "virtualbox":
            subprocess.run(["VBoxManage", "startvm", vm_name, "--type", "headless"])
        else:
            raise NotImplementedError(f"Platform {self.vm_platform} not implemented")

        # Wait for VM to boot
        time.sleep(30)

    def stop_vm(self, vm_name: str):
        """Stop a VM"""
        logger.info(f"Stopping VM: {vm_name}")

        if self.vm_platform == "virtualbox":
            subprocess.run(["VBoxManage", "controlvm", vm_name, "poweroff"])
        else:
            raise NotImplementedError(f"Platform {self.vm_platform} not implemented")

    def execute_in_vm(self, vm_name: str, command: str, username: str, password: str):
        """
        Execute command inside VM.

        Args:
            vm_name: Name of VM
            command: Command to execute
            username: VM username
            password: VM password
        """
        if self.vm_platform == "virtualbox":
            subprocess.run(
                [
                    "VBoxManage",
                    "guestcontrol",
                    vm_name,
                    "run",
                    "--exe",
                    "cmd.exe",
                    "--username",
                    username,
                    "--password",
                    password,
                    "--",
                    "/c",
                    command,
                ]
            )
        else:
            raise NotImplementedError(f"Platform {self.vm_platform} not implemented")

    def collect_data_from_vm(
        self, vm_name: str, persona_type: PersonaType, duration_seconds: int, output_path: str
    ):
        """
        Full workflow: Start VM, run simulation, collect data, stop VM.

        Args:
            vm_name: Name of VM
            persona_type: Type of persona to simulate
            duration_seconds: How long to run simulation
            output_path: Where to save collected data
        """
        logger.info(f"Starting data collection from VM '{vm_name}' with persona '{persona_type.value}'")

        try:
            # Start VM
            self.start_vm(vm_name)

            # TODO: Install event logger in VM
            # TODO: Start event logger in VM
            # TODO: Run persona simulation in VM
            # TODO: Retrieve collected data from VM

            logger.info("Data collection complete")

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise

        finally:
            # Stop VM
            self.stop_vm(vm_name)
