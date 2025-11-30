"""Windows event logger using WMI and psutil"""

import time
import logging
import threading
import threading
from datetime import datetime, timezone
from typing import List, Callable, Optional, Set
from queue import Queue
from pathlib import Path
import hashlib
import os
import psutil
import wmi
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import win32evtlog
    import win32con
    import win32api
    import win32security
except ImportError:
    win32evtlog = None

from .event_types import (
    EventType,
    BaseEvent,
    ProcessEvent,
    FileEvent,
    RegistryEvent,
    NetworkEvent,
    DLLEvent,
    DLLEvent,
)


logger = logging.getLogger(__name__)


class FileEventHandler(FileSystemEventHandler):
    """Handle file system events from watchdog"""

    def __init__(self, callback: Callable[[BaseEvent], None]):
        self.callback = callback

    def _get_file_hash(self, filepath: str) -> Optional[str]:
        """Calculate SHA256 hash of file (if < 100MB)"""
        try:
            # Check size first to avoid blocking on large files
            if os.path.getsize(filepath) > 100 * 1024 * 1024:
                return "SKIPPED_TOO_LARGE"
                
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return None

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_hash = self._get_file_hash(event.src_path)
        evt = FileEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.FILE_WRITE, # Treat creation as write
            pid=0, # Watchdog doesn't give PID
            process_name="unknown",
            user="unknown",
            file_path=event.src_path,
            operation="create",
            metadata={"sha256": file_hash} if file_hash else {}
        )
        self.callback(evt)

    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_hash = self._get_file_hash(event.src_path)
        evt = FileEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.FILE_WRITE,
            pid=0,
            process_name="unknown",
            user="unknown",
            file_path=event.src_path,
            operation="modify",
            metadata={"sha256": file_hash} if file_hash else {}
        )
        self.callback(evt)

    def on_deleted(self, event):
        if event.is_directory:
            return
            
        evt = FileEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.FILE_DELETE,
            pid=0,
            process_name="unknown",
            user="unknown",
            file_path=event.src_path,
            operation="delete"
        )
        self.callback(evt)


class WindowsEventLogger:
    """
    Comprehensive Windows event logger for cybersecurity data collection.

    Captures:
    - Process creation/termination
    - File I/O operations
    - Registry access
    - Network connections
    - DLL loading

    Uses a combination of WMI event subscriptions and psutil polling.
    """

    def __init__(
        self,
        event_types: Optional[List[EventType]] = None,
        callback: Optional[Callable[[BaseEvent], None]] = None,
        buffer_size: int = 10000,
        watch_dir: Optional[str] = None,
    ):
        """
        Initialize the event logger.

        Args:
            event_types: List of event types to monitor (None = all)
            callback: Function to call for each event
            buffer_size: Maximum events in buffer before flushing
            watch_dir: Directory to monitor for file events
        """
        self.event_types = event_types or list(EventType)
        self.callback = callback
        self.buffer_size = buffer_size
        self.watch_dir = watch_dir

        self.event_buffer: Queue = Queue(maxsize=buffer_size)
        self.running = False
        self.threads: List[threading.Thread] = []

        # Track seen processes/connections to detect changes
        self.seen_pids: Set[int] = set()
        self.seen_connections: Set[tuple] = set()

        # WMI connection
        try:
            self.wmi_conn = wmi.WMI()
        except Exception as e:
            logger.warning(f"Failed to initialize WMI: {e}")
            self.wmi_conn = None

        # File observer
        self.observer = None
        if self.watch_dir:
            self.observer = Observer()

    def _get_file_hash(self, filepath: str) -> Optional[str]:
        """Calculate SHA256 hash of file (if < 100MB)"""
        try:
            # Check size first to avoid blocking on large files
            if os.path.getsize(filepath) > 100 * 1024 * 1024:
                return "SKIPPED_TOO_LARGE"
                
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return None

    def start(self):
        """Start event monitoring"""
        if self.running:
            logger.warning("Logger already running")
            return

        self.running = True
        logger.info("Starting Windows event logger...")

        # Start monitoring threads
        if EventType.PROCESS_CREATE in self.event_types or EventType.PROCESS_EXIT in self.event_types:
            t = threading.Thread(target=self._monitor_processes, daemon=True)
            t.start()
            self.threads.append(t)

        if EventType.NETWORK_CONNECT in self.event_types:
            t = threading.Thread(target=self._monitor_network, daemon=True)
            t.start()
            self.threads.append(t)

        # WMI event watchers (process creation)
        if self.wmi_conn and EventType.PROCESS_CREATE in self.event_types:
            t = threading.Thread(target=self._wmi_process_watcher, daemon=True)
            t.start()
            self.threads.append(t)

        # File monitoring
        if self.observer and (EventType.FILE_WRITE in self.event_types or EventType.FILE_DELETE in self.event_types):
            event_handler = FileEventHandler(self._add_event)
            self.observer.schedule(event_handler, self.watch_dir, recursive=True)
            self.observer.start()
            logger.info(f"Started file monitoring on {self.watch_dir}")

        logger.info(f"Started {len(self.threads)} monitoring threads")

    def stop(self):
        """Stop event monitoring"""
        logger.info("Stopping event logger...")
        self.running = False

        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("Event logger stopped")
        
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def get_events(self, batch_size: Optional[int] = None) -> List[BaseEvent]:
        """
        Retrieve events from buffer.

        Args:
            batch_size: Maximum number of events to retrieve (None = all)

        Returns:
            List of events
        """
        events = []
        max_events = batch_size or self.event_buffer.qsize()

        for _ in range(min(max_events, self.event_buffer.qsize())):
            try:
                events.append(self.event_buffer.get_nowait())
            except:
                break

        return events

    def _add_event(self, event: BaseEvent):
        """Add event to buffer and optionally call callback"""
        try:
            self.event_buffer.put(event, block=False)
            if self.callback:
                self.callback(event)
        except:
            logger.warning("Event buffer full, dropping event")

    def _monitor_processes(self):
        """Monitor process creation and termination using psutil"""
        logger.info("Process monitoring thread started")

        while self.running:
            try:
                current_pids = set()

                for proc in psutil.process_iter(
                    ["pid", "name", "username", "create_time", "cmdline", "ppid", "exe"]
                ):
                    try:
                        info = proc.info
                        pid = info["pid"]
                        current_pids.add(pid)

                        # New process detected
                        if pid not in self.seen_pids:
                            image_path = info.get("exe")
                            image_hash = self._get_file_hash(image_path) if image_path else None
                            
                            event = ProcessEvent(
                                timestamp=datetime.fromtimestamp(info["create_time"], timezone.utc),
                                event_type=EventType.PROCESS_CREATE,
                                pid=pid,
                                process_name=info["name"] or "unknown",
                                user=info["username"] or "unknown",
                                parent_pid=info.get("ppid"),
                                command_line=" ".join(info.get("cmdline") or []),
                                image_path=image_path,
                                image_hash=image_hash,
                            )
                            self._add_event(event)
                            self.seen_pids.add(pid)

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Detect terminated processes
                if EventType.PROCESS_EXIT in self.event_types:
                    terminated = self.seen_pids - current_pids
                    for pid in terminated:
                        event = ProcessEvent(
                            timestamp=datetime.now(timezone.utc),
                            event_type=EventType.PROCESS_EXIT,
                            pid=pid,
                            process_name="unknown",
                            user="unknown",
                        )
                        self._add_event(event)

                    self.seen_pids = current_pids

                time.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                time.sleep(5)

    def _monitor_network(self):
        """Monitor network connections using psutil"""
        logger.info("Network monitoring thread started")

        while self.running:
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if conn.status == "ESTABLISHED":
                        # Create unique connection identifier
                        conn_id = (
                            conn.laddr.ip if conn.laddr else "",
                            conn.laddr.port if conn.laddr else 0,
                            conn.raddr.ip if conn.raddr else "",
                            conn.raddr.port if conn.raddr else 0,
                            conn.pid,
                        )

                        if conn_id not in self.seen_connections:
                            try:
                                proc = psutil.Process(conn.pid) if conn.pid else None
                                process_name = proc.name() if proc else "unknown"
                                user = proc.username() if proc else "unknown"
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                process_name = "unknown"
                                user = "unknown"

                            event = NetworkEvent(
                                timestamp=datetime.now(timezone.utc),
                                event_type=EventType.NETWORK_CONNECT,
                                pid=conn.pid or 0,
                                process_name=process_name,
                                user=user,
                                protocol=conn.type.name if hasattr(conn.type, "name") else "TCP",
                                local_address=conn.laddr.ip if conn.laddr else "",
                                local_port=conn.laddr.port if conn.laddr else 0,
                                remote_address=conn.raddr.ip if conn.raddr else "",
                                remote_port=conn.raddr.port if conn.raddr else 0,
                            )
                            self._add_event(event)
                            self.seen_connections.add(conn_id)

                # Clean up closed connections periodically
                if len(self.seen_connections) > 10000:
                    self.seen_connections.clear()

                time.sleep(2)  # Poll every 2 seconds

            except Exception as e:
                logger.error(f"Error in network monitoring: {e}")
                time.sleep(5)

    def _wmi_process_watcher(self):
        """Watch for process creation events using WMI (more reliable than polling)"""
        logger.info("WMI process watcher thread started")

        try:
            # Create WMI event watcher for process creation
            watcher = self.wmi_conn.Win32_Process.watch_for("creation")

            while self.running:
                try:
                    new_process = watcher(timeout_ms=1000)

                    if new_process:
                        image_path = new_process.ExecutablePath
                        image_hash = self._get_file_hash(image_path) if image_path else None
                        
                        event = ProcessEvent(
                            timestamp=datetime.now(timezone.utc),
                            event_type=EventType.PROCESS_CREATE,
                            pid=new_process.ProcessId,
                            process_name=new_process.Name,
                            user=new_process.GetOwner()[2] if new_process.GetOwner() else "unknown",
                            parent_pid=new_process.ParentProcessId,
                            command_line=new_process.CommandLine or "",
                            image_path=image_path,
                            image_hash=image_hash,
                        )
                        self._add_event(event)
                        self.seen_pids.add(new_process.ProcessId)

                except wmi.x_wmi_timed_out:
                    continue
                except Exception as e:
                    logger.error(f"WMI process watcher error: {e}")
                    time.sleep(5)

        except Exception as e:
            logger.error(f"Failed to start WMI watcher: {e}")


class EventCollector:
    """High-level interface for collecting events and saving to disk"""

    def __init__(self, event_logger: WindowsEventLogger, output_path: str, chunk_size: int = 10000):
        self.logger = event_logger
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.events: List[BaseEvent] = []
        
        # Scan for existing chunks to prevent overwrite
        self.chunk_counter = 0
        if self.output_path.parent.exists():
            stem = self.output_path.stem
            suffix = self.output_path.suffix
            # Pattern: stem_part_N.suffix
            existing_chunks = list(self.output_path.parent.glob(f"{stem}_part_*{suffix}"))
            if existing_chunks:
                max_chunk = -1
                for chunk in existing_chunks:
                    try:
                        # Extract N from stem_part_N.suffix
                        # chunk.stem is stem_part_N
                        parts = chunk.stem.split("_part_")
                        if len(parts) == 2:
                            n = int(parts[1])
                            if n > max_chunk:
                                max_chunk = n
                    except ValueError:
                        continue
                
                if max_chunk >= 0:
                    self.chunk_counter = max_chunk + 1
                    logger.info(f"Found existing chunks. Resuming at chunk {self.chunk_counter}")

    def collect(self, duration_seconds: int):
        """
        Collect events for specified duration.

        Args:
            duration_seconds: How long to collect events
        """
        logger.info(f"Collecting events for {duration_seconds} seconds...")
        self.logger.start()

        start_time = time.time()
        try:
            while time.time() - start_time < duration_seconds:
                # Retrieve events from buffer
                events = self.logger.get_events(batch_size=1000)
                self.events.extend(events)

                if len(self.events) >= self.chunk_size:
                    self._flush_to_disk()

                time.sleep(1)

        finally:
            self.logger.stop()

        # Flush remaining events
        if self.events:
            self._flush_to_disk()
            
        logger.info(f"Collection complete. Total chunks written: {self.chunk_counter}")

    def _flush_to_disk(self):
        """Write accumulated events to a new Parquet chunk"""
        import pandas as pd
        
        if not self.events:
            return

        logger.info(f"Flushing {len(self.events)} events to disk...")
        
        # Prepare output path for this chunk
        # If output_path is "events.parquet", chunks will be "events_part_0.parquet", "events_part_1.parquet", etc.
        stem = self.output_path.stem
        suffix = self.output_path.suffix
        chunk_filename = f"{stem}_part_{self.chunk_counter}{suffix}"
        chunk_path = self.output_path.parent / chunk_filename
        
        data = [e.to_dict() for e in self.events]
        # Convert metadata to string to avoid PyArrow empty struct error
        for d in data:
            if 'metadata' in d:
                d['metadata'] = str(d['metadata'])
        
        df = pd.DataFrame(data)
        
        # Ensure parent directory exists
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(str(chunk_path), index=False)
        logger.info(f"Saved chunk {self.chunk_counter} to {chunk_path}")
        
        self.chunk_counter += 1
        self.events = []  # Clear memory
