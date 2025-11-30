"""Event type definitions and data structures"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class EventType(Enum):
    """Types of Windows system events"""

    PROCESS_CREATE = "process_create"
    PROCESS_EXIT = "process_exit"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    REGISTRY_READ = "registry_read"
    REGISTRY_WRITE = "registry_write"
    NETWORK_CONNECT = "network_connect"
    NETWORK_DNS = "network_dns"
    DLL_LOAD = "dll_load"


@dataclass(kw_only=True)
class BaseEvent:
    """Base class for all events"""

    timestamp: datetime
    event_type: EventType
    pid: int
    process_name: str
    user: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "pid": self.pid,
            "process_name": self.process_name,
            "user": self.user,
            "metadata": self.metadata,
        }


@dataclass(kw_only=True)
class ProcessEvent(BaseEvent):
    """Process creation/exit event"""

    parent_pid: Optional[int] = None
    parent_process: Optional[str] = None
    command_line: Optional[str] = None
    exit_code: Optional[int] = None
    image_path: Optional[str] = None
    image_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "parent_pid": self.parent_pid,
                "parent_process": self.parent_process,
                "command_line": self.command_line,
                "exit_code": self.exit_code,
                "image_path": self.image_path,
                "image_hash": self.image_hash,
            }
        )
        return d


@dataclass(kw_only=True)
class FileEvent(BaseEvent):
    """File I/O event"""

    file_path: str
    operation: str  # read, write, delete
    bytes_transferred: Optional[int] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "file_path": self.file_path,
                "operation": self.operation,
                "bytes_transferred": self.bytes_transferred,
                "success": self.success,
            }
        )
        return d


@dataclass(kw_only=True)
class RegistryEvent(BaseEvent):
    """Registry access event"""

    key_path: str
    value_name: Optional[str] = None
    operation: str = "read"  # read, write, delete
    value_data: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "key_path": self.key_path,
                "value_name": self.value_name,
                "operation": self.operation,
                "value_data": self.value_data,
            }
        )
        return d


@dataclass(kw_only=True)
class NetworkEvent(BaseEvent):
    """Network connection event"""

    protocol: str  # TCP, UDP
    local_address: str
    local_port: int
    remote_address: str
    remote_port: int
    direction: str = "outbound"  # inbound, outbound
    dns_query: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "protocol": self.protocol,
                "local_address": self.local_address,
                "local_port": self.local_port,
                "remote_address": self.remote_address,
                "remote_port": self.remote_port,
                "direction": self.direction,
                "dns_query": self.dns_query,
            }
        )
        return d


@dataclass(kw_only=True)
class DLLEvent(BaseEvent):
    """DLL load event"""

    dll_path: str
    base_address: Optional[str] = None
    image_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "dll_path": self.dll_path,
                "base_address": self.base_address,
                "image_size": self.image_size,
            }
        )
        return d
