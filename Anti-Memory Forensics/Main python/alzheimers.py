import ctypes
import sys
from ctypes import wintypes as w

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# -----------------------------------------------------------------------------
# WinAPI Definitions
# -----------------------------------------------------------------------------

kernel32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
kernel32.CreateFileW.restype = w.HANDLE

kernel32.DeviceIoControl.argtypes = [
    w.HANDLE, w.DWORD, w.LPVOID, w.DWORD,
    w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID
]
kernel32.DeviceIoControl.restype = w.BOOL

kernel32.CloseHandle.argtypes = [w.HANDLE]
kernel32.CloseHandle.restype = w.BOOL

# Constants
DEVICE_NAME = r"\\.\AntiForensicDemo"

# IOCTL Calculation
def CTL_CODE(device_type, function, method=0, access=0):
    return (device_type << 16) | (access << 14) | (function << 2) | method

# IOCTL Codes
IOCTL_WRITE_PTE = CTL_CODE(0x800, 0x101, method=0, access=0)
IOCTL_HIDE_VAD = CTL_CODE(0x800, 0x200, method=0, access=0)
IOCTL_PROBE_REGIONS = CTL_CODE(0x800, 0x400, method=0, access=0)

# Types
ULONGLONG = ctypes.c_uint64

# -----------------------------------------------------------------------------
# Structures (Must match Kernel Driver EXACTLY)
# -----------------------------------------------------------------------------

class PTE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("ProcessId", w.ULONG),
        # C compiler adds padding to align ULONGLONG to 8 bytes
        ("VirtualAddress", ULONGLONG),
        ("PteValue", ULONGLONG),    # Input: Value to write
        ("OldValue", ULONGLONG),    # Output: Previous value
        ("Success", w.BOOLEAN),     # Output: Result
    ]

class VAD_HIDE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("ProcessId", w.ULONG),
        ("StartAddress", ULONGLONG),
        ("EndAddress", ULONGLONG),
    ]

class PROBE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("ProcessId", w.ULONG),
        ("RegionCount", w.ULONG),
    ]

class MEMORY_REGION(ctypes.Structure):
    _fields_ = [
        ("StartAddress", ULONGLONG),
        ("EndAddress", ULONGLONG),
        ("Protection", w.ULONG),
        ("Type", w.ULONG),
    ]

# -----------------------------------------------------------------------------
# Logic
# -----------------------------------------------------------------------------

def ioctl(handle, code, in_buffer, out_buffer_type=None):
    """Helper to send IOCTLs"""
    bytes_ret = w.DWORD()
    
    if out_buffer_type:
        out_buf = out_buffer_type()
    else:
        out_buf = in_buffer

    success = kernel32.DeviceIoControl(
        handle,
        code,
        ctypes.byref(in_buffer), ctypes.sizeof(in_buffer),
        ctypes.byref(out_buf), ctypes.sizeof(out_buf),
        ctypes.byref(bytes_ret),
        None)
        
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())
        
    return out_buf

def pte_make_nonpresent(handle, pid: int, va: int):
    print(f"[*] Sending PTE HIDE request for PID {pid} @ {hex(va)}...")
    
    req = PTE_REQUEST()
    req.ProcessId = pid
    req.VirtualAddress = va
    req.PteValue = 0        # 0 = Not Present / Hidden
    req.OldValue = 0       
    req.Success = False    

    try:
        res = ioctl(handle, IOCTL_WRITE_PTE, req)
        
        if res.Success:
            print(f"[+] Success! PTE Modified.")
            print(f"    Old PTE Value: {hex(res.OldValue)}")
            print(f"    New PTE Value: {hex(res.PteValue)}")
            print("    [!] SAVE THE OLD VALUE! You need it to restore/verify.")
        else:
            print("[-] Driver returned failure (Check DbgView).")
            
    except OSError as e:
        print(f"[-] IOCTL Failed: {e}")

def pte_restore(handle, pid: int, va: int, old_pte_value: int):
    print(f"[*] Restoring PTE for PID {pid} @ {hex(va)}...")
    print(f"    Restoring Value: {hex(old_pte_value)}")
    
    req = PTE_REQUEST()
    req.ProcessId = pid
    req.VirtualAddress = va
    req.PteValue = old_pte_value  # Write the valid value back
    req.OldValue = 0
    req.Success = False

    try:
        res = ioctl(handle, IOCTL_WRITE_PTE, req)
        if res.Success:
            print(f"[+] Success! PTE Restored.")
            print(f"    The memory is visible to the OS again.")
            print("    You can now safely close the target process without BSOD.")
        else:
            print("[-] Driver returned failure.")
    except OSError as e:
        print(f"[-] IOCTL Failed: {e}")

def hide_vad(handle, pid: int, start_va: int, end_va: int):
    print(f"[*] Requesting VAD Unlink for PID {pid} Range {hex(start_va)}-{hex(end_va)}...")
    
    req = VAD_HIDE_REQUEST()
    req.ProcessId = pid
    req.StartAddress = start_va
    req.EndAddress = end_va

    try:
        ioctl(handle, IOCTL_HIDE_VAD, req)
        print("[+] VAD Hiding request sent.")
        print("    Check Process Hacker: The memory region should be gone.")
        print("    Check DbgView: Look for '[SUCCESS] Removed ... VAD'")
    except OSError as e:
        print(f"[-] IOCTL Failed: {e}")

def probe_regions(handle, pid: int):
    print(f"[*] Probing memory regions for PID {pid}...")
    
    req = PROBE_REQUEST(ProcessId=pid, RegionCount=0)
    
    MAX_REGIONS = 2000
    output_buffer = (MEMORY_REGION * MAX_REGIONS)()
    bytes_ret = w.DWORD()
    
    success = kernel32.DeviceIoControl(
        handle,
        IOCTL_PROBE_REGIONS,
        ctypes.byref(req), ctypes.sizeof(req),
        ctypes.byref(output_buffer), ctypes.sizeof(output_buffer),
        ctypes.byref(bytes_ret),
        None)
    
    if not success:
        print(f"[-] Failed to probe regions: {ctypes.WinError(ctypes.get_last_error())}")
        return

    count = bytes_ret.value // ctypes.sizeof(MEMORY_REGION)
    print(f"[+] Found {count} regions.\n")
    
    print(f"{'Start':<18} {'End':<18} {'Size':<10} {'Type':<10} {'Prot'}")
    print("-" * 70)
    
    for i in range(count):
        reg = output_buffer[i]
        size = reg.EndAddress - reg.StartAddress
        
        t_str = "Unk"
        if reg.Type == 0x20000: t_str = "Priv"
        elif reg.Type == 0x40000: t_str = "Map"
        elif reg.Type == 0x1000000: t_str = "Img"
        
        print(f"{hex(reg.StartAddress):<18} {hex(reg.EndAddress):<18} {hex(size):<10} {t_str:<10} {reg.Protection}")

# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------

def usage():
    print("Usage: python alzheimers.py <op> <pid> [args]")
    print("Commands:")
    print("  probe <pid>")
    print("  pte <pid> <hex_addr>                (Hide memory content)")
    print("  restore <pid> <hex_addr> <hex_val>  (Restore memory)")
    print("  vad <pid> <hex_addr> [hex_end]      (Delete memory record)")

def main():
    if len(sys.argv) < 3:
        usage() 
        return

    op = sys.argv[1].lower()
    try:
        pid = int(sys.argv[2])
    except ValueError:
        print("Invalid PID")
        return

    handle = kernel32.CreateFileW(
        DEVICE_NAME,
        0xC0000000, # GENERIC_READ | GENERIC_WRITE
        0, None, 3, 0, None # OPEN_EXISTING
    )

    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    if handle == INVALID_HANDLE_VALUE or handle == 0:
        print(f"[-] Failed to open driver: {DEVICE_NAME}")
        print("    Make sure the driver is loaded (sc start AntiForensicDemo)")
        return

    try:
        if op == "probe":
            probe_regions(handle, pid)

        elif op == "pte":
            if len(sys.argv) < 4:
                print("Error: Missing address")
                return
            addr = int(sys.argv[3], 16)
            pte_make_nonpresent(handle, pid, addr)

        elif op == "restore":
            if len(sys.argv) < 5:
                print("Error: Restore requires <pid> <hex_va> <hex_old_value>")
                return
            addr = int(sys.argv[3], 16)
            old_val = int(sys.argv[4], 16)
            pte_restore(handle, pid, addr, old_val)

        elif op == "vad":
            if len(sys.argv) < 4:
                print("Error: VAD operation requires <pid> <hex_address>")
                return
            
            start = int(sys.argv[3], 16)
            
            # Optional End Address logic (Point Query vs Range)
            if len(sys.argv) >= 5:
                end = int(sys.argv[4], 16)
            else:
                print("[*] Point-Query Mode: Searching for VAD containing 0x%x" % start)
                end = start 

            hide_vad(handle, pid, start, end)

        else:
            usage()

    except Exception as e:
        print(f"[-] An error occurred: {e}")

    finally:
        if handle and handle != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(handle)

if __name__ == "__main__":
    main()