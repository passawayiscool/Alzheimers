# Anti-Forensics Demo - Memory Evasion Toolkit

Educational Windows kernel driver and user-mode tooling that demonstrate page table (PTE) manipulation, VAD tree hiding, and research paper–accurate PTE remapping techniques.

## ⚠️ Critical Warning
Hard‑coded for **Windows 10 x64 22H2 (Build 19041–19045)**. Using on other Windows versions (including 11) will likely trigger a **BSOD** due to mismatched structure offsets. Use only inside an isolated test VM with snapshots.

## Components
| Component | Purpose |
|----------|---------|
| `kernal driver/kernal_driver.c` | Kernel driver exposing IOCTL interface (`\\.\kernal_driver`) |
| `Main python/alzheimers.py` | User-mode controller (PTE, VAD, probe) |
| `innocent/innocent.c` + helpers | Target process & DLL for demonstration |

## Supported IOCTLs
| Code | Meaning | Summary |
|------|---------|---------|
| CTL_CODE(0x800,0x100) | READ_PTE | Read page table entry for virtual address |
| CTL_CODE(0x800,0x101) | WRITE_PTE | Write/restore a PTE value |
| CTL_CODE(0x800,0x200) | HIDE_VAD | Remove VAD node (region disappears from enumeration) |
| CTL_CODE(0x800,0x400) | PROBE_REGIONS | Enumerate process memory regions |

## Key Technique Highlights
1. **VAD Manipulation** – Hides a region by pruning its AVL node (Win10 offset 0x7d8 used here).
2. **PTE Nulling (Hide Content)** – Sets PTE to 0 to make content appear absent while physical data still exists.
3. **Probe Regions** – Enumerates process memory regions to select safe benign targets.
4. **Restoration Path** – Ability to restore a previously hidden PTE using saved value.

## Phase 1: VM & OS Configuration
Perform inside the Windows 10 22H2 x64 VM (Administrator).
```cmd
bcdedit /set testsigning on
bcdedit /set debug on
reg add "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard" /v EnableVirtualizationBasedSecurity /t REG_DWORD /d 0 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard" /v RequirePlatformSecurityFeatures /t REG_DWORD /d 0 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard\Scenarios\HypervisorEnforcedCodeIntegrity" /v Enabled /t REG_DWORD /d 0 /f
shutdown /r /t 0
```
Also disable: Windows Security → Device Security → Core isolation → Memory integrity (OFF) and disable Secure Boot in VM firmware.

## Phase 2: Build the Driver
Open Visual Studio (Administrator) with WDK installed → Set `Release | x64` → Build. Output: `AntiForensicDemo.sys` in the driver project folder.

## Phase 3: Certificate & Signing
Create and trust a self‑signed code signing certificate once; sign every build.
```powershell
# Create & export
$cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=AntiForensicDemo Test" -CertStoreLocation Cert:\CurrentUser\My
$pwd = ConvertTo-SecureString "P@ssw0rd!" -AsPlainText -Force
Export-PfxCertificate -Cert $cert -FilePath .\AntiForensicDemoTest.pfx -Password $pwd
Export-Certificate    -Cert $cert -FilePath .\AntiForensicDemoTest.cer
Import-Certificate -FilePath .\AntiForensicDemoTest.cer -CertStoreLocation Cert:\CurrentUser\TrustedPublisher
Import-Certificate -FilePath .\AntiForensicDemoTest.cer -CertStoreLocation Cert:\CurrentUser\Root
```
Sign & verify (`x64 Native Tools Command Prompt for VS`):
```cmd
signtool sign /fd SHA256 /f AntiForensicDemoTest.pfx /p P@ssw0rd! AntiForensicDemo.sys
signtool verify /kp /v AntiForensicDemo.sys
```

## Phase 4: Deploy Driver
```cmd
mkdir C:\Drivers
copy AntiForensicDemo.sys C:\Drivers\
sc create AntiForensicDemo type= kernel binPath= C:\Drivers\AntiForensicDemo.sys
sc start AntiForensicDemo
```

## Phase 5: Build Innocent Test Program
In **x64 Native Tools Command Prompt for VS (Administrator)** run:
```cmd
cd innocent
build.bat
```
Outputs: `helper.dll` and `innocent.exe`. Launch `innocent.exe` and record the printed PID plus any sample command lines for later driver interaction.

## Phase 6: Package Python Controller (`alzheimers.exe`)
From project root (PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pyinstaller --onefile "Main python\alzheimers.py" --name alzheimers
```
Result: `dist\alzheimers.exe`. Use that instead of `python alzheimers.py` for commands below.

## Phase 7: Usage Examples
PTE content hide (safe / reversible):
```cmd
alzheimers.exe pte <pid> <hex_virtual_address>
alzheimers.exe restore <pid> <hex_virtual_address> <old_hex_pte_value>
```
VAD hide (irreversible in session – creates zombie region):
```cmd
alzheimers.exe pte <pid> <hex_va>
alzheimers.exe vad <pid> <hex_va>
```
Probe regions:
```cmd
alzheimers.exe probe <pid>
```

## Phase 8: Driver Update Cycle
```cmd
sc stop AntiForensicDemo   # Reboot if stop hangs
signtool sign /fd SHA256 /f AntiForensicDemoTest.pfx /p P@ssw0rd! AntiForensicDemo.sys
copy /Y AntiForensicDemo.sys C:\Drivers\
sc start AntiForensicDemo
```

## Phase 9: Debugging (WinDbg)
```text
!process 0 0 innocent.exe          ; find EPROCESS
.process /r /p <EPROCESS_ADDRESS>  ; switch context
!pte <VirtualAddress>              ; inspect PTE before/after hide
; physical ghost: take old PTE value & zero last 3 hex digits
!db <PhysicalAddress>              ; view raw bytes
kv                                 ; stack trace if crash
!analyze -v                        ; automatic dump analysis
```

## Technical Details
PTE (x64) bits 0..63 include Present, RW, US, PWT, PCD, A, D, PS, Global, PFN (12–51), NX (63). VAD tree is an AVL rooted at `EPROCESS + 0x7d8` (Win10 build target). Demo omits `WorkingSetMutex` acquisition – race window possible. Physical pages intentionally leaked during some remap paths for crash prevention. Large pages (2MB/1GB) not fully supported. VBS / HVCI must be disabled.

## Limitations
- No WorkingSetMutex synchronization
- Kernel-aware tools still detect altered PTEs
- ETW/instrumentation can observe transitions
- Large page handling incomplete
- Requires exact Windows build offsets
- PTE remapping technique removed (historical documentation only)

## Cleanup
```cmd
sc stop AntiForensicDemo
sc delete AntiForensicDemo
bcdedit /set testsigning off
```

## Educational Value
Demonstrates page table traversal, PFN remapping, VAD AVL manipulation, kernel ↔ user IOCTL flows, and anti-forensics trade‑offs.

## Documentation Files
- `setup.md` – Detailed setup & phases
- `how_it_works.txt` – Architecture overview
- `innocent/BUILD_INSTRUCTIONS.txt` – Test program build steps
- `requirements.txt` – Python packaging requirements

## References
Windows Internals (Russinovich et al.), Intel SDM, WDK docs, academic papers on memory forensics & anti‑forensics.

---
License: Educational / demonstration use only. Author not responsible for misuse.
