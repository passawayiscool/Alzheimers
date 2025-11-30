# AntiForensic Driver Setup & Development Guide

## Phase 1: Virtual Machine Configuration
**CRITICAL**: This driver is HARDCODED for Windows 10 x64 22H2 (Build 19041–19045).
Do NOT run this on Windows 11, Windows 8, or older Windows 10 builds, or it will cause an immediate BSOD (System Crash) due to mismatched kernel offsets.


1.  **VM Settings (Hypervisor Level):**
    *   Disable **Secure Boot** in the VM firmware/settings.

2.  **Windows Registry & Security Configuration:**
    *   Open **CMD as Administrator** and run the following to disable VBS, Device Guard, and enable Test Signing:
    ```cmd
    bcdedit /set testsigning on
    bcdedit /set debug on

    :: Disable Virtualization Based Security (VBS) and Device Guard
    reg add "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard" /v "EnableVirtualizationBasedSecurity" /t REG_DWORD /d 0 /f
    reg add "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard" /v "RequirePlatformSecurityFeatures" /t REG_DWORD /d 0 /f
    reg add "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard\Scenarios\HypervisorEnforcedCodeIntegrity" /v "Enabled" /t REG_DWORD /d 0 /f
    ```

3.  **Disable Memory Integrity:**
    *   Go to **Windows Security** -> **Device Security** -> **Core isolation**.
    *   Ensure **Memory integrity** is set to **OFF**.

4.  **Reboot the VM:**
    ```cmd
    shutdown /r /t 0
    ```

---

## Phase 2: Building the Driver
*Reference: [Simple Setup Video Step](https://youtube.com/shorts/rOMhIsa8Xds?si=dsWuRCjC9-3O0QFC)*

1.  Open Visual Studio **as Administrator**.
2.  Open your solution/project (containing `kernal_driver.c`).
3.  Set build configuration to **Release** and Architecture to **x64**.
4.  **Build Solution**.
    *   *Output:* This should generate `AntiForensicDemo.sys`.

---

## Phase 3: Certificate Setup & Signing
*You only need to generate the certificate (Step A) once. You must sign (Step B) every time you rebuild the driver.*

### A. Create and Trust Self-Signed Cert (Run in Admin PowerShell)
```powershell
# 1. Create Certificate
$cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=AntiForensicDemo Test" -CertStoreLocation Cert:\CurrentUser\My
$pwd = ConvertTo-SecureString "P@ssw0rd!" -AsPlainText -Force
Export-PfxCertificate -Cert $cert -FilePath .\AntiForensicDemoTest.pfx -Password $pwd
Export-Certificate    -Cert $cert -FilePath .\AntiForensicDemoTest.cer

# 2. Trust Certificate
Import-Certificate -FilePath .\AntiForensicDemoTest.cer -CertStoreLocation Cert:\CurrentUser\TrustedPublisher
Import-Certificate -FilePath .\AntiForensicDemoTest.cer -CertStoreLocation Cert:\CurrentUser\Root
```

### B. Sign the Driver (Run in "x64 Native Tools Command Prompt for VS")
*Navigate to the folder containing your .sys file and .pfx file.*
```cmd
signtool sign /fd SHA256 /f AntiForensicDemoTest.pfx /p P@ssw0rd! AntiForensicDemo.sys
signtool verify /kp /v AntiForensicDemo.sys
```

---

## Phase 4: Deployment & Testing

### 1. Install the Driver Service
*Run in Admin CMD:*
```cmd
mkdir C:\Drivers
copy AntiForensicDemo.sys C:\Drivers\
sc create AntiForensicDemo type= kernel binPath= C:\Drivers\AntiForensicDemo.sys
sc start AntiForensicDemo
```

### 2. Build and Run the Test App ("Innocent Process")
*Run in x64 Native Tools Command Prompt:*
```cmd
cl /O2 innocent.c /Fe:innocent.exe
.\innocent.exe
```
*Note: The app will print its PID and the specific commands to run.*

### 3. Run the Controller (alzheimers.exe)
*Run in Admin CMD. Ensure `alzheimers.exe` is in the directory.*

**A. PTE Hiding (Safe Mode - Can Exit Cleanly)**
Use this to hide memory content (`??`) but keep the region listed.
```cmd
:: 1. Hide the memory
alzheimers.exe pte <pid> <hex_virtual_address>

:: 2. Restore the memory (Use the Old Value printed by step 1)
alzheimers.exe restore <pid> <hex_virtual_address> <hex_old_pte_value>
```

**B. VAD Hiding (Rootkit Mode - Zombie Process)**
Use this to remove the memory region from lists (Process Hacker).
**WARNING:** You cannot safely close `innocent.exe` after this. You must power off the VM.
```cmd
:: 1. Hide the content first
alzheimers.exe pte <pid> <hex_virtual_address>

:: 2. Delete the administrative record
alzheimers.exe vad <pid> <hex_virtual_address>
```

---

## Phase 5: Iterative Development (Updating Code)
If you modify `kernal_driver.c` and rebuild, follow this sequence to update the running driver:

1.  **Stop the service:**
    ```cmd
    sc stop AntiForensicDemo
    ```
    *(If `sc stop` fails or hangs due to open handles, you must reboot the VM).*

2.  **Re-sign the new .sys file:**
    ```cmd
    signtool sign /fd SHA256 /f AntiForensicDemoTest.pfx /p P@ssw0rd! AntiForensicDemo.sys
    ```

3.  **Replace the file:**
    ```cmd
    copy /Y AntiForensicDemo.sys C:\Drivers\
    ```

4.  **Start the service:**
    ```cmd
    sc start AntiForensicDemo
    ```

---

## Phase 6: Debugging (WinDbg Commands)
If checking memory states via Kernel Debugger:

1.  **Find Process Info:**
    ```text
    !process 0 0 innocent.exe
    ```
    *(Copy the `PROCESS` address, e.g., `ffff9c0a...`)*

2.  **Switch Context:**
    ```text
    .process /r /p <PROCESS_ADDRESS>
    ```

3.  **Inspect Page Table Entry (Virtual):**
    ```text
    !pte <Virtual_Address>
    ```
    *   *Before Attack:* Shows valid hex (e.g., `...025`).
    *   *After Attack:* Shows `0000000000000000`.

4.  **Inspect Physical Memory (The "Ghost" Data):**
    *   Take the Old PTE Value (e.g., `0x12345025`).
    *   Change last 3 digits to 0 (`0x12345000`).
    ```text
    !db <Physical_Address>
    ```

## Phase 7: Debugging (BSOD Analysis via Memory Dump)

If your VM crashes (Blue Screen of Death), Windows saves a record of the memory state. Analyzing this on your Host machine can tell you exactly which line of code caused the crash.

### 1. Locate the Dump File
*   **Inside the VM**, copy the dump file to your Host machine (via Shared Folder or Copy/Paste).
*   **Full Dump:** `C:\Windows\MEMORY.DMP` (Contains everything).
*   **Minidump:** `C:\Windows\Minidump\*.dmp` (Contains just the crash stack).

### 2. Open in WinDbg (On Host)
1.  Open **WinDbg Preview** (or Classic).
2.  **File** -> **Open Dump File** -> Select the `.dmp` file you copied.

### 3. The KV Command (Stack Trace)
This is the most important command. It shows the "Chain of Events" leading to the crash.

```text
kv
```

**How to read the Output:**
Look at the **"Call Site"** column. Read from **Bottom to Top**.

*   **Bottom rows:** Generic Windows kernel functions (`nt!KiStartSystemThread`).
*   **Middle rows:** **YOUR DRIVER** (`AntiForensicDemo!IoctlWritePte`).
*   **Top rows:** The specific crash handler (`nt!KeBugCheckEx`).

**Example Analysis:**
```text
# Child-SP          RetAddr               Call Site
00 ffff...          fffff...              nt!KeBugCheckEx          <-- The Crash (STOP)
01 ffff...          fffff...              nt!MiDeleteFinalPageTables <-- Windows cleanup routine
...
05 ffff...          fffff...              AntiForensicDemo!IoctlWritePte+0x40  <-- YOUR CODE
```
*   *Interpretation:* The driver called a function, which eventually triggered a memory management cleanup routine, which detected corruption and called BugCheck.

### 4. Inspecting Variables
If `kv` shows the crash happened at `AntiForensicDemo!IoctlWritePte`, you can inspect the registers/memory at that moment (if using a Full Dump):

```text
!analyze -v
```
*   This command performs an automated analysis and often points directly to the specific error type (e.g., `MEMORY_MANAGEMENT`) and the culprit driver.