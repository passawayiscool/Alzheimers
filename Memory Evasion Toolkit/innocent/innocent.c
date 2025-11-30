#include <stdio.h>
#include <windows.h>
#include <process.h> 
#include <conio.h> 

typedef void (*InitFunc)(void);
typedef void (*WorkFunc)(void);
typedef void (*ActivityFunc)(void); 
typedef int (*SecretFunc)(void);

// REMOVED GLOBAL SHELLCODE FROM HERE
 
int main(void) {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    printf("========================================\n");
    printf("Anti-Forensics Demo - innocent.exe (Interactive Mode)\n");
    printf("========================================\n");
    printf("Process ID: %d\n", _getpid());
    printf("\n");
    
    HMODULE hDll = LoadLibrary("helper.dll");
    if (!hDll) {
        printf("Failed to load helper.dll\n");
        return 1;
    }

    InitFunc init_func = (InitFunc)GetProcAddress(hDll, "initialize_config");
    WorkFunc work_func = (WorkFunc)GetProcAddress(hDll, "do_work");
    ActivityFunc noise_func = (ActivityFunc)GetProcAddress(hDll, "simulate_rootkit_activity");
    
    // ---------------------------------------------------------
    // STEP 1: DEFINE SHELLCODE (Local Stack Variable)
    // ---------------------------------------------------------
    // MOV EAX, 0x1337; RET
    unsigned char local_shellcode[] = { 
        0xB8, 0x37, 0x13, 0x00, 0x00, 
        0xC3 
    };

    // ---------------------------------------------------------
    // STEP 2: ALLOCATE (VirtualAlloc)
    // ---------------------------------------------------------
    void* secret_mem = VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if (!secret_mem) return 1;

    // ---------------------------------------------------------
    // STEP 3: INJECT & WIPE SOURCE
    // ---------------------------------------------------------
    // Copy to the target
    memcpy(secret_mem, local_shellcode, sizeof(local_shellcode));

    // *** WIPE FIX ***
    // We immediately overwrite the stack copy with zeros.
    // This ensures the ONLY copy of the code is in the hidden memory.
    RtlSecureZeroMemory(local_shellcode, sizeof(local_shellcode));

    printf("[SETUP] 1. Secret Injected at:   %p (Private VAD)\n", secret_mem);
    printf("[SETUP] 2. Source Code Wiped:    Stack Cleaned.\n");
    printf("[SETUP] 3. Helper DLL Loaded at: %p (For camouflage)\n", hDll);
    printf("\n");

    if(init_func) init_func();
    
    int res = ((SecretFunc)secret_mem)(); 
    printf("[PHASE 2] Secret Result: 0x%x\n\n", res);

    // Phase 3
    printf("========================================\n");
    printf("PHASE 3: Ready for Manipulation\n");
    printf("========================================\n");
    
    printf("Commands:\n");
    printf("  1. alzheimers.exe pte %d %p\n", _getpid(), secret_mem);
    printf("  2. alzheimers.exe vad %d %p\n", _getpid(), secret_mem);
    printf("  3. alzheimers.exe restore %d %p <OLD_VAL>\n", _getpid(), secret_mem);
    
    printf("\n");
    printf("CRITICAL WARNING:\n");
    printf(" - You can Hide (PTE/VAD) and Restore freely while running.\n");
    printf(" - BUT: Do NOT close this window using 'X'.\n");
    printf(" - Closing the process triggers kernel cleanup if you ran PTE -> BSOD(maybe).\n");
    printf(" - Closing the process is safe if you only ran VAD.\n");
    printf(" - Always use 'q' to exit and attempt cleanup.\n");  
    printf("\n");
    printf("Press ENTER to enter Zombie Mode (Safe State)...\n");
    getchar();

    // ---------------------------------------------------------
    // PHASE 4: INTERACTIVE CHECK LOOP
    // ---------------------------------------------------------
    printf("\n[INFO] Zombie Mode Active.\n");
    printf("[INFO] Press 'q' to Stop Loop and Attempt Cleanup.\n");

    int running = 1;
    int counter = 0;

    while (running) {
        if (_kbhit()) {
            char ch = _getch();
            if (ch == 'q' || ch == 'Q') {
                running = 0;
                break; 
            }
        }

        counter++;
        if(work_func) work_func(); 
        if (counter % 5 == 0) {
            if(noise_func) noise_func();
        }

        printf("[CHECK] Status: ");
        __try {
            int val = ((SecretFunc)secret_mem)();
            printf("VISIBLE (Returned 0x%x)\n", val);
        }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            printf("HIDDEN / GONE (Access Violation)\n");
        }
        
        Sleep(1000);
    }

    // ---------------------------------------------------------
    // PHASE 5: VAD REPAIR (CLEANUP)
    // ---------------------------------------------------------
    printf("\n========================================\n");
    printf("[CLEANUP] Attempting VAD Repair for Safe Exit...\n");
    printf("========================================\n");

    void* repair = VirtualAlloc(secret_mem, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (repair == secret_mem) {
        printf("[SUCCESS] VAD Patched! New VAD created at target address.\n");
        printf("[EXIT] Safe to exit. Memory manager satisfied.\n");
    } else {
        printf("[WARNING] Patch Failed (Error %d). VAD likely still exists.\n", GetLastError());
        printf("          If you hid the PTE, make sure you ran RESTORE first.\n");
        printf("[EXIT] Exiting (Risk: Low if VAD exists, High if VAD missing).\n");
    }

    Sleep(1000);
    WSACleanup();
    FreeLibrary(hDll);
    return 0;
}