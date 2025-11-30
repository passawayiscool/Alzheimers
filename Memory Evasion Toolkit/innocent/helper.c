#include "helper.h"
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <time.h>

#pragma comment(lib, "Ws2_32.lib")

// --- Internal Activity Helpers ---

void mimic_network_beacon() {
    // Simulates a "C2 Heartbeat" to Google DNS
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return;

    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr("8.8.8.8"); 
    server.sin_port = htons(53); 

    printf("    [NET] Helper DLL: Sending heartbeat to 8.8.8.8:53...\n");
    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) != SOCKET_ERROR) {
        const char* payload = "HELO_ROOTKIT_V1";
        send(sock, payload, (int)strlen(payload), 0);
    }
    closesocket(sock);
}

void mimic_file_staging() {
    // Simulates writing stolen data to temp
    char tempPath[MAX_PATH];
    char filePath[MAX_PATH];
    
    GetTempPath(MAX_PATH, tempPath);
    sprintf_s(filePath, MAX_PATH, "%sstage_data.log", tempPath);

    printf("    [FS]  Helper DLL: Touching file %s\n", filePath);

    FILE* f = fopen(filePath, "a");
    if (f) {
        fprintf(f, "[%lld] LOG_ENTRY: System Check OK\n", (long long)time(NULL));
        fclose(f);
    }
}

// --- Exported Functions ---

void initialize_config(void) {
    printf("[HELPER] DLL Loaded & Initialized.\n");
}

void do_work(void) {
    // Standard "safe" heartbeat
    // This runs frequently to keep the thread stack moving
    // printf("[HELPER] Safe worker thread active.\n"); 
}

void simulate_rootkit_activity(void) {
    // This is called by innocent.c to create noise
    printf("[ACTIVITY] Helper DLL generating camouflage traffic...\n");
    mimic_file_staging();
    mimic_network_beacon();
    printf("[ACTIVITY] Noise generation complete.\n");
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
    return TRUE;
}