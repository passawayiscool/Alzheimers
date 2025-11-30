#ifndef HELPER_H
#define HELPER_H

#ifdef HELPER_EXPORTS
#define HELPER_API __declspec(dllexport)
#else
#define HELPER_API __declspec(dllimport)
#endif

// Initialization
HELPER_API void initialize_config(void);
   
// Safe background task (Keeps the process looking busy)
HELPER_API void do_work(void);

// Noise Generator: Simulates C2 traffic and File I/O
// This helps confuse analysts looking for the "real" malicious activity.
HELPER_API void simulate_rootkit_activity(void);

#endif // HELPER_H