@echo off
REM Build script for innocent.exe and helper.dll
REM Run this from x64 Native Tools Command Prompt for VS

echo ========================================
echo Building Anti-Forensics Demo
echo ========================================
echo.

REM Step 1: Build helper.dll
echo [1/3] Building helper.dll...
cl /LD /O2 helper.c /Fe:helper.dll /DHELPER_EXPORTS
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to build helper.dll
    pause
    exit /b 1
)
echo [SUCCESS] helper.dll built successfully
echo.

REM Step 2: Build innocent.exe
echo [2/3] Building innocent.exe... (links with ws2_32.lib)
cl /O2 innocent.c ws2_32.lib /Fe:innocent.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to build innocent.exe
    pause
    exit /b 1
)
echo [SUCCESS] innocent.exe built successfully
echo.

REM Step 3: Clean up intermediate files
echo [3/3] Cleaning up...
del *.obj *.exp 2>nul
echo.

echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Files created:
echo   - helper.dll
echo   - helper.lib
echo   - innocent.exe
echo.
echo To run:
echo   1. innocent.exe
echo   2. Note the function addresses
echo   3. Use alzheimers.exe pte to hide dead code
echo.
pause
