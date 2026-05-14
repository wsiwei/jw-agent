@echo off
echo ========================================
echo Agent Interactive Chat Client
echo ========================================
echo.

REM Check if service is running
curl -s http://localhost:9527/health >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Service may not be running
    echo Please start the service first by running: start.bat
    echo.
    set /p continue="Continue anyway? (Y/N): "
    if /i not "%continue%"=="Y" exit /b 1
)

echo [INFO] Starting chat client...
echo.

python chat_client.py

pause
