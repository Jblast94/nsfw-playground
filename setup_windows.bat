@echo off
echo ========================================
echo NSFW Playground - Windows Setup Script
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Checking pip installation...
pip --version
if %errorlevel% neq 0 (
    echo ERROR: pip not found! Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('No CUDA devices found')"

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To start the application:
echo   python text_generation_api.py
echo.
echo Then open: http://localhost:8000
echo.
pause