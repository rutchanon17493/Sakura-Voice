@echo off
echo ============================================================
echo  Sakura Indic Agent Setup
echo  Fully local: IndicWhisper + Ollama + Indic Parler-TTS
echo ============================================================
echo.

echo [1/4] Installing base requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: pip install -r requirements.txt failed.
    pause & exit /b 1
)
echo.

echo [2/4] Installing PyTorch (CPU-only build)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ERROR: PyTorch install failed.
    pause & exit /b 1
)
echo.

echo [3/4] Installing Parler-TTS (from GitHub)...
pip install git+https://github.com/huggingface/parler-tts.git
if %errorlevel% neq 0 (
    echo ERROR: Parler-TTS install failed. Check your internet connection.
    pause & exit /b 1
)
echo.

echo [4/4] Checking Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama not found on PATH.
    echo   Download and install from: https://ollama.ai/download
    echo   Then run:  ollama pull qwen2.5:7b
    echo.
) else (
    echo Ollama found. Pulling qwen2.5:7b (^~4 GB — this may take a while)...
    ollama pull qwen2.5:7b
)
echo.

echo ============================================================
echo  Setup complete!
echo.
echo  To run the indic agent:
echo    python run.py --agent indic
echo.
echo  NOTE: First run downloads ^~2.8 GB of models from
echo  HuggingFace Hub (IndicWhisper + Parler-TTS mini).
echo  Subsequent runs use the local cache.
echo.
echo  To see all 17 supported languages:
echo    python run.py --list-indic-languages
echo ============================================================
pause
