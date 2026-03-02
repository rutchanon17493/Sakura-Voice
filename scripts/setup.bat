@echo off
echo Installing Sakura dependencies...
pip install -r requirements.txt
echo.
echo Installing PyAudio...
pip install pyaudio
echo.
echo Setup complete. Run:
echo   python run.py                  -- local agent
echo   python run.py --agent sarvam   -- Sarvam AI agent (needs SARVAM_API_KEY)
echo   python run.py --agent indic    -- fully local Indic agent (run setup_indic.bat first)
pause
