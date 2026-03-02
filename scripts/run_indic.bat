@echo off
echo Starting Indic Voice Agent (IndicWhisper + Ollama + Indic Parler-TTS)...
echo Requires Ollama running with model pulled. First run downloads ~2.8 GB of models.
python run.py --agent indic
pause
