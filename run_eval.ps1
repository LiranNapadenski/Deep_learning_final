# run_eval.ps1

$Env:GEMINI_API_KEY = "AIzaSyDOgV5Nw6jhPHxnUvDAD9e1QUqXO2ETIsM"

Write-Host "Creating Virtual Environment (.venv)..."
python -m venv .venv

Write-Host "Activating Virtual Environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies in isolated environment..."
# Upgrade pip first to avoid resolving issues
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Virtual environment is ready!"

Write-Host "Running evaluation: baseline..."
python evaluate.py --system baseline --output baseline_results.csv

Write-Host "Running evaluation: mem0..."
python evaluate.py --system mem0 --output mem0_results.csv

Write-Host "Running evaluation: zep..."
python evaluate.py --system zep --output zep_results.csv
Write-Host ""
Write-Host "To stay in this environment for future commands, leave this terminal open."
Write-Host "Or manually activate it anytime later by running: .\.venv\Scripts\Activate.ps1"
